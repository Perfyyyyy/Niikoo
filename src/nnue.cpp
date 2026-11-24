#include "nnue.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <new>
#ifdef __linux__
#include <sys/mman.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define NNUE_X86 1
// AVX-VNNI (VEX-encoded vpdpbusd) intrinsics need GCC >= 11 or clang >= 12; older compilers just skip that path.
#if (defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || (defined(__clang__) && __clang_major__ >= 12)
#define NNUE_AVXVNNI 1
#endif
#endif

// Net file layout, v3 (little endian), written by nnue_tools/nnue_format.py:
//   char    magic[8] = "CHSNNUE3"
//   uint32  L1 (hidden), int32 QA, int32 QW, int32 SCALE, uint32 kingBuckets, outputBuckets, l2, l3
//   int16   ftWeights[kingBuckets * 768][L1]     (real weight * QA)
//   int16   ftBias[L1]                           (real bias * QA)
//   int16   l2Weights[outputBuckets][l2][L1]     (real weight * QW)
//   float   l2Bias[outputBuckets][l2]
//   float   l3Weights[outputBuckets][l3][l2], l3Bias[outputBuckets][l3]
//   float   outWeights[outputBuckets][l3], outBias[outputBuckets]
//
// Inference, for output bucket b:
//   c = clamp(acc, 0, QA);  p = (c[i] * c[i + L1/2] + 128) >> 8      (pairwise per perspective: side to move first, then the other)
//   z[j] = sum_i p[i] * l2W[b][j][i]            (int32, exact)
//   h2 = clamp(z / (PSCALE * QW) + l2B, 0, 1),  PSCALE = QA*QA/256       (float from here on)
//   h3 = clamp(l3W * h2 + l3B, 0, 1);  out = outW . h3 + outB;  score = trunc(out * SCALE)
//
// Speed-oriented changes relative to the first v3 engine code (the net file format is unchanged):
//  * l2 weights are held as int8 in memory. At load every row (one output neuron of one bucket) is rescaled so that
//    max|w| <= 64 (rows that already fit are kept exactly) and rounded; the per-row factor goes into s2[bucket][j].
//    |w| <= 64 and inputs <= 254 guarantee that the int16 pair sums of maddubs never saturate, so the integer sums
//    are exact for the int8 weights, and the plain and AVX2 code agree bit for bit.
//  * the dense layer has VNNI paths: vpdpbusd does u8 x s8 -> int32 accumulate in one instruction (replacing
//    maddubs + madd + add). AVX-VNNI is preferred, AVX512-VNNI (256-bit, needs AVX512VL) is the fallback, otherwise the AVX2
//    path runs. All three add the same exact int32 products, so the integer sums are identical.
//  * prefetchFeature() pulls a first-layer weight row (~1.5 KB at L1 = 768) towards the cache ahead of the update.
//  * l3 weights are stored transposed ([bucket][l2][l3]) so the AVX2 tail is 16 broadcast multiply-adds over l3 lanes.
//    The output dot product has a fixed summation order (8 lane partial sums, then a fixed tree) that the plain code
//    reproduces exactly; no FMA is used, so plain and AVX2 float tails give identical results.

namespace nnue {

namespace{

// 2 MB aligned storage for the big first-layer table: rows start on cache lines, and on linux the table can sit
// on huge pages (rows are read at random, with 4 KB pages that is a TLB miss per row)
template <class T> struct BigAlloc {
    using value_type = T;
#ifdef __linux__
    static constexpr size_t ALIGN = 2u << 20;
#else
    static constexpr size_t ALIGN = 64;
#endif
    BigAlloc() = default;
    template <class U> BigAlloc(const BigAlloc<U>&) {}
    T* allocate(size_t n) {
        const size_t sz = (n * sizeof(T) + ALIGN - 1) / ALIGN * ALIGN;
        void* p = ::operator new(sz, std::align_val_t(ALIGN));
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        madvise(p, sz, MADV_HUGEPAGE);
#endif
        return (T*)p;
    }
    void deallocate(T* p, size_t) { ::operator delete(p, std::align_val_t(ALIGN)); }
};
template <class T, class U> bool operator==(const BigAlloc<T>&, const BigAlloc<U>&) { return true; }
template <class T, class U> bool operator!=(const BigAlloc<T>&, const BigAlloc<U>&) { return false; }
} // namespace

namespace {
struct Net {
    int hidden = 0, qa = 255, qw = 512, scale = 400;
    std::vector<int16_t BigAlloc<int16_t>> ftW;
    std::vector<int16_t> ftB;
    std::vector<int8_t>  l2W8;                      // [bucket][L2][L1], rows rescaled to max|w| <= 64
    std::vector<float>   s2;                        // [bucket][L2]: 1 / (PSCALE * QW * rowScale)
    std::vector<float>   l2B, l3WT, l3B, outW, outB;   // l3WT is [bucket][L2][L3] (transposed)
};
Net g_net;
bool g_loaded = false;
uint32_t g_netId = 0;        // bumped on every load / unload
bool g_avx2 = false;         // this CPU can run the AVX2 paths (checked once, at load)
bool g_simdAllowed = true;   // tests can switch the SIMD paths off to compare against the plain C++ code
int  g_vnni = 0;             // 0 none, 1 AVX-VNNI, 2 AVX512-VNNI (+VL); set at load
bool g_vnniAllowed = true;   // tests can switch VNNI off to compare against the AVX2 path

bool fail(std::string* err, const char* msg) { if (err) *err = msg; return false; }

// reads straight from a memory block (a file that was loaded into memory, or the net baked into the binary)
struct Cursor {
    const uint8_t* p;
    size_t left;
    bool read(void* dst, size_t bytes) {
        if (bytes > left) return false;
        std::memcpy(dst, p, bytes);
        p += bytes;
        left -= bytes;
        return true;
    }
};

template <class V>
bool readVec(Cursor& c, V& v, size_t n) {
    using T = typename V::value_type;
    if (n > c.left / sizeof(T)) return false;
    v.resize(n);
    return c.read(v.data(), n * sizeof(T));
}
}  // namespace

// the one and only parser, load() and the embedded net both go through here
bool loadFromMemory(const void* data, size_t size, std::string* err) {
    Cursor c{(const uint8_t*)data, size};
    char magic[8];
    if (!c.read(magic, 8)) return fail(err, "bad net header");
    if (std::memcmp(magic, "CHSNNUE1", 8) == 0 || std::memcmp(magic, "CHSNNUE2", 8) == 0)
        return fail(err, "this is an old (v1/v2) net; the engine needs a v3 net, retrain with the new train.py");
    uint32_t hidden = 0, kb = 0, ob = 0, l2 = 0, l3 = 0; int32_t qa = 0, qw = 0, scale = 0;
    if (std::memcmp(magic, "CHSNNUE3", 8) != 0 || !c.read(&hidden, 4) || !c.read(&qa, 4) || !c.read(&qw, 4) ||
        !c.read(&scale, 4) || !c.read(&kb, 4) || !c.read(&ob, 4) || !c.read(&l2, 4) || !c.read(&l3, 4))
        return fail(err, "bad net header");
    if (hidden == 0 || hidden > (uint32_t)MAX_HIDDEN || (hidden % 64) || qa <= 0 || qa > 255 || qw <= 0 || scale <= 0 ||
        kb != (uint32_t)KING_BUCKETS || ob != (uint32_t)OUTPUT_BUCKETS || l2 != (uint32_t)L2 || l3 != (uint32_t)L3)
        return fail(err, "unsupported net parameters");
    Net n;                                          // read into a temporary so a bad file never clobbers a working net
    n.hidden = (int)hidden; n.qa = qa; n.qw = qw; n.scale = scale;
    const size_t H = hidden;
    std::vector<int16_t> l2W16;
    std::vector<float> l3W;
    bool ok = readVec(c, n.ftW, (size_t)KING_BUCKETS * NUM_FEATURES * H) && readVec(c, n.ftB, H) &&
              readVec(c, l2W16, (size_t)OUTPUT_BUCKETS * L2 * H) && readVec(c, n.l2B, (size_t)OUTPUT_BUCKETS * L2) &&
              readVec(c, l3W, (size_t)OUTPUT_BUCKETS * L3 * L2) && readVec(c, n.l3B, (size_t)OUTPUT_BUCKETS * L3) &&
              readVec(c, n.outW, (size_t)OUTPUT_BUCKETS * L3) && readVec(c, n.outB, (size_t)OUTPUT_BUCKETS);
    if (!ok || c.left != 0) return fail(err, "net file has the wrong size");   // too short, or trailing bytes (wrong file)

    // int8 requantisation of the first dense layer, per row. |w8| <= 64 keeps maddubs (u8 <= 254 times s8, two summed
    // into int16) below 2 * 254 * 64 = 32512 < 32767, and the int32 row sums far below 2^31.
    n.l2W8.resize(l2W16.size());
    n.s2.resize((size_t)OUTPUT_BUCKETS * L2);
    for (size_t row = 0; row < (size_t)OUTPUT_BUCKETS * L2; ++row) {
        int rowMax = 0;
        for (size_t i = 0; i < H; ++i) rowMax = std::max(rowMax, std::abs((int)l2W16[row * H + i]));
        const double s = rowMax <= 64 ? 1.0 : 64.0 / rowMax;
        for (size_t i = 0; i < H; ++i) {
            long v = std::lround((double)l2W16[row * H + i] * s);
            n.l2W8[row * H + i] = (int8_t)std::max(-64L, std::min(64L, v));
        }
        n.s2[row] = (float)(1.0 / ((double)qa * qa / 256.0 * qw * s));
    }
    // transpose l3 weights: file order is [bucket][k < L3][j < L2], the engine wants [bucket][j][k]
    n.l3WT.resize(l3W.size());
    for (size_t b = 0; b < (size_t)OUTPUT_BUCKETS; ++b)
        for (size_t k = 0; k < (size_t)L3; ++k)
            for (size_t j = 0; j < (size_t)L2; ++j) n.l3WT[(b * L2 + j) * L3 + k] = l3W[(b * L3 + k) * L2 + j];

    g_net = std::move(n);
    ++g_netId;
#ifdef NNUE_X86
    __builtin_cpu_init();
    g_avx2 = __builtin_cpu_supports("avx2");
    g_vnni = 0;
    if (g_avx2) {
#ifdef NNUE_AVXVNNI
        unsigned a = 0, b = 0, c = 0, d = 0;            // CPUID.(EAX=7,ECX=1):EAX bit 4 = AVX-VNNI (YMM state is covered by the avx2 check)
        if (__get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4))) g_vnni = 1;
#endif
        if (g_vnni == 0 && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vl")) g_vnni = 2;
    }
#endif
    g_loaded = true;
    return true;
}

bool load(const std::string& path, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(err, "cannot open net file");
    std::vector<uint8_t> buf;
    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(f) : -1;
    ok = ok && size >= 0 && std::fseek(f, 0, SEEK_SET) == 0;
    if (ok) {
        buf.resize((size_t)size);
        ok = std::fread(buf.data(), 1, buf.size(), f) == buf.size();
    }
    std::fclose(f);
    if (!ok) return fail(err, "cannot read net file");
    return loadFromMemory(buf.data(), buf.size(), err);
}

void unload()        { g_loaded = false; ++g_netId; }
bool loaded()        { return g_loaded; }
int  hiddenSize()    { return g_net.hidden; }

void resetPerspective(Accumulator& acc, int perspective) {
    std::memcpy(acc.v[perspective], g_net.ftB.data(), (size_t)g_net.hidden * sizeof(int16_t));
}

void reset(Accumulator& acc) {
    resetPerspective(acc, 0);
    resetPerspective(acc, 1);
}

void copy(Accumulator& dst, const Accumulator& src) {
    for (int p = 0; p < 2; ++p)
        std::memcpy(dst.v[p], src.v[p], (size_t)g_net.hidden * sizeof(int16_t));
}

void copyPerspective(Accumulator& dst, const Accumulator& src, int perspective) {
    std::memcpy(dst.v[perspective], src.v[perspective], (size_t)g_net.hidden * sizeof(int16_t));
}

#ifdef NNUE_X86
__attribute__((target("avx2"))) static void addAVX2(int16_t* a, const int16_t* w, int n) {
    for (int i = 0; i < n; i += 16)
        _mm256_storeu_si256((__m256i*)(a + i), _mm256_add_epi16(_mm256_loadu_si256((const __m256i*)(a + i)),
                                                                _mm256_loadu_si256((const __m256i*)(w + i))));
}
__attribute__((target("avx2"))) static void subAVX2(int16_t* a, const int16_t* w, int n) {
    for (int i = 0; i < n; i += 16)
        _mm256_storeu_si256((__m256i*)(a + i), _mm256_sub_epi16(_mm256_loadu_si256((const __m256i*)(a + i)),
                                                                _mm256_loadu_si256((const __m256i*)(w + i))));
}
#endif

static inline const int16_t* featureRow(int feature) { return g_net.ftW.data() + (size_t)feature * g_net.hidden; }

void addFeatureRaw(int16_t* a, int feature) {
    const int H = g_net.hidden;
    const int16_t* w = featureRow(feature);
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) { addAVX2(a, w, H); return; }       // H is a multiple of 64 (checked at load)
#endif
    for (int i = 0; i < H; ++i) a[i] = (int16_t)(a[i] + w[i]);
}

void subFeatureRaw(int16_t* a, int feature) {
    const int H = g_net.hidden;
    const int16_t* w = featureRow(feature);
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) { subAVX2(a, w, H); return; }
#endif
    for (int i = 0; i < H; ++i) a[i] = (int16_t)(a[i] - w[i]);
}

void addFeature(Accumulator& acc, int perspective, int feature) { addFeatureRaw(acc.v[perspective], feature); }
void subFeature(Accumulator& acc, int perspective, int feature) { subFeatureRaw(acc.v[perspective], feature); }

uint32_t netId() { return g_netId; }
void loadBias(int16_t* dst) { std::memcpy(dst, g_net.ftB.data(), (size_t)g_net.hidden * sizeof(int16_t)); }

#ifdef NNUE_X86
// dst = src - sum(sw[0..NS)) + sum(aw[0..NA)) in one pass; NS and NA are compile-time so the inner loops unroll.
template <int NS, int NA>
__attribute__((target("avx2"))) static void updateAVX2(int16_t* dst, const int16_t* src, const int16_t* const* sw,
                                                       const int16_t* const* aw, int n) {
    for (int i = 0; i < n; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
        for (int s = 0; s < NS; ++s) v = _mm256_sub_epi16(v, _mm256_loadu_si256((const __m256i*)(sw[s] + i)));
        for (int a = 0; a < NA; ++a) v = _mm256_add_epi16(v, _mm256_loadu_si256((const __m256i*)(aw[a] + i)));
        _mm256_storeu_si256((__m256i*)(dst + i), v);
    }
}
#endif

void updatePerspective(int16_t* dst, const int16_t* src, const int* subs, int ns, const int* adds, int na) {
    const int H = g_net.hidden;
    const int16_t* sw[4];
    const int16_t* aw[4];
    for (int i = 0; i < ns; ++i) sw[i] = featureRow(subs[i]);
    for (int i = 0; i < na; ++i) aw[i] = featureRow(adds[i]);
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) {
        switch (ns * 5 + na) {                                       // the shapes chess moves produce, then a generic case
            case 0:        if (dst != src) std::memcpy(dst, src, (size_t)H * sizeof(int16_t)); return;   // null move
            case 1 * 5 + 1: updateAVX2<1, 1>(dst, src, sw, aw, H); return;                               // quiet move, promotion
            case 2 * 5 + 1: updateAVX2<2, 1>(dst, src, sw, aw, H); return;                               // capture, en passant
            case 2 * 5 + 2: updateAVX2<2, 2>(dst, src, sw, aw, H); return;                               // castling
            default: break;
        }
    }
#endif
    for (int i = 0; i < H; ++i) {
        int v = src[i];
        for (int s = 0; s < ns; ++s) v -= sw[s][i];
        for (int a = 0; a < na; ++a) v += aw[a][i];
        dst[i] = (int16_t)v;
    }
}

void setSimd(bool on) { g_simdAllowed = on; }
bool simdActive()     { return g_avx2 && g_simdAllowed; }

void prefetchFeature(int feature) {
    if (!g_loaded) return;
    const char* p = (const char*)featureRow(feature);
    const char* end = p + (size_t)g_net.hidden * sizeof(int16_t);
    for (; p < end; p += 64) __builtin_prefetch(p, 0, 3);   // one hint per cache line of the row
    __builtin_prefetch(end - 1, 0, 3);                      // last line, in case the row is not 64-byte aligned
}

void setVnni(bool on) { g_vnniAllowed = on; }
const char* simdName() {
    if (!(g_avx2 && g_simdAllowed)) return "plain";
    if (g_vnniAllowed && g_vnni == 1) return "avx-vnni";
    if (g_vnniAllowed && g_vnni == 2) return "avx512-vnni";
    return "avx2";
}

// ---- evaluation ----
static_assert(L2 % 4 == 0 && L3 % 8 == 0, "AVX2 tail assumes L2 % 4 == 0 and L3 % 8 == 0");

// Integer part, plain: pairwise product, then 16 int8 dot products. Exact, identical to the AVX2 code.
static void denseIntPlain(const Accumulator& acc, int stm, int bucket, int32_t z[L2]) {
    const Net& n = g_net;
    const int H = n.hidden, half = H / 2, qa = n.qa;
    uint8_t x[MAX_HIDDEN];
    for (int side = 0; side < 2; ++side) {
        const int16_t* a = acc.v[side == 0 ? stm : (stm ^ 1)];
        uint8_t* dst = x + side * half;
        for (int i = 0; i < half; ++i) {
            const int c0 = std::max(0, std::min((int)a[i], qa));
            const int c1 = std::max(0, std::min((int)a[i + half], qa));
            dst[i] = (uint8_t)((c0 * c1 + 128) >> 8);
        }
    }
    const int8_t* w2 = n.l2W8.data() + (size_t)bucket * L2 * H;
    for (int j = 0; j < L2; ++j) {
        const int8_t* wj = w2 + (size_t)j * H;
        int32_t sum = 0;
        for (int i = 0; i < H; ++i) sum += (int32_t)x[i] * (int32_t)wj[i];
        z[j] = sum;
    }
}

// Float tail, plain. Same operations in the same order as the AVX2 tail (no FMA, fixed reduction tree).
static float tailPlain(const int32_t z[L2], int bucket) {
    const Net& n = g_net;
    float h2[L2];
    for (int j = 0; j < L2; ++j) {
        const float v = (float)z[j] * n.s2[(size_t)bucket * L2 + j] + n.l2B[(size_t)bucket * L2 + j];
        h2[j] = std::max(0.f, std::min(1.f, v));
    }
    float h3[L3];
    for (int k = 0; k < L3; ++k) h3[k] = n.l3B[(size_t)bucket * L3 + k];
    const float* w3 = n.l3WT.data() + (size_t)bucket * L2 * L3;
    for (int j = 0; j < L2; ++j)
        for (int k = 0; k < L3; ++k) h3[k] = h3[k] + w3[j * L3 + k] * h2[j];
    for (int k = 0; k < L3; ++k) h3[k] = std::max(0.f, std::min(1.f, h3[k]));
    const float* wo = n.outW.data() + (size_t)bucket * L3;
    float t[8];
    for (int l = 0; l < 8; ++l) {
        t[l] = wo[l] * h3[l];
        for (int c = 1; c < L3 / 8; ++c) t[l] = t[l] + wo[c * 8 + l] * h3[c * 8 + l];
    }
    const float u0 = t[0] + t[4], u1 = t[1] + t[5], u2 = t[2] + t[6], u3 = t[3] + t[7];
    const float total = (u0 + u2) + (u1 + u3);
    return n.outB[bucket] + total;
}

#ifdef NNUE_X86
// Pairwise product packed to u8 (shared by the AVX2 and VNNI dense paths). x must be 32-byte aligned.
__attribute__((always_inline, target("avx2"))) static inline void packPairwiseAVX2(const Accumulator& acc, int stm, uint8_t* x) {
    const Net& n = g_net;
    const int H = n.hidden, half = H / 2;
    const __m256i zero = _mm256_setzero_si256(), qmax = _mm256_set1_epi16((short)n.qa), rnd = _mm256_set1_epi16(128);
    for (int side = 0; side < 2; ++side) {
        const int16_t* a = acc.v[side == 0 ? stm : (stm ^ 1)];
        uint8_t* dst = x + side * half;
        for (int i = 0; i < half; i += 32) {
            __m256i p[2];
            for (int h = 0; h < 2; ++h) {
                const __m256i c0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(a + i + 16 * h)), zero), qmax);
                const __m256i c1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(a + i + 16 * h + half)), zero), qmax);
                // c0 * c1 + 128 <= 65153 fits 16 bits unsigned; the logical shift gives (c0*c1 + 128) >> 8 exactly (<= 254)
                p[h] = _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(c0, c1), rnd), 8);
            }
            // packus works within 128-bit lanes (order a0-7 b0-7 a8-15 b8-15); the permute restores a0-15 b0-15
            _mm256_store_si256((__m256i*)(dst + i), _mm256_permute4x64_epi64(_mm256_packus_epi16(p[0], p[1]), 0xD8));
        }
    }
}

// Integer part, AVX2: maddubs(x_u8, w_s8) -> int16 pair sums -> madd(.,1) -> int32.
// Four output rows are processed together so every x load is used four times.
__attribute__((target("avx2"))) static void denseIntAVX2(const Accumulator& acc, int stm, int bucket, int32_t z[L2]) {
    const Net& n = g_net;
    const int H = n.hidden;
    alignas(32) uint8_t x[MAX_HIDDEN];
    packPairwiseAVX2(acc, stm, x);
    const __m256i zero = _mm256_setzero_si256(), ones = _mm256_set1_epi16(1);
    const int8_t* w2 = n.l2W8.data() + (size_t)bucket * L2 * H;
    for (int j = 0; j < L2; j += 4) {
        const int8_t* w0 = w2 + (size_t)j * H;
        __m256i s0 = zero, s1 = zero, s2 = zero, s3 = zero;
        for (int i = 0; i < H; i += 32) {
            const __m256i xv = _mm256_load_si256((const __m256i*)(x + i));
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((const __m256i*)(w0 + i))), ones));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((const __m256i*)(w0 + H + i))), ones));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((const __m256i*)(w0 + 2 * H + i))), ones));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_maddubs_epi16(xv, _mm256_loadu_si256((const __m256i*)(w0 + 3 * H + i))), ones));
        }
        // horizontal sums of the four accumulators in one go: result lanes = (s0, s1, s2, s3)
        const __m256i h01 = _mm256_hadd_epi32(s0, s1), h23 = _mm256_hadd_epi32(s2, s3);
        const __m256i h   = _mm256_hadd_epi32(h01, h23);
        _mm_storeu_si128((__m128i*)(z + j), _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));
    }
}

// Integer part, VNNI: one vpdpbusd per 32 inputs and row (u8 x s8, four products summed straight into int32 lanes).
// DPBUSD(acc, u8, s8) is the intrinsic of the respective instruction set. No int16 intermediate, so no saturation question at all.
#define NNUE_DENSE_VNNI_BODY(DPBUSD)                                                                                    \
    const Net& n = g_net;                                                                                               \
    const int H = n.hidden;                                                                                             \
    alignas(32) uint8_t x[MAX_HIDDEN];                                                                                  \
    packPairwiseAVX2(acc, stm, x);                                                                                      \
    const __m256i zero = _mm256_setzero_si256();                                                                        \
    const int8_t* w2 = n.l2W8.data() + (size_t)bucket * L2 * H;                                                         \
    for (int j = 0; j < L2; j += 4) {                                                                                   \
        const int8_t* w0 = w2 + (size_t)j * H;                                                                          \
        __m256i s0 = zero, s1 = zero, s2 = zero, s3 = zero;                                                             \
        for (int i = 0; i < H; i += 32) {                                                                               \
            const __m256i xv = _mm256_load_si256((const __m256i*)(x + i));                                              \
            s0 = DPBUSD(s0, xv, _mm256_loadu_si256((const __m256i*)(w0 + i)));                                          \
            s1 = DPBUSD(s1, xv, _mm256_loadu_si256((const __m256i*)(w0 + H + i)));                                      \
            s2 = DPBUSD(s2, xv, _mm256_loadu_si256((const __m256i*)(w0 + 2 * H + i)));                                  \
            s3 = DPBUSD(s3, xv, _mm256_loadu_si256((const __m256i*)(w0 + 3 * H + i)));                                  \
        }                                                                                                               \
        const __m256i h01 = _mm256_hadd_epi32(s0, s1), h23 = _mm256_hadd_epi32(s2, s3);                                 \
        const __m256i h   = _mm256_hadd_epi32(h01, h23);                                                                \
        _mm_storeu_si128((__m128i*)(z + j), _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1)));  \
    }

#ifdef NNUE_AVXVNNI
__attribute__((target("avx2,avxvnni"))) static void denseIntAVXVNNI(const Accumulator& acc, int stm, int bucket, int32_t z[L2]) {
    NNUE_DENSE_VNNI_BODY(_mm256_dpbusd_avx_epi32)
}
#endif
__attribute__((target("avx2,avx512f,avx512vl,avx512vnni"))) static void denseIntAVX512VNNI(const Accumulator& acc, int stm, int bucket, int32_t z[L2]) {
    NNUE_DENSE_VNNI_BODY(_mm256_dpbusd_epi32)
}
#undef NNUE_DENSE_VNNI_BODY

// Float tail, AVX2 (mul + add, never FMA, so the results equal tailPlain exactly).
__attribute__((target("avx2"))) static float tailAVX2(const int32_t z[L2], int bucket) {
    const Net& n = g_net;
    const __m256 zero = _mm256_setzero_ps(), one = _mm256_set1_ps(1.f);
    alignas(32) float h2[L2];
    for (int j = 0; j < L2; j += 8) {
        __m256 v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i*)(z + j))),
                                               _mm256_loadu_ps(n.s2.data() + (size_t)bucket * L2 + j)),
                                 _mm256_loadu_ps(n.l2B.data() + (size_t)bucket * L2 + j));
        _mm256_store_ps(h2 + j, _mm256_min_ps(_mm256_max_ps(v, zero), one));
    }
    __m256 h3[L3 / 8];
    for (int c = 0; c < L3 / 8; ++c) h3[c] = _mm256_loadu_ps(n.l3B.data() + (size_t)bucket * L3 + c * 8);
    const float* w3 = n.l3WT.data() + (size_t)bucket * L2 * L3;
    for (int j = 0; j < L2; ++j) {
        const __m256 hj = _mm256_broadcast_ss(h2 + j);
        for (int c = 0; c < L3 / 8; ++c)
            h3[c] = _mm256_add_ps(h3[c], _mm256_mul_ps(hj, _mm256_loadu_ps(w3 + j * L3 + c * 8)));
    }
    const float* wo = n.outW.data() + (size_t)bucket * L3;
    __m256 t = zero;
    for (int c = 0; c < L3 / 8; ++c) {
        const __m256 prod = _mm256_mul_ps(_mm256_loadu_ps(wo + c * 8), _mm256_min_ps(_mm256_max_ps(h3[c], zero), one));
        t = c == 0 ? prod : _mm256_add_ps(t, prod);
    }
    __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));   // u[m] = t[m] + t[m+4]
    u = _mm_add_ps(u, _mm_movehl_ps(u, u));                                          // (u0+u2, u1+u3, ...)
    u = _mm_add_ss(u, _mm_shuffle_ps(u, u, 1));                                      // (u0+u2) + (u1+u3)
    return n.outB[bucket] + _mm_cvtss_f32(u);
}
#endif

int evaluate(const Accumulator& acc, int stm, int bucket) {
    bucket = std::max(0, std::min(bucket, OUTPUT_BUCKETS - 1));
    alignas(32) int32_t z[L2];
    float o;
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) {
#ifdef NNUE_AVXVNNI
        if (g_vnniAllowed && g_vnni == 1) denseIntAVXVNNI(acc, stm, bucket, z); else
#endif
        if (g_vnniAllowed && g_vnni == 2) denseIntAVX512VNNI(acc, stm, bucket, z);
        else denseIntAVX2(acc, stm, bucket, z);
        o = tailAVX2(z, bucket);
    } else
#endif
    {
        denseIntPlain(acc, stm, bucket, z);
        o = tailPlain(z, bucket);
    }
    return (int)(o * (float)g_net.scale);               // truncates toward zero
}

}  // namespace nnue
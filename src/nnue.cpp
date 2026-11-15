#include "nnue.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define NNUE_X86 1
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
// The integer part is exact and identical in the plain and the AVX2 code; the float tail is shared code.

namespace nnue {

namespace {
struct Net {
    int hidden = 0, qa = 255, qw = 512, scale = 400;
    float s2 = 0.f;                                 // 1 / (PSCALE * QW)
    std::vector<int16_t> ftW, ftB, l2W;
    std::vector<float> l2B, l3W, l3B, outW, outB;
};
Net g_net;
bool g_loaded = false;
bool g_avx2 = false;         // this CPU can run the AVX2 paths (checked once, at load)
bool g_simdAllowed = true;   // tests can switch the SIMD paths off to compare against the plain C++ code

bool fail(std::string* err, const char* msg) { if (err) *err = msg; return false; }

template <class T>
bool readVec(FILE* f, std::vector<T>& v, size_t n) {
    v.resize(n);
    return std::fread(v.data(), sizeof(T), n, f) == n;
}
}  // namespace

bool load(const std::string& path, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail(err, "cannot open net file");
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8) { std::fclose(f); return fail(err, "bad net header"); }
    if (std::memcmp(magic, "CHSNNUE1", 8) == 0 || std::memcmp(magic, "CHSNNUE2", 8) == 0) {
        std::fclose(f);
        return fail(err, "this is an old (v1/v2) net; the engine needs a v3 net, retrain with the new train.py");
    }
    uint32_t hidden = 0, kb = 0, ob = 0, l2 = 0, l3 = 0; int32_t qa = 0, qw = 0, scale = 0;
    if (std::memcmp(magic, "CHSNNUE3", 8) != 0 || std::fread(&hidden, 4, 1, f) != 1 || std::fread(&qa, 4, 1, f) != 1 ||
        std::fread(&qw, 4, 1, f) != 1 || std::fread(&scale, 4, 1, f) != 1 || std::fread(&kb, 4, 1, f) != 1 ||
        std::fread(&ob, 4, 1, f) != 1 || std::fread(&l2, 4, 1, f) != 1 || std::fread(&l3, 4, 1, f) != 1) {
        std::fclose(f);
        return fail(err, "bad net header");
    }
    if (hidden == 0 || hidden > (uint32_t)MAX_HIDDEN || (hidden % 32) || qa <= 0 || qa > 255 || qw <= 0 || scale <= 0 ||
        kb != (uint32_t)KING_BUCKETS || ob != (uint32_t)OUTPUT_BUCKETS || l2 != (uint32_t)L2 || l3 != (uint32_t)L3) {
        std::fclose(f);
        return fail(err, "unsupported net parameters");
    }
    Net n;                                          // read into a temporary so a bad file never clobbers a working net
    n.hidden = (int)hidden; n.qa = qa; n.qw = qw; n.scale = scale;
    n.s2 = (float)(1.0 / ((double)qa * qa / 256.0 * qw));
    const size_t H = hidden;
    bool ok = readVec(f, n.ftW, (size_t)KING_BUCKETS * NUM_FEATURES * H) && readVec(f, n.ftB, H) &&
              readVec(f, n.l2W, (size_t)OUTPUT_BUCKETS * L2 * H) && readVec(f, n.l2B, (size_t)OUTPUT_BUCKETS * L2) &&
              readVec(f, n.l3W, (size_t)OUTPUT_BUCKETS * L3 * L2) && readVec(f, n.l3B, (size_t)OUTPUT_BUCKETS * L3) &&
              readVec(f, n.outW, (size_t)OUTPUT_BUCKETS * L3) && readVec(f, n.outB, (size_t)OUTPUT_BUCKETS);
    const bool extra = ok && std::fgetc(f) != EOF;      // trailing bytes: wrong file
    std::fclose(f);
    if (!ok || extra) return fail(err, "net file has the wrong size");

    // The dense sums must not overflow int32 (then plain and SIMD code are exactly equal). Inputs are <= 254.
    int maxW = 0;
    for (int16_t w : n.l2W) maxW = std::max(maxW, std::abs((int)w));
    if ((int64_t)H * 254 * maxW >= ((int64_t)1 << 31)) return fail(err, "net weights are too large for the integer layer");

    g_net = std::move(n);
#ifdef NNUE_X86
    __builtin_cpu_init();
    g_avx2 = __builtin_cpu_supports("avx2");
#endif
    g_loaded = true;
    return true;
}

void unload()        { g_loaded = false; }
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

void addFeature(Accumulator& acc, int perspective, int feature) {
    const int H = g_net.hidden;
    const int16_t* w = g_net.ftW.data() + (size_t)feature * H;
    int16_t* a = acc.v[perspective];
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) { addAVX2(a, w, H); return; }       // H is a multiple of 32 (checked at load)
#endif
    for (int i = 0; i < H; ++i) a[i] = (int16_t)(a[i] + w[i]);
}

void subFeature(Accumulator& acc, int perspective, int feature) {
    const int H = g_net.hidden;
    const int16_t* w = g_net.ftW.data() + (size_t)feature * H;
    int16_t* a = acc.v[perspective];
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) { subAVX2(a, w, H); return; }
#endif
    for (int i = 0; i < H; ++i) a[i] = (int16_t)(a[i] - w[i]);
}

void setSimd(bool on) { g_simdAllowed = on; }
bool simdActive()     { return g_avx2 && g_simdAllowed; }

// ---- evaluation: integer part (pairwise product + first dense layer), plain and AVX2 ----
static void denseIntPlain(const Accumulator& acc, int stm, int bucket, int32_t z[L2]) {
    const Net& n = g_net;
    const int H = n.hidden, half = H / 2, qa = n.qa;
    int16_t x[MAX_HIDDEN];
    for (int side = 0; side < 2; ++side) {
        const int16_t* a = acc.v[side == 0 ? stm : (stm ^ 1)];
        int16_t* dst = x + side * half;
        for (int i = 0; i < half; ++i) {
            const int c0 = std::max(0, std::min((int)a[i], qa));
            const int c1 = std::max(0, std::min((int)a[i + half], qa));
            dst[i] = (int16_t)((c0 * c1 + 128) >> 8);
        }
    }
    const int16_t* w2 = n.l2W.data() + (size_t)bucket * L2 * H;
    for (int j = 0; j < L2; ++j) {
        const int16_t* wj = w2 + (size_t)j * H;
        int32_t sum = 0;
        for (int i = 0; i < H; ++i) sum += (int32_t)x[i] * (int32_t)wj[i];
        z[j] = sum;
    }
}

#ifdef NNUE_X86
__attribute__((target("avx2"))) static void denseIntAVX2(const Accumulator& acc, int stm, int bucket, int32_t z[L2]) {
    const Net& n = g_net;
    const int H = n.hidden, half = H / 2;
    alignas(32) int16_t x[MAX_HIDDEN];
    const __m256i zero = _mm256_setzero_si256(), qmax = _mm256_set1_epi16((short)n.qa), rnd = _mm256_set1_epi16(128);
    for (int side = 0; side < 2; ++side) {
        const int16_t* a = acc.v[side == 0 ? stm : (stm ^ 1)];
        int16_t* dst = x + side * half;
        for (int i = 0; i < half; i += 16) {
            const __m256i c0 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(a + i)), zero), qmax);
            const __m256i c1 = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256((const __m256i*)(a + i + half)), zero), qmax);
            // c0 * c1 <= 255 * 255 and + 128 both fit in 16 bits unsigned; the logical shift gives (c0*c1 + 128) >> 8 exactly
            _mm256_store_si256((__m256i*)(dst + i), _mm256_srli_epi16(_mm256_add_epi16(_mm256_mullo_epi16(c0, c1), rnd), 8));
        }
    }
    const int16_t* w2 = n.l2W.data() + (size_t)bucket * L2 * H;
    for (int j = 0; j < L2; ++j) {
        const int16_t* wj = w2 + (size_t)j * H;
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        for (int i = 0; i < H; i += 32) {
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(x + i)),
                                                        _mm256_loadu_si256((const __m256i*)(wj + i))));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)(x + i + 16)),
                                                        _mm256_loadu_si256((const __m256i*)(wj + i + 16))));
        }
        __m256i t = _mm256_add_epi32(s0, s1);
        __m128i u = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
        u = _mm_hadd_epi32(u, u);
        u = _mm_hadd_epi32(u, u);
        z[j] = _mm_cvtsi128_si32(u);
    }
}
#endif

int evaluate(const Accumulator& acc, int stm, int bucket) {
    bucket = std::max(0, std::min(bucket, OUTPUT_BUCKETS - 1));
    int32_t z[L2];
#ifdef NNUE_X86
    if (g_avx2 && g_simdAllowed) denseIntAVX2(acc, stm, bucket, z);
    else
#endif
        denseIntPlain(acc, stm, bucket, z);

    // float tail (shared by both paths, so their results are identical)
    const Net& n = g_net;
    float h2[L2];
    for (int j = 0; j < L2; ++j) {
        const float v = (float)z[j] * n.s2 + n.l2B[(size_t)bucket * L2 + j];
        h2[j] = std::max(0.f, std::min(1.f, v));
    }
    const float* w3 = n.l3W.data() + (size_t)bucket * L3 * L2;
    float h3[L3];
    for (int k = 0; k < L3; ++k) {
        float s = n.l3B[(size_t)bucket * L3 + k];
        for (int j = 0; j < L2; ++j) s += w3[k * L2 + j] * h2[j];
        h3[k] = std::max(0.f, std::min(1.f, s));
    }
    float o = n.outB[bucket];
    const float* wo = n.outW.data() + (size_t)bucket * L3;
    for (int k = 0; k < L3; ++k) o += wo[k] * h3[k];
    return (int)(o * (float)n.scale);                   // truncates toward zero
}

}  // namespace nnue
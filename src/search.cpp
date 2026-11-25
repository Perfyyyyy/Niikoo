#include "search.hpp"
#include "eval.hpp"
#include "tables.hpp"
#include "tb.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#endif

using namespace chess;
using Clock = std::chrono::steady_clock;

static const int MATE = 100000, MATE_BOUND = MATE - 512, INF = 1000000;
static const int MAX_PLY = 128;
static const int TB_WIN  = 30000;   // tb win, way above any eval but below mate
static const int MAX_THREADS = 256;

// aspiration window
static const int ASP_MIN_DEPTH = 5;
static const int ASP_DELTA     = 25;
static const int ASP_MAX_DELTA = 800;
static const int ASP_FH_REDUCE = 1, ASP_FH_MAX = 3;   // fail high retries search this much shallower (0 = off)

// qsearch pruning
static const int DELTA_MARGIN = 200;
static const int BIG_DELTA    = 900 + DELTA_MARGIN;

// move ordering bands, int16 so stay under 32767
static const int S_TT      = 32000;
static const int S_CAPTURE = 20000;   // + mvv-lva
static const int S_KILLER1 = 15000;
static const int S_KILLER2 = 14000;
static const int S_COUNTER = 13000;   // quiet that refuted opp's last move
static const int S_BAD     = -28000;  // + mvv-lva, SEE < 0 caps go after all quiets
static const int S_BAD_THRESH = -26000;   // below all quiets, above losing caps
static const int QUIET_MAX = 12000, QUIET_MIN = -24000;   // clamp for quiet hist
static const int MAX_HIST  = 8192;    // hist range

// nmp / lmr
static const int NMP_MIN_DEPTH = 3;
static const int LMR_MIN_DEPTH = 3;
static const int LMR_MIN_MOVES = 3;

// pruning margins
static const int RFP_MAX_DEPTH = 6, RFP_MARGIN = 80;                    // rfp
static const int FP_MAX_DEPTH  = 4, FP_BASE = 120, FP_MARGIN = 100;     // fp
static const int LMP_MAX_DEPTH = 4;                                     // lmp
static const int IIR_MIN_DEPTH = 4;                                     // iir
static const int IIR_CUT_MIN_DEPTH = 6;                                 // cut nodes without a tt move lose a 2nd ply from here
static const int IIR_PV_EXACT_SLACK = 3;                                // pv nodes skip iir when tt has an exact entry this close to depth
static const int SE_MIN_DEPTH = 7, SE_MARGIN = 2;                       // singular, margin = SE_MARGIN * depth
static const int RAZOR_MAX_DEPTH = 3, RAZOR_BASE = 200, RAZOR_MARGIN = 150;   // razoring
static const int PROBCUT_MIN_DEPTH = 5, PROBCUT_MARGIN = 160, PROBCUT_REDUCTION = 4;
static const int NMP_BASE = 3, NMP_DIV = 3, NMP_EVAL_DIV = 200, NMP_EVAL_MAX = 3;
// history pruning: skip quiet if hist < -HP_MARGIN * depth
static const int HP_MAX_DEPTH = 3, HP_MARGIN = 2500;      // 5 tables summed so the scale is big
// extra lmr at cut nodes
static const int LMR_CUT_NODE = 1;
// extra lmr tweaks, 0 = off
static const int LMR_EVAL_LOW_MARGIN  = 150;   // non-pv, eval way below alpha -> reduce more
static const int LMR_EVAL_HIGH_MARGIN = 200;   // cut node, eval way above beta -> reduce more
static const int LMR_STRONG_IMPROVE   = 80;    // eval jumped vs 2 plies ago -> reduce less
static const int LMR_TT_LOWER         = 1;     // non-pv, tt says fail high -> reduce more
// root lmr (0 = off) + root hist
static const int ROOT_LMR_MIN_DEPTH = 5, ROOT_LMR_MIN_MOVES = 4;
static const int ROOT_HIST_UPDATE = 1;
// time mgmt: decay + weight of best move changes
static const double TM_CHANGE_DECAY = 0.6, TM_CHANGE_WEIGHT = 0.15;
// double singular ext, max SE_MAX_DOUBLE per line
static const int SE_DOUBLE_MARGIN = 24, SE_DOUBLE_MARGIN_PV = 60, SE_MAX_DOUBLE = 4;   // pv needs a bigger margin
static const int SE_NEG_CUT = 2;   // neg ext at cut nodes (0 = off)
// correction hist: learns how wrong the static eval is
static const int CORR_SIZE = 16384, CORR_GRAIN = 256, CORR_MAX_CP = 64;
// 5 tables: pawn (x4), minor, non-pawn per colour, last move (x2 each)
// sum / CORR_DIV, capped at CORR_MAX_TOTAL
// eval scaled down by halfmove clock
static const int EVAL_HM_BASE = 200;
static const int CORR_DIV = 10, CORR_MAX_TOTAL = 96, CORR_CONT_SIZE = 6 * 64 + 1, CORR_CONT_NONE = 6 * 64;
// pawn structure hist
static const int PAWN_HIST_SIZE = 512;
static_assert(PAWN_HIST_SIZE <= CORR_SIZE, "ph = low bits of the pawn corr index");
// qsearch: quiet checks at first ply only
static const int QCHECK_MARGIN = 100;
// fail high blend toward beta
static const int FH_MIN_DEPTH = 3;
static const int SEE_PRUNE_MAX_DEPTH = 6, SEE_PRUNE_MARGIN = 90;       // skip caps with SEE < -margin * depth
// capture futility
static const int CFP_MAX_DEPTH = 5, CFP_BASE = 150, CFP_MARGIN = 80;
// hist bonus / malus
static const int HIST_B_MUL = 40, HIST_B_BASE = 20, HIST_B_MAX = 1200;
static const int HIST_M_MUL = 50, HIST_M_BASE = 20, HIST_M_MAX = 1400;
// fail low rewards the move that led in
static const int PRIOR_DIV = 2;
// after reduced search beats alpha: go deeper if way better, shallower if barely
static const int DEEPER_BASE = 50, DEEPER_MUL = 2;
// low ply hist, near root ordering
static const int LOWPLY = 5;
// nmp verification at this depth and up
static const int NMP_VERIFY_DEPTH = 12;

static const int TT_NO_EVAL = -32768;     // no eval stored

enum Flag : uint8_t { EXACT, LOWER, UPPER };

// ---- TT ----
// lockless, shared by all threads. slot stores key^data so a torn read just looks like a miss
struct TTEntry {                     // unpacked slot
    uint64_t key     = 0;
    int32_t  score   = 0;
    int16_t  eval    = TT_NO_EVAL;   // raw static eval
    uint16_t move    = 0;            // raw Move bits
    int8_t   depth   = 0;
    uint8_t  ageFlag = 0;            // age << 2 | flag
    bool     pv      = false;        // was/is a pv node
    int flag() const { return ageFlag & 3; }
    int age()  const { return ageFlag >> 2; }
};
struct Slot { uint64_t keyx = 0, data = 0; };
static const int BUCKET_SIZE = 4;
struct alignas(64) Bucket { Slot e[BUCKET_SIZE]; };
static_assert(sizeof(Slot) == 16 && sizeof(Bucket) == 64, "TT layout");

// layout: score 18 | eval 16 | move 16 | depth 7 | pv 1 | age+flag 6
static inline uint64_t packData(int32_t score, int16_t eval, uint16_t move, int8_t depth, bool pv, uint8_t ageFlag) {
    const int32_t sc = std::max(-131071, std::min(score, 131071));
    return (uint64_t)((uint32_t)sc & 0x3FFFFu) | ((uint64_t)(uint16_t)eval << 18) | ((uint64_t)move << 34) |
           ((uint64_t)((uint8_t)depth & 0x7F) << 50) | ((uint64_t)(pv ? 1 : 0) << 57) | ((uint64_t)(ageFlag & 63) << 58);
}
static inline TTEntry loadSlot(const Slot& s) {
    const uint64_t d = __atomic_load_n(&s.data, __ATOMIC_RELAXED);
    const uint64_t k = __atomic_load_n(&s.keyx, __ATOMIC_RELAXED);
    TTEntry t;
    t.key     = k ^ d;
    t.score   = ((int32_t)(d & 0x3FFFFu) ^ 0x20000) - 0x20000;      // sign extend
    t.eval    = (int16_t)(uint16_t)(d >> 18);
    t.move    = (uint16_t)(d >> 34);
    t.depth   = (int8_t)((d >> 50) & 0x7F);
    t.pv      = ((d >> 57) & 1) != 0;
    t.ageFlag = (uint8_t)((d >> 58) & 63);
    return t;
}
static inline void storeSlot(Slot& s, uint64_t key, uint64_t data) {
    __atomic_store_n(&s.data, data, __ATOMIC_RELAXED);
    __atomic_store_n(&s.keyx, key ^ data, __ATOMIC_RELAXED);
}

// ---- search control ----
static std::atomic<bool> stopFlag{false};      // uci stop
static std::atomic<bool> abortFlag{false};     // everyone unwind
static std::atomic<bool> g_pondering{false};   // go ponder active
static std::atomic<int64_t> g_deadlineNs{INT64_MAX};   // hard limit ns, max while pondering
static int64_t   g_hardNs = 0, g_softNs = 0, g_searchStartNs = 0;   // limits of running search
static std::mutex g_timeMx;                    // ponderhit vs search start
static uint64_t  g_totalNodes = 0;
static std::vector<Move> g_lastPv;             // pv of returned move, for ponder
static uint64_t  g_lastPvKey = 0;              // root hash of that pv

static inline int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

void requestStop() { stopFlag = true; }
void resetStop() { stopFlag = false; }
bool stopRequested() { return stopFlag.load(); }

void setPondering(bool on) { g_pondering = on; }
bool isPondering() { return g_pondering.load(); }
// opp played the expected move: hard limit starts now, ponder time counts toward soft
// if soft is already used up, only a short grace period
void ponderHit() {
    std::lock_guard<std::mutex> lk(g_timeMx);
    if (!g_pondering.load()) return;
    const int64_t n = nowNs();
    int64_t limit = g_hardNs;
    if (n - g_searchStartNs >= g_softNs)
        limit = std::min(limit, std::max<int64_t>(g_softNs / 4, 20000000LL));
    g_deadlineNs = n + limit;
    g_pondering = false;
}

// ---- the table ----
// allocator for the TT: 2MB aligned + madvise so linux can back it with huge pages (fewer tlb misses)
template <class T> struct HugeAlloc {
    using value_type = T;
#ifdef __linux__
    static constexpr size_t ALIGN = 2u << 20;
#else
    static constexpr size_t ALIGN = 64;
#endif
    HugeAlloc() = default;
    template <class U> HugeAlloc(const HugeAlloc<U>&) {}
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
template <class T, class U> bool operator==(const HugeAlloc<T>&, const HugeAlloc<U>&) { return true; }
template <class T, class U> bool operator!=(const HugeAlloc<T>&, const HugeAlloc<U>&) { return false; }
using TTVec = std::vector<Bucket, HugeAlloc<Bucket>>;

static const size_t DEFAULT_HASH_MB = 128, MAX_HASH_MB = 65536;
static TTVec TT(DEFAULT_HASH_MB * 1024 * 1024 / sizeof(Bucket));
static uint64_t ttBuckets = TT.size();
static size_t   ttSizeMB  = DEFAULT_HASH_MB;
static int      ttAge     = 0;       // bumped each search, wraps at 16

static inline Bucket& bucketFor(uint64_t key) {
    // multiply-shift, size doesn't have to be pow2
    return TT[(uint64_t)(((unsigned __int128)key * ttBuckets) >> 64)];
}

// prefetch the bucket, TT misses hurt
static inline void ttPrefetch(uint64_t key) { __builtin_prefetch(&bucketFor(key)); }

// true on hit, refreshes age too
static inline bool ttProbe(uint64_t key, TTEntry& out) {
    Bucket& bk = bucketFor(key);
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        TTEntry c = loadSlot(bk.e[i]);
        if (c.key == key) {
            if (c.age() != ttAge) {
                c.ageFlag = (uint8_t)((ttAge << 2) | c.flag());
                storeSlot(bk.e[i], key, packData(c.score, c.eval, c.move, c.depth, c.pv, c.ageFlag));
            }
            out = c;
            return true;
        }
    }
    return false;
}

// same key: update in place unless old one is better. else evict lowest depth - 4*age diff
static void ttStore(uint64_t key, uint16_t mv, int score, int depth, int flag, int eval = TT_NO_EVAL, int pv = -1) {
    Bucket& bk = bucketFor(key);
    Slot* victim = nullptr;
    TTEntry vc;
    int victimValue = 1 << 30;
    for (int i = 0; i < BUCKET_SIZE; ++i) {
        Slot& s = bk.e[i];
        const TTEntry c = loadSlot(s);
        if (c.key == key) {
            const bool sameAge = c.age() == ttAge;
            if (sameAge && ((depth <= 0 && c.depth > 0) || (flag != EXACT && depth + 3 <= c.depth)))
                return;                                   // keep the deeper one
            if (mv == 0) mv = c.move;                     // keep old move
            if (eval == TT_NO_EVAL) eval = c.eval;        // keep old eval
            if (pv < 0) pv = c.pv;                        // keep pv flag
            victim = &s; vc = c;
            break;
        }
        const int value = c.key == 0 ? -100000 : c.depth - 4 * ((ttAge - c.age() + 16) & 15);
        if (value < victimValue) { victimValue = value; victim = &s; vc = c; }
    }
    // qsearch entries shouldn't evict deeper ones
    if (depth <= 0 && vc.key != key && victimValue > 0) return;

    const int8_t d8 = (int8_t)std::max(0, std::min(depth, 127));
    if (eval != TT_NO_EVAL) eval = std::max(-20000, std::min(eval, 20000));
    storeSlot(*victim, key, packData((int32_t)score, (int16_t)eval, mv, d8, pv > 0, (uint8_t)((ttAge << 2) | flag)));
}

void setHashSize(size_t mb) {
    mb = std::max<size_t>(1, std::min(mb, MAX_HASH_MB));
    TTVec().swap(TT);                       // free old one first
    try {
        TT.resize(mb * 1024 * 1024 / sizeof(Bucket));
        ttSizeMB = mb;
    } catch (const std::bad_alloc&) {
        std::cerr << "Warning: could not allocate " << mb << " MB hash, using 1 MB" << std::endl;
        TT.resize(1024 * 1024 / sizeof(Bucket));
        ttSizeMB = 1;
    }
    ttBuckets = TT.size();
    ttAge = 0;
}
size_t hashSizeMB() { return ttSizeMB; }

// permille, for info hashfull
static int hashfull() {
    const uint64_t n = std::min<uint64_t>(ttBuckets, 250);
    int used = 0;
    for (uint64_t i = 0; i < n; ++i)
        for (int j = 0; j < BUCKET_SIZE; ++j) {
            const TTEntry c = loadSlot(TT[i].e[j]);
            if (c.key != 0 && c.age() == ttAge) ++used;
        }
    return (int)(used * 1000 / (n * BUCKET_SIZE));
}

// ---- per thread state ----
// nothing in here is shared between threads
// finny table: per perspective + king key, cached acc and the pieces it was built from
// refresh = apply the diff instead of re-adding everything
struct FinnyEntry {
    alignas(64) int16_t v[nnue::MAX_HIDDEN];
    uint64_t pcs[2][6];                           // pieces v was built from
    bool     valid = false;
};

struct ThreadData {
    int      id = 0;                              // 0 = main
    uint64_t nodes = 0;
    uint64_t rootBestNodes = 0;                   // nodes on best root move, for time mgmt
    Move     killers[MAX_PLY + 1][2];
    int      history[2][64][64];                  // [side][from][to]
    // cont hist: quiet score given the move 1, 2, 4 plies back
    // [which][prev pt*64 + prev to][(side*6 + pt)*64 + to]
    int      contHist[3][6 * 64][2 * 6 * 64];
    // pawn hist [pawn hash][(side*6 + pt)*64 + to]
    int      pawnHist[PAWN_HIST_SIZE][2 * 6 * 64];
    // capture hist [side][pt][to][victim]
    int      capHist[2][6][64][6];
    Move     counterMove[6][64];                  // [prev pt][prev to] -> refuting quiet
    bool     stkQuiet[MAX_PLY + 2];               // move into this ply was quiet
    int      cutoffCnt[MAX_PLY + 4];              // beta cutoffs per ply, reset 2 plies up
    int      lowPlyHist[LOWPLY][64 * 64];         // [ply][from*64 + to], low plies only
    uint64_t rootKey = 0;                         // root the order below belongs to
    uint64_t rootNodes[64 * 64];                  // nodes per root move, from*64 + to
    int      nmpMinPly = 0;                       // no nmp below this ply (verification)
    uint64_t posKey[MAX_PLY + 2];                 // hash per ply, for repetitions
    uint8_t  extCount[MAX_PLY + 2];               // double exts on the line
    // move into each ply (stkTo < 0 = none)
    int      stkPt[MAX_PLY + 2], stkTo[MAX_PLY + 2];
    int      evalStack[MAX_PLY + 2];              // static eval per ply
    // nnue: acc per ply, lazy. delta[p] = move into ply p, accValid[p] = acc[p] has it applied
    struct FeatureRef { int8_t colour, pt, sq; };
    struct AccDelta { int nsub = 0, nadd = 0; FeatureRef sub[2], add[2]; };
    nnue::Accumulator acc[MAX_PLY + 2];
    AccDelta delta[MAX_PLY + 2];
    bool     accValid[MAX_PLY + 2];
    uint8_t  accKey[MAX_PLY + 2][2];              // king key each perspective was built with
    FinnyEntry finny[2][32];                      // [persp][king key]
    uint32_t finnyNetId = 0;                      // which net the cache is for
    // pv, triangular table
    Move     pv[MAX_PLY + 2][MAX_PLY + 2];
    int      pvLen[MAX_PLY + 2];
    // last finished iteration, main reads it after stop
    int      doneDepth = 0, doneScore = 0;
    Move     doneBest;
    Move     donePv[MAX_PLY + 2];
    int      donePvLen = 0;

    void clear() {
        std::memset(pvLen, 0, sizeof(pvLen));
        doneDepth = doneScore = donePvLen = 0;
        doneBest = Move();
        std::memset(history, 0, sizeof(history));
        std::memset(contHist, 0, sizeof(contHist));
        std::memset(capHist, 0, sizeof(capHist));
        for (auto& row : counterMove) for (Move& m : row) m = Move();
        for (auto& k : killers) k[0] = k[1] = Move();
        std::memset(stkPt, 0, sizeof(stkPt));
        for (int& t : stkTo) t = -1;
        std::memset(evalStack, 0, sizeof(evalStack));
        std::memset(pawnHist, 0, sizeof(pawnHist));
        std::memset(posKey, 0, sizeof(posKey));
        std::memset(stkQuiet, 0, sizeof(stkQuiet));
        nmpMinPly = 0;
        std::memset(extCount, 0, sizeof(extCount));
        nodes = 0;
        rootBestNodes = 0;
        std::memset(cutoffCnt, 0, sizeof(cutoffCnt));
        std::memset(lowPlyHist, 0, sizeof(lowPlyHist));
        std::memset(rootNodes, 0, sizeof(rootNodes));
        rootKey = 0;
    }
};

static std::vector<std::unique_ptr<ThreadData>> g_threads;

// correction tables, shared by all threads so helpers don't each relearn the same eval errors.
// racy on purpose (relaxed loads/stores, a lost update just costs a little learning). per thread history stays private
struct CorrTables {
    int16_t pawn[2][CORR_SIZE];                   // [stm][pawn hash]
    int16_t minor[2][CORR_SIZE];                  // by minors
    int16_t nonPawn[2][2][CORR_SIZE];             // [colour][stm], that colour's N/B/R/Q
    int16_t cont[2][CORR_CONT_SIZE];              // [stm][last move]
    void clear() { std::memset(this, 0, sizeof(*this)); }
};
static CorrTables g_corr;
static inline int corrLoad(const int16_t& e) { return __atomic_load_n(&e, __ATOMIC_RELAXED); }
static inline void corrStore(int16_t& e, int v) { __atomic_store_n(&e, (int16_t)v, __ATOMIC_RELAXED); }

// ---- nnue acc ----
// call BEFORE makeMove. ply = ply after the move. nothing computed until evalAt
static inline void accSetMove(ThreadData& td, const Board& b, const Move m, int ply) {
    ThreadData::AccDelta& d = td.delta[ply];
    td.accValid[ply] = false;
    const int us = b.sideToMove() == Color::WHITE ? 0 : 1;
    const int from = m.from().index(), to = m.to().index();
    const int pt = (int)b.at(m.from()).type();
    const int type = m.typeOf();
    d.nsub = d.nadd = 0;
    auto sub = [&](int c, int p, int sq) { d.sub[d.nsub++] = {(int8_t)c, (int8_t)p, (int8_t)sq}; };
    auto add = [&](int c, int p, int sq) { d.add[d.nadd++] = {(int8_t)c, (int8_t)p, (int8_t)sq}; };
    if (type == Move::CASTLING) {                      // king takes own rook
        const bool kingSide = to > from;
        const int rank = from & ~7;
        sub(us, 5, from); sub(us, 3, to);
        add(us, 5, rank + (kingSide ? 6 : 2)); add(us, 3, rank + (kingSide ? 5 : 3));
    } else if (type == Move::ENPASSANT) {
        sub(us, 0, from); sub(us ^ 1, 0, us == 0 ? to - 8 : to + 8); add(us, 0, to);
    } else {
        const int victim = (int)b.at(m.to()).type();   // 6 = empty
        sub(us, pt, from);
        if (victim <= 5) sub(us ^ 1, victim, to);
        add(us, type == Move::PROMOTION ? (int)m.promotionType() : pt, to);
    }
}

static inline void accSetNull(ThreadData& td, int ply) {   // null move: nothing changes
    td.accValid[ply] = false;
    td.delta[ply].nsub = td.delta[ply].nadd = 0;
}

// v2 net: features depend on own king bucket + mirror, so a king move to a new key = full rebuild for that side (rare)
// placement at ply j = board at ply minus the deltas in between, no board copies
#ifdef NNUE_VERIFY
static std::atomic<uint64_t> g_dbgRefreshNow{0}, g_dbgRefreshBack{0}, g_dbgEvals{0};
uint64_t nnueVerifyStat(int which) { return which == 0 ? g_dbgRefreshNow.load() : which == 1 ? g_dbgRefreshBack.load() : g_dbgEvals.load(); }
#endif

static void accRefreshPerspective(ThreadData& td, const Board& b, int ply, int j, int p) {
    static const PieceType ACC_PT[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                                        PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
    uint64_t pcs[2][6];
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < 6; ++pt)
            pcs[c][pt] = b.pieces(ACC_PT[pt], c == 0 ? Color::WHITE : Color::BLACK).getBits();
    for (int i = ply; i > j; --i) {                    // undo deltas
        const ThreadData::AccDelta& d = td.delta[i];
        for (int a = 0; a < d.nadd; ++a) pcs[d.add[a].colour][d.add[a].pt] &= ~(1ULL << d.add[a].sq);
        for (int s = 0; s < d.nsub; ++s) pcs[d.sub[s].colour][d.sub[s].pt] |= 1ULL << d.sub[s].sq;
    }
#ifdef NNUE_VERIFY
    (j == ply ? g_dbgRefreshNow : g_dbgRefreshBack)++;
#endif
    const int key = pcs[p][5] ? nnue::kingKey(p, __builtin_ctzll(pcs[p][5])) : 0;
    td.accKey[j][p] = (uint8_t)key;
    // update the cached acc to the placement, then copy it out
    FinnyEntry& fe = td.finny[p][key];
    if (!fe.valid) {                                   // empty = just bias
        nnue::loadBias(fe.v);
        std::memset(fe.pcs, 0, sizeof(fe.pcs));
        fe.valid = true;
    }
    for (int c = 0; c < 2; ++c)
        for (int pt = 0; pt < 6; ++pt) {
            const uint64_t cur = pcs[c][pt], old = fe.pcs[c][pt];
            uint64_t rem = old & ~cur, add = cur & ~old;
            while (rem) {
                const int sq = __builtin_ctzll(rem);
                rem &= rem - 1;
                nnue::subFeatureRaw(fe.v, nnue::featureIndex(p, key, c, pt, sq));
            }
            while (add) {
                const int sq = __builtin_ctzll(add);
                add &= add - 1;
                nnue::addFeatureRaw(fe.v, nnue::featureIndex(p, key, c, pt, sq));
            }
            fe.pcs[c][pt] = cur;
        }
    std::memcpy(td.acc[j].v[p], fe.v, (size_t)nnue::hiddenSize() * sizeof(int16_t));
}

// bring acc[ply] up to date from nearest valid ancestor
static void accUpdateTo(ThreadData& td, const Board& b, int ply) {
    int k = ply;
    while (k > 0 && !td.accValid[k]) --k;
    for (int j = k + 1; j <= ply; ++j) {
        const ThreadData::AccDelta& d = td.delta[j];
        for (int p = 0; p < 2; ++p) {
            int key = td.accKey[j - 1][p];
            for (int i = 0; i < d.nadd; ++i)           // own king moved to a new key?
                if (d.add[i].pt == 5 && d.add[i].colour == p) key = nnue::kingKey(p, d.add[i].sq);
            if (key != td.accKey[j - 1][p]) {
                accRefreshPerspective(td, b, ply, j, p);
                continue;
            }
            td.accKey[j][p] = (uint8_t)key;
            int subs[2], adds[2];
            for (int i = 0; i < d.nsub; ++i) subs[i] = nnue::featureIndex(p, key, d.sub[i].colour, d.sub[i].pt, d.sub[i].sq);
            for (int i = 0; i < d.nadd; ++i) adds[i] = nnue::featureIndex(p, key, d.add[i].colour, d.add[i].pt, d.add[i].sq);
            // acc[j] = acc[j-1] +/- features, one pass
            nnue::updatePerspective(td.acc[j].v[p], td.acc[j - 1].v[p], subs, d.nsub, adds, d.nadd);
        }
        td.accValid[j] = true;
    }
}

// static eval for stm (net only, search() refuses to run without one)
static inline int evalAt(ThreadData& td, const Board& b, int ply) {
    accUpdateTo(td, b, ply);
#ifdef NNUE_VERIFY
    {
        ++g_dbgEvals;
        nnue::Accumulator ref;
        nnueRefresh(b, ref);
        for (int p = 0; p < 2; ++p)
            if (std::memcmp(ref.v[p], td.acc[ply].v[p], (size_t)nnue::hiddenSize() * sizeof(int16_t)) != 0) {
                std::cerr << "NNUE accumulator mismatch at ply " << ply << " perspective " << p << " fen " << b.getFen() << std::endl;
                std::abort();
            }
    }
#endif
    const int bucket = nnue::outputBucket(__builtin_popcountll(b.occ().getBits()));
    int s = nnue::evaluate(td.acc[ply], b.sideToMove() == Color::WHITE ? 0 : 1, bucket);
    s = nnueAdjust(b, s);                              // mop-up / drawish
    return std::max(-20000, std::min(20000, s));
}

// root: acc[0] from scratch
static int rootEval(ThreadData& td, const Board& b) {
    if (td.finnyNetId != nnue::netId()) {              // new net, cache is junk
        for (auto& perPersp : td.finny) for (FinnyEntry& fe : perPersp) fe.valid = false;
        td.finnyNetId = nnue::netId();
    }
    nnueRefresh(b, td.acc[0]);
    for (int p = 0; p < 2; ++p) {
        const uint64_t k = b.pieces(PieceType::KING, p == 0 ? Color::WHITE : Color::BLACK).getBits();
        td.accKey[0][p] = (uint8_t)(k ? nnue::kingKey(p, __builtin_ctzll(k)) : 0);
    }
    td.accValid[0] = true;
    return b.inCheck() ? -INF : evalAt(td, b, 0);
}

void setThreads(int n) {                              // not while searching
    n = std::max(1, std::min(n, MAX_THREADS));
    while ((int)g_threads.size() > n) g_threads.pop_back();
    while ((int)g_threads.size() < n) {
        auto t = std::make_unique<ThreadData>();
        t->id = (int)g_threads.size();
        t->clear();
        g_threads.push_back(std::move(t));
    }
}
int threadCount() { return std::max<int>(1, (int)g_threads.size()); }
static void ensureThreads() { if (g_threads.empty()) setThreads(1); }

static uint64_t totalNodes() {
    uint64_t n = 0;
    for (auto& t : g_threads) n += __atomic_load_n(&t->nodes, __ATOMIC_RELAXED);
    return n;
}

// ---- upcoming repetition (cuckoo) ----
// spots a repeat one reversible move away without searching. needs zobrist keys, -DNO_CUCKOO to disable
// hashes of the game positions before the root, oldest first (last = 1 ply before root). set by the uci side
// between searches, empty = only the current search line is checked
static std::vector<uint64_t> g_gameKeys;
void setGameHistory(const std::vector<uint64_t>& keys) { g_gameKeys = keys; }

#ifndef NO_CUCKOO
static uint64_t cuckooKey[8192];
static uint16_t cuckooMove[8192];                // from | to << 6
static inline int cuckH1(uint64_t h) { return (int)(h & 0x1fff); }
static inline int cuckH2(uint64_t h) { return (int)((h >> 16) & 0x1fff); }

// zobrist tables are private so measure the keys: hash is xor of piece keys, so hash(s1) ^ hash(s2) = key diff of a move s1->s2
// boards come from FENs, kings parked out of the way
static uint64_t fenHash(const char* sq64, bool whiteToMove) {
    std::string fen;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            const char c = sq64[r * 8 + f];
            if (c == '.') { ++empty; continue; }
            if (empty) { fen += (char)('0' + empty); empty = 0; }
            fen += c;
        }
        if (empty) fen += (char)('0' + empty);
        if (r) fen += '/';
    }
    fen += whiteToMove ? " w - - 0 1" : " b - - 0 1";
    Board bd(fen);
    return bd.hash();
}

// key for a reversible move s1<->s2 (incl stm flip)
static uint64_t cuckooPairKey(char pc, int s1, int s2, uint64_t sideKey) {
    static const int WK[3] = {4, 0, 7}, BK[3] = {60, 56, 63};
    auto pick = [&](const int* cand) { for (int i = 0; i < 3; ++i) if (cand[i] != s1 && cand[i] != s2) return cand[i]; return cand[2]; };
    const int wk = pick(WK), bk = pick(BK);
    char a[64], b[64];
    for (int i = 0; i < 64; ++i) a[i] = b[i] = '.';
    if (pc != 'K') a[wk] = b[wk] = 'K';                 // piece might be a king
    if (pc != 'k') a[bk] = b[bk] = 'k';
    a[s1] = pc;
    b[s2] = pc;
    return fenHash(a, true) ^ fenHash(b, true) ^ sideKey;
}

static void initCuckoo() {
    static bool done = false;
    if (done) return;
    done = true;
    char base[64];
    for (int i = 0; i < 64; ++i) base[i] = '.';
    base[4] = 'K'; base[60] = 'k';
    const uint64_t sideKey = fenHash(base, true) ^ fenHash(base, false);
    static const PieceType types[5] = {PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
    static const char WCH[5] = {'N', 'B', 'R', 'Q', 'K'}, BCH[5] = {'n', 'b', 'r', 'q', 'k'};
    for (int c = 0; c < 2; ++c)
        for (int t = 0; t < 5; ++t) {
            const PieceType pt = types[t];
            const char pc = c == 0 ? WCH[t] : BCH[t];
            for (int s1 = 0; s1 < 64; ++s1) {
                const Square sq1(s1);
                uint64_t att = 0;
                if (pt == PieceType::KNIGHT)      att = attacks::knight(sq1).getBits();
                else if (pt == PieceType::BISHOP) att = attacks::bishop(sq1, Bitboard(0)).getBits();
                else if (pt == PieceType::ROOK)   att = attacks::rook(sq1, Bitboard(0)).getBits();
                else if (pt == PieceType::QUEEN)  att = attacks::bishop(sq1, Bitboard(0)).getBits() | attacks::rook(sq1, Bitboard(0)).getBits();
                else                              att = attacks::king(sq1).getBits();
                for (int s2 = s1 + 1; s2 < 64; ++s2) {
                    if (!((att >> s2) & 1)) continue;
                    uint64_t key = cuckooPairKey(pc, s1, s2, sideKey);
                    uint16_t move = (uint16_t)(s1 | (s2 << 6));
                    int i = cuckH1(key);
                    for (;;) {
                        std::swap(cuckooKey[i], key);
                        std::swap(cuckooMove[i], move);
                        if (!move) break;
                        i = (i == cuckH1(key)) ? cuckH2(key) : cuckH1(key);
                    }
                }
            }
        }
}

// squares strictly between
static inline uint64_t betweenSq(int s1, int s2) {
    const uint64_t b1 = 1ULL << s1, b2 = 1ULL << s2;
    if (attacks::rook(Square(s1), Bitboard(0)).getBits() & b2)
        return attacks::rook(Square(s1), Bitboard(b2)).getBits() & attacks::rook(Square(s2), Bitboard(b1)).getBits();
    if (attacks::bishop(Square(s1), Bitboard(0)).getBits() & b2)
        return attacks::bishop(Square(s1), Bitboard(b2)).getBits() & attacks::bishop(Square(s2), Bitboard(b1)).getBits();
    return 0;
}

static bool hasUpcomingRepetition(const ThreadData& td, const Board& b, int ply) {
    const int n = (int)g_gameKeys.size();
    const int lim = std::min((int)b.halfMoveClock(), ply + n);
    if (lim < 3) return false;
    const uint64_t key = b.hash(), occ = b.occ().getBits();
    for (int i = 3; i <= lim; i += 2) {                 // same side to move, 3, 5, ... back
        const int idx = ply - i;
        const uint64_t prev = idx >= 0 ? td.posKey[idx] : g_gameKeys[n + idx];
        const uint64_t diff = key ^ prev;
        int j = cuckH1(diff);
        if (cuckooKey[j] != diff) {
            j = cuckH2(diff);
            if (cuckooKey[j] != diff) continue;
        }
        const int s1 = cuckooMove[j] & 63, s2 = cuckooMove[j] >> 6;
        if (betweenSq(s1, s2) & occ) continue;           // blocked
        const int from = ((occ >> s1) & 1) ? s1 : s2;    // where the piece is now
        if (!((occ >> from) & 1)) continue;
        if (b.at(Square(from)).color() != b.sideToMove()) continue;
        if (idx >= 0) return true;
        // pre-root position: only counts if it already occurred once more (the repeat would be the 3rd)
        for (int k = i + 2; k <= lim; k += 2)
            if (g_gameKeys[n + ply - k] == prev) return true;
    }
    return false;
}
#endif

static int  LMR[64][64];                 // [depth][moves], read-only after init
static const int NO_EVAL = -INF;         // in check, no eval
static bool lmrReady = false;
static bool printInfo = true;

static void initLMR() {
    if (lmrReady) return;
#ifndef NO_CUCKOO
    initCuckoo();
#endif
    for (int d = 1; d < 64; ++d)
        for (int m = 1; m < 64; ++m)
            LMR[d][m] = (int)(0.75 + std::log((double)d) * std::log((double)m) / 2.25);
    lmrReady = true;
}

// zeroing a huge TT on one thread is slow, split it
static void clearTable() {
    const size_t n = TT.size();
    const int nt = (n * sizeof(Bucket) >= (64u << 20)) ? std::max(1, (int)g_threads.size()) : 1;
    if (nt <= 1) {
        std::fill(TT.begin(), TT.end(), Bucket());
        return;
    }
    std::vector<std::thread> workers;
    const size_t chunk = (n + nt - 1) / nt;
    for (int i = 0; i < nt; ++i) {
        const size_t lo = (size_t)i * chunk, hi = std::min(n, lo + chunk);
        if (lo >= hi) break;
        workers.emplace_back([lo, hi] { std::fill(TT.begin() + lo, TT.begin() + hi, Bucket()); });
    }
    for (auto& w : workers) w.join();
}

void clearTT() {
    ensureThreads();
    clearTable();
    ttAge = 0;
    g_corr.clear();
    for (auto& t : g_threads) t->clear();
}

void setSearchInfo(bool on) { printInfo = on; }

// mate scores stored relative to node
static int toTT(int s, int ply)   { return s >= MATE_BOUND ? s + ply : (s <= -MATE_BOUND ? s - ply : s); }
static int fromTT(int s, int ply) { return s >= MATE_BOUND ? s - ply : (s <= -MATE_BOUND ? s + ply : s); }

// ---- time ----
struct Timeout {};

// each thread counts its own nodes, only main reads the clock and aborts everyone
static inline void checkTime(ThreadData& td) {
    const uint64_t n = td.nodes + 1;
    __atomic_store_n(&td.nodes, n, __ATOMIC_RELAXED);
    if ((n & 2047) == 0) {
        if (stopFlag.load(std::memory_order_relaxed) || abortFlag.load(std::memory_order_relaxed)) throw Timeout();
        if (td.id == 0 && nowNs() > g_deadlineNs.load(std::memory_order_relaxed)) {
            abortFlag = true;
            throw Timeout();
        }
    }
}

uint64_t searchNodes() { return g_totalNodes; }

static inline int sideIdx(const Board& b) { return b.sideToMove() == Color::WHITE ? 0 : 1; }

static inline void gravity(int& h, int bonus) {
    h += bonus - h * std::abs(bonus) / MAX_HIST;      // gravity
}

static inline int curIdx(int side, int pt, int to) { return (side * 6 + pt) * 64 + to; }

// pawn structure hash for pawn hist + corr
static inline uint64_t pawnKey(const Board& b) {
    const uint64_t w = b.pieces(PieceType::PAWN, Color::WHITE).getBits();
    const uint64_t k = b.pieces(PieceType::PAWN, Color::BLACK).getBits();
    uint64_t h = w * 0x9E3779B97F4A7C15ULL ^ k * 0xC2B2AE3D27D4EB4FULL;
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ULL;                       // 2nd round so high squares reach low bits
    h ^= h >> 32;
    return h;
}
static inline int pawnHistIdx(const Board& b) { return (int)(pawnKey(b) & (PAWN_HIST_SIZE - 1)); }

// cont hist looks 1, 2, 4 plies back
static const int CH_BACK[3] = {0, 1, 3};      // offset into stkPt/stkTo

// quiet score: main + pawn + cont hist
// ph = pawnHistIdx of this node
static inline int histScore(ThreadData& td, int side, int ply, int pt, const Move& m, int ph) {
    const int to = m.to().index(), cur = curIdx(side, pt, to);
    int s = td.history[side][m.from().index()][to] + td.pawnHist[ph][cur];
    if (ply < LOWPLY) s += td.lowPlyHist[ply][m.from().index() * 64 + to] / (1 + ply);
    for (int k = 0; k < 3; ++k) {
        const int p = ply - CH_BACK[k];
        if (p >= 0 && td.stkTo[p] >= 0) s += td.contHist[k][td.stkPt[p] * 64 + td.stkTo[p]][cur];
    }
    return s;
}

// board = position before the move
static void updateQuiet(ThreadData& td, const Board& b, int side, int ply, const Move& m, int bonus, int ph) {
    const int pt = (int)b.at(m.from()).type(), to = m.to().index(), cur = curIdx(side, pt, to);
    gravity(td.history[side][m.from().index()][to], bonus);
    gravity(td.pawnHist[ph][cur], bonus);
    if (ply < LOWPLY) gravity(td.lowPlyHist[ply][m.from().index() * 64 + to], bonus);
    for (int k = 0; k < 3; ++k) {
        const int p = ply - CH_BACK[k];
        if (p >= 0 && td.stkTo[p] >= 0) gravity(td.contHist[k][td.stkPt[p] * 64 + td.stkTo[p]][cur], bonus);
    }
}

// victim type, ep counts as pawn
static inline int capVictim(const Board& b, const Move& m) {
    int v = (int)b.at(m.to()).type();
    return v > 5 ? 0 : v;
}
static inline int capHistScore(const ThreadData& td, const Board& b, int side, const Move& m) {
    return td.capHist[side][(int)b.at(m.from()).type()][m.to().index()][capVictim(b, m)];
}
static inline void updateCapture(ThreadData& td, const Board& b, int side, const Move& m, int bonus) {
    gravity(td.capHist[side][(int)b.at(m.from()).type()][m.to().index()][capVictim(b, m)], bonus);
}

static int see(const Board& b, const Move& m);       // below

// ---- move ordering ----
// tt > good caps > killers > counter > quiets by hist > bad caps
// only scores, pickMove does the selecting
// useCont off in qsearch
static void orderMoves(ThreadData& td, const Board& b, Movelist& moves, Move ttMove, int ply, bool useCont = true, int phIn = -1) {
    const int side = sideIdx(b);
    const Move counter = (useCont && td.stkTo[ply] >= 0) ? td.counterMove[td.stkPt[ply]][td.stkTo[ply]] : Move();
    const int ph = useCont ? (phIn >= 0 ? phIn : pawnHistIdx(b)) : 0;
    for (Move& m : moves) {
        int s;
        if (m == ttMove) {
            s = S_TT;
        } else if (b.isCapture(m)) {
            int v = (int)b.at(m.to()).type();
            if (v > 5) v = 0;                           // ep
            int a = (int)b.at(m.from()).type();
            const int mvvLva = 100 + 10 * ORDER_VALUE[v] - ORDER_VALUE[a]
                             + td.capHist[side][a][m.to().index()][v] / 16;   // small, only reorders inside band
            const bool promo = m.typeOf() == Move::PROMOTION;
            // SEE only if attacker > victim
            if (!promo && ORDER_VALUE[v] < ORDER_VALUE[a] && see(b, m) < 0) {
                s = S_BAD + mvvLva;
            } else {
                s = S_CAPTURE + mvvLva;
                if (promo) s += 80;
            }
        } else if (m.typeOf() == Move::PROMOTION) {
            s = S_CAPTURE + 80;
        } else if (m == td.killers[ply][0]) {
            s = S_KILLER1;
        } else if (m == td.killers[ply][1]) {
            s = S_KILLER2;
        } else if (counter != Move() && m == counter) {
            s = S_COUNTER;
        } else {
            if (useCont) s = histScore(td, side, ply, (int)b.at(m.from()).type(), m, ph);
            else         s = td.history[side][m.from().index()][m.to().index()];
            s = std::max(QUIET_MIN, std::min(s, QUIET_MAX));
        }
        m.setScore((int16_t)s);
    }
}

// selection instead of sort, most nodes cut in 1-3 moves
static inline void pickMove(Movelist& moves, int i) {
    int bi = i;
    for (int j = i + 1; j < moves.size(); ++j)
        if (moves[j].score() > moves[bi].score()) bi = j;
    if (bi != i) std::swap(moves[i], moves[bi]);
}

// ---- SEE ----
static const int SEE_VAL[6] = {100, 320, 330, 500, 900, 20000};
static const PieceType PT_LIST[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                                     PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

static int see(const Board& b, const Move& m) {
    const Square toSq = m.to();
    const int to = toSq.index();
    const bool ep = m.typeOf() == Move::ENPASSANT;
    const bool whiteMoves = b.sideToMove() == Color::WHITE;

    uint64_t pc[2][6];
    for (int pt = 0; pt < 6; ++pt) {
        pc[0][pt] = b.pieces(PT_LIST[pt], Color::WHITE).getBits();
        pc[1][pt] = b.pieces(PT_LIST[pt], Color::BLACK).getBits();
    }

    auto attackersTo = [&](uint64_t occ) -> uint64_t {
        Bitboard o(occ);
        uint64_t r = 0;
        r |= attacks::pawn(Color::BLACK, toSq).getBits() & pc[0][0];
        r |= attacks::pawn(Color::WHITE, toSq).getBits() & pc[1][0];
        r |= attacks::knight(toSq).getBits() & (pc[0][1] | pc[1][1]);
        r |= attacks::bishop(toSq, o).getBits() & (pc[0][2] | pc[1][2] | pc[0][4] | pc[1][4]);
        r |= attacks::rook(toSq, o).getBits()   & (pc[0][3] | pc[1][3] | pc[0][4] | pc[1][4]);
        r |= attacks::king(toSq).getBits() & (pc[0][5] | pc[1][5]);
        return r & occ;
    };

    int victim = ep ? 0 : (int)b.at(toSq).type();
    if (victim > 5) victim = 0;
    int aType = (int)b.at(m.from()).type();

    uint64_t occ = b.occ().getBits();
    if (ep) occ &= ~(1ULL << (whiteMoves ? to - 8 : to + 8));

    int gain[40];
    int d = 0;
    gain[0] = SEE_VAL[victim];
    uint64_t fromSet = 1ULL << m.from().index();
    int side = whiteMoves ? 0 : 1;

    do {
        ++d;
        gain[d] = SEE_VAL[aType] - gain[d - 1];
        if (std::max(-gain[d - 1], gain[d]) < 0) break;
        occ &= ~fromSet;
        uint64_t att = attackersTo(occ);
        side ^= 1;
        fromSet = 0;
        for (int pt = 0; pt < 6; ++pt) {
            uint64_t x = att & pc[side][pt];
            if (x) { fromSet = x & (~x + 1); aType = pt; break; }
        }
    } while (fromSet);

    while (--d) gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
    return gain[0];
}

static inline int correctEval(const ThreadData& td, const Board& b, int side, int ply, int eval);   // defined later

// repeat on current line = draw after ONE repeat, pre-root positions need threefold
static inline bool isRepeated(const ThreadData& td, const Board& b, int ply) {
    const uint64_t key = b.hash();
    const int lim = std::min((int)b.halfMoveClock(), ply);
    for (int back = 4; back <= lim; back += 2)          // earliest recurrence is 4 plies
        if (td.posKey[ply - back] == key) return true;
    return b.isRepetition(2);
}
// +-1 instead of 0 so draws aren't all identical, helps dodge repetition cycles
static inline int drawScore(const ThreadData& td) { return 1 - (int)(td.nodes & 2); }

// ---- qsearch ----
// quiet direct check (no discovered)
static inline bool quietGivesDirectCheck(const Board& b, const Move& m) {
    const Color us = b.sideToMove();
    const uint64_t kbit = b.pieces(PieceType::KING, us == Color::WHITE ? Color::BLACK : Color::WHITE).getBits();
    if (!kbit) return false;
    const int from = m.from().index();
    const Square toSq = m.to();
    const uint64_t occ = (b.occ().getBits() & ~(1ULL << from)) | (1ULL << toSq.index());
    switch ((int)b.at(m.from()).type()) {                // 0 p, 1 n, 2 b, 3 r, 4 q, 5 k
        case 0:  return (attacks::pawn(us, toSq).getBits() & kbit) != 0;
        case 1:  return (attacks::knight(toSq).getBits() & kbit) != 0;
        case 2:  return (attacks::bishop(toSq, Bitboard(occ)).getBits() & kbit) != 0;
        case 3:  return (attacks::rook(toSq, Bitboard(occ)).getBits() & kbit) != 0;
        case 4:  return ((attacks::bishop(toSq, Bitboard(occ)).getBits() |
                          attacks::rook(toSq, Bitboard(occ)).getBits()) & kbit) != 0;
        default: return false;
    }
}

// gives check w/o making the move (incl discovered)
static inline bool givesCheck(const Board& b, const Move m) {
    const Color us = b.sideToMove();
    const Color them = us == Color::WHITE ? Color::BLACK : Color::WHITE;
    const uint64_t kbit = b.pieces(PieceType::KING, them).getBits();
    if (!kbit) return false;
    const Square ksq(__builtin_ctzll(kbit));
    const int from = m.from().index(), to = m.to().index(), type = m.typeOf();
    const int pt = (int)b.at(m.from()).type();
    uint64_t pc[6];
    for (int i = 0; i < 6; ++i) pc[i] = b.pieces(PT_LIST[i], us).getBits();
    uint64_t occ = b.occ().getBits() & ~(1ULL << from);
    pc[pt] &= ~(1ULL << from);
    int dest = to, newPt = pt;
    if (type == Move::CASTLING) {                      // only the rook can check
        const int rank = from & ~7;
        occ &= ~(1ULL << to);
        pc[3] &= ~(1ULL << to);
        occ |= 1ULL << (rank + (to > from ? 6 : 2));   // king
        dest = rank + (to > from ? 5 : 3);             // rook
        newPt = 3;
    } else if (type == Move::ENPASSANT) {
        occ &= ~(1ULL << (us == Color::WHITE ? to - 8 : to + 8));
    } else if (type == Move::PROMOTION) {
        newPt = (int)m.promotionType();
    }
    occ |= 1ULL << dest;
    pc[newPt] |= 1ULL << dest;
    return (attacks::pawn(them, ksq).getBits() & pc[0]) ||
           (attacks::knight(ksq).getBits() & pc[1]) ||
           (attacks::bishop(ksq, Bitboard(occ)).getBits() & (pc[2] | pc[4])) ||
           (attacks::rook(ksq, Bitboard(occ)).getBits() & (pc[3] | pc[4]));
}

// checks: also try quiet checks, first qs ply only
static int quiesce(ThreadData& td, Board& b, int alpha, int beta, int ply, bool checks = false) {
    checkTime(td);
    td.posKey[ply] = b.hash();
    if (b.isInsufficientMaterial() || b.isHalfMoveDraw() || isRepeated(td, b, ply)) return drawScore(td);
    if (ply >= MAX_PLY) return evalAt(td, b, ply);

    const bool inCheck = b.inCheck();
    const int alphaOrig = alpha;

    uint64_t key = b.hash();
    Move ttMove{};
    int ttEvalQ = TT_NO_EVAL, rawQ = TT_NO_EVAL;       // tt eval / raw eval
    int ttScoreQ = 0, ttFlagQ = -1;                    // tt score / bound (-1 = none)
    TTEntry e;
    if (ttProbe(key, e)) {
        ttMove = Move(e.move);
        int s = fromTT(e.score, ply);
        ttScoreQ = s; ttFlagQ = e.flag(); ttEvalQ = e.eval;
        if (e.flag() == EXACT) return s;
        if (e.flag() == LOWER && s >= beta) return s;
        if (e.flag() == UPPER && s <= alpha) return s;
    }

    Movelist moves;
    int stand = -INF, best, refined = -INF;
    if (inCheck) {
        movegen::legalmoves(moves, b);
        if (moves.empty()) return -(MATE - ply);
        best = -INF;
    } else {
        rawQ = ttEvalQ != TT_NO_EVAL ? ttEvalQ : evalAt(td, b, ply);        // tt may have it
        stand = correctEval(td, b, sideIdx(b), ply, rawQ);    // same corrected eval as main search
        refined = stand;                                // tt score is better if its bound points that way
        if (ttFlagQ >= 0 && std::abs(ttScoreQ) < TB_WIN - 1000 &&
            ((ttFlagQ == LOWER && ttScoreQ > stand) || (ttFlagQ == UPPER && ttScoreQ < stand)))
            refined = ttScoreQ;
        if (refined >= beta) return refined;
        if (stand + BIG_DELTA < alpha) return stand + BIG_DELTA;
        if (refined > alpha) alpha = refined;
        best = refined;
        movegen::legalmoves<movegen::MoveGenType::CAPTURE>(moves, b);
    }
    orderMoves(td, b, moves, ttMove, ply, false);

    Move bestMove{};
    for (int i = 0; i < moves.size(); ++i) {
        pickMove(moves, i);
        const Move m = moves[i];
        if (!inCheck) {
            const bool promo = m.typeOf() == Move::PROMOTION;
            int victim = (m.typeOf() == Move::ENPASSANT) ? 0 : (int)b.at(m.to()).type();
            if (victim > 5) victim = 0;
            int attacker = (int)b.at(m.from()).type();

            int gain = PIECE_VALUE[victim];
            if (promo) gain += PIECE_VALUE[(int)m.promotionType()] - PIECE_VALUE[0];
            if (stand + gain + DELTA_MARGIN < alpha) continue;

            // losing caps already marked by orderMoves, tt move wasn't
            if (!promo && ORDER_VALUE[victim] < ORDER_VALUE[attacker] &&
                (m == ttMove ? see(b, m) < 0 : m.score() < 0)) continue;
        }

        const int movedPt = (int)b.at(m.from()).type();
        accSetMove(td, b, m, ply + 1);
        b.makeMove(m);
        td.stkPt[ply + 1] = movedPt; td.stkTo[ply + 1] = m.to().index(); td.stkQuiet[ply + 1] = false;   // keep corr context valid
        ttPrefetch(b.hash());
        int s = -quiesce(td, b, -beta, -alpha, ply + 1);
        b.unmakeMove(m);

        if (s > best) { best = s; bestMove = m; }
        if (s > alpha) alpha = s;
        if (alpha >= beta) break;
    }

    // quiet checks, first ply only
    if (checks && !inCheck && alpha < beta && stand + QCHECK_MARGIN >= alpha) {
        Movelist qm;
        movegen::legalmoves<movegen::MoveGenType::QUIET>(qm, b);
        for (int i = 0; i < qm.size(); ++i) {
            const Move m = qm[i];
            if (m.typeOf() == Move::PROMOTION || m.typeOf() == Move::CASTLING) continue;
            if (!quietGivesDirectCheck(b, m)) continue;
            const int movedPt = (int)b.at(m.from()).type();
            accSetMove(td, b, m, ply + 1);
            b.makeMove(m);
            td.stkPt[ply + 1] = movedPt; td.stkTo[ply + 1] = m.to().index(); td.stkQuiet[ply + 1] = false;
            ttPrefetch(b.hash());
            int s = -quiesce(td, b, -beta, -alpha, ply + 1);
            b.unmakeMove(m);
            if (s > best) { best = s; bestMove = m; }
            if (s > alpha) alpha = s;
            if (alpha >= beta) break;
        }
    }

    // depth 0 entries, ttStore won't evict real ones
    if (refined == stand || !(bestMove == Move())) {     // skip if score is just the tt's own estimate
        const int flag = best <= alphaOrig ? UPPER : (best >= beta ? LOWER : EXACT);
        ttStore(key, flag == UPPER ? 0 : bestMove.move(), toTT(best, ply), 0, flag, rawQ);
    }
    return best;
}

// ---- correction history ----
// tables learn how off the static eval is per context, combined into one adjustment
static inline uint64_t fold64(uint64_t h) { h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 32; return h; }

struct CorrIdx { int pawn, minor, npw, npb, cont; };
static inline CorrIdx corrIdx(const ThreadData& td, const Board& b, int ply) {
    auto bb = [&](PieceType t, Color c) { return b.pieces(t, c).getBits(); };
    const uint64_t wn = bb(PieceType::KNIGHT, Color::WHITE), wb = bb(PieceType::BISHOP, Color::WHITE),
                   wr = bb(PieceType::ROOK,   Color::WHITE), wq = bb(PieceType::QUEEN,  Color::WHITE),
                   bn = bb(PieceType::KNIGHT, Color::BLACK), bbp = bb(PieceType::BISHOP, Color::BLACK),
                   br = bb(PieceType::ROOK,   Color::BLACK), bq = bb(PieceType::QUEEN,  Color::BLACK);
    CorrIdx ix;
    ix.pawn  = (int)(pawnKey(b) & (CORR_SIZE - 1));
    ix.minor = (int)(fold64(wn * 0x9E3779B97F4A7C15ULL ^ wb * 0xC2B2AE3D27D4EB4FULL ^
                            bn * 0x165667B19E3779F9ULL ^ bbp * 0x27D4EB2F165667C5ULL) & (CORR_SIZE - 1));
    ix.npw   = (int)(fold64(wn * 0x9E3779B97F4A7C15ULL ^ wb * 0xC2B2AE3D27D4EB4FULL ^
                            wr * 0x165667B19E3779F9ULL ^ wq * 0x27D4EB2F165667C5ULL) & (CORR_SIZE - 1));
    ix.npb   = (int)(fold64(bn * 0x9E3779B97F4A7C15ULL ^ bbp * 0xC2B2AE3D27D4EB4FULL ^
                            br * 0x165667B19E3779F9ULL ^ bq * 0x27D4EB2F165667C5ULL) & (CORR_SIZE - 1));
    ix.cont  = td.stkTo[ply] >= 0 ? td.stkPt[ply] * 64 + td.stkTo[ply] : CORR_CONT_NONE;
    return ix;
}

// static eval + learned correction
static inline int correctEvalIx(const ThreadData& td, const CorrIdx& ix, int side, int halfMove, int eval) {
    const int sum = 4 * corrLoad(g_corr.pawn[side][ix.pawn]) + 2 * corrLoad(g_corr.minor[side][ix.minor]) +
                    2 * corrLoad(g_corr.nonPawn[0][side][ix.npw]) + 2 * corrLoad(g_corr.nonPawn[1][side][ix.npb]) +
                    2 * corrLoad(g_corr.cont[side][ix.cont]);
    int c = sum / (CORR_GRAIN * CORR_DIV);
    c = std::max(-CORR_MAX_TOTAL, std::min(CORR_MAX_TOTAL, c));
    // 50 move rule: eval shrinks as halfmove clock grows
    // done here not on raw eval so tt eval stays clock independent
    const int hm = std::min(halfMove, 100);
    const int v = (eval + c) * (EVAL_HM_BASE - hm) / EVAL_HM_BASE;
    return std::max(-20000, std::min(20000, v));
}
static inline int correctEval(const ThreadData& td, const Board& b, int side, int ply, int eval) {
    return correctEvalIx(td, corrIdx(td, b, ply), side, (int)b.halfMoveClock(), eval);
}

// pull entries toward (score - raw eval), deeper = more weight
static inline void updateCorr(ThreadData& td, const CorrIdx& ix, int side, int depth, int diff) {
    const int target = std::max(-CORR_MAX_CP, std::min(CORR_MAX_CP, diff)) * CORR_GRAIN;
    const int w = std::min(depth + 1, 16);
    (void)td;
    auto upd = [&](int16_t& e) { const int v = corrLoad(e); corrStore(e, v + (target - v) * w / 256); };
    upd(g_corr.pawn[side][ix.pawn]);
    upd(g_corr.minor[side][ix.minor]);
    upd(g_corr.nonPawn[0][side][ix.npw]);
    upd(g_corr.nonPawn[1][side][ix.npb]);
    upd(g_corr.cont[side][ix.cont]);
}

// corrected root eval, plies 2 and 4 read it for improving
static inline int rootStaticEval(ThreadData& td, const Board& b) {
    const int r = rootEval(td, b);
    return r == NO_EVAL ? NO_EVAL : correctEval(td, b, sideIdx(b), 0, r);
}

static bool hasNonPawnMaterial(const Board& b) {
    const Color c = b.sideToMove();
    return (b.pieces(PieceType::KNIGHT, c).getBits() | b.pieces(PieceType::BISHOP, c).getBits() |
            b.pieces(PieceType::ROOK, c).getBits()   | b.pieces(PieceType::QUEEN, c).getBits()) != 0;
}

// ---- main search ----
// cutNode: expected to fail high
static int negamax(ThreadData& td, Board& b, int depth, int alpha, int beta, int ply, bool canNull, bool cutNode, Move excluded = Move()) {
    checkTime(td);
    td.pvLen[ply] = ply;                              // empty pv until alpha is raised in a pv node
    td.posKey[ply] = b.hash();
    if (b.isInsufficientMaterial() || b.isHalfMoveDraw() || isRepeated(td, b, ply)) return drawScore(td);
    if (ply >= MAX_PLY) return evalAt(td, b, ply);

    // mate distance pruning (not at root)
    if (ply > 0) {
        alpha = std::max(alpha, -MATE + ply);
        beta  = std::min(beta,   MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }
#ifndef NO_CUCKOO
    // can force a repeat: worth at least a draw
    if (ply > 0 && alpha < 0 && hasUpcomingRepetition(td, b, ply)) {
        alpha = drawScore(td);
        if (alpha >= beta) return alpha;
    }
#endif

    const bool inCheck = b.inCheck();
    if (inCheck) ++depth;
    if (depth <= 0) return quiesce(td, b, alpha, beta, ply, true);      // true = quiet checks at first ply

    const bool pvNode = beta - alpha > 1;
    // singular verification: same pos, one move excluded. never touches the TT
    const bool singular = !(excluded == Move());
    if (!singular) {
        td.extCount[ply] = ply > 0 ? td.extCount[ply - 1] : 0;      // double exts so far
        td.cutoffCnt[ply + 2] = 0;
    }

    uint64_t key = b.hash();
    const int alphaOrig = alpha;
    Move ttMove{};
    int ttScore = 0, ttDepth = -1, ttFlag = 0, ttEval = TT_NO_EVAL;
    bool nodePv = pvNode;                              // pv now or was before (flag in TT)
    TTEntry e;
    if (!singular && ttProbe(key, e)) {
        ttMove = Move(e.move);
        ttScore = fromTT(e.score, ply); ttDepth = e.depth; ttFlag = e.flag(); ttEval = e.eval;
        if (e.pv) nodePv = true;
        // only cut in non-pv, keeps pv exact
        if (!pvNode && e.depth >= depth) {
            int s = fromTT(e.score, ply);
            const bool ttCut = e.flag() == EXACT || (e.flag() == LOWER && s >= beta) || (e.flag() == UPPER && s <= alpha);
            if (ttCut) return s;
        }
    }

    // ---- syzygy wdl probe ----
    // win/loss is just a bound, draw is exact
    int tbFloor = -INF, tbCeil = INF;
    if (!singular && ply > 0 && depth >= tbProbeDepth() && b.halfMoveClock() == 0 && tbCanProbe(b)) {
        int wdl;
        if (tbProbeWDL(b, wdl)) {
            int score, flag;
            if (wdl >= 2)       { score =  TB_WIN - ply; flag = LOWER; }
            else if (wdl <= -2) { score = -TB_WIN + ply; flag = UPPER; }
            else                { score = 0;             flag = EXACT; }   // incl cursed / blessed
            if (flag == EXACT || (flag == LOWER && score >= beta) || (flag == UPPER && score <= alpha)) {
                ttStore(key, 0, flag == LOWER ? TB_WIN : (flag == UPPER ? -TB_WIN : 0), std::min(depth + 6, 100), flag);
                return score;
            }
            if (pvNode) {
                if (flag == LOWER) { tbFloor = score; alpha = std::max(alpha, score); }
                else               tbCeil = score;
            }
        }
    }

    // iir
    // depends on node type: cut nodes reduce more, pv nodes with a recent exact entry skip it
    if (!singular && depth >= IIR_MIN_DEPTH && ttMove == Move()) {
        const bool recentExact = pvNode && ttDepth >= 0 && ttFlag == EXACT && ttDepth >= depth - IIR_PV_EXACT_SLACK;
        if (!recentExact) {
            const int d0 = depth;
            --depth;
            if (cutNode && d0 >= IIR_CUT_MIN_DEPTH) --depth;
        }
    }

    // static eval when not in check
    const bool prunable = !pvNode && !inCheck && std::abs(beta) < MATE_BOUND;
    const int side = sideIdx(b);
    int staticEval = NO_EVAL, rawEval = NO_EVAL;
    CorrIdx cix{};                                     // computed once, reused for ph, the eval and the corr update
    int ph;
    if (!inCheck) {
        cix = corrIdx(td, b, ply);
        ph = cix.pawn & (PAWN_HIST_SIZE - 1);
        rawEval = ttEval != TT_NO_EVAL ? ttEval : evalAt(td, b, ply);   // reuse tt eval
        staticEval = correctEvalIx(td, cix, side, (int)b.halfMoveClock(), rawEval);      // net + correction
    } else {
        ph = pawnHistIdx(b);
    }
    td.evalStack[ply] = staticEval;

    // pruneEval: tt score if its bound points the right way, else static
    // only for pruning, improving + corr update use static
    int pruneEval = staticEval;
    if (!inCheck && ttDepth >= 0 && std::abs(ttScore) < TB_WIN - 1000 &&
        (ttFlag == EXACT || (ttFlag == LOWER && ttScore > staticEval) || (ttFlag == UPPER && ttScore < staticEval)))
        pruneEval = ttScore;

    // improving: static eval better than 2 plies ago (4 if in check then)
    bool improving = false;
    int evalGain = 0;                                  // gain vs 2 (or 4) plies ago
    if (!inCheck) {
        int prev = ply >= 2 ? td.evalStack[ply - 2] : NO_EVAL;
        if (prev == NO_EVAL && ply >= 4) prev = td.evalStack[ply - 4];
        improving = prev != NO_EVAL && staticEval > prev;
        if (prev != NO_EVAL) evalGain = staticEval - prev;
    }

    if (prunable && !singular && depth <= RAZOR_MAX_DEPTH && pruneEval + RAZOR_BASE + RAZOR_MARGIN * depth <= alpha) {
        const int v = quiesce(td, b, alpha - 1, alpha, ply);
        if (v < alpha) return v;
    }

    // rfp, improving side needs a smaller margin
    if (prunable && !singular && depth <= RFP_MAX_DEPTH && pruneEval - RFP_MARGIN * (depth - improving) >= beta && !nodePv && hasNonPawnMaterial(b))
        return (pruneEval + beta) / 2;                 // blend toward beta instead of returning the raw eval (still >= beta)

    // --- null move ---
    if (!pvNode && canNull && !inCheck && depth >= NMP_MIN_DEPTH && ply >= td.nmpMinPly &&
        std::abs(beta) < MATE_BOUND && hasNonPawnMaterial(b) && pruneEval >= beta) {
        int R = NMP_BASE + depth / NMP_DIV + std::min((pruneEval - beta) / NMP_EVAL_DIV, NMP_EVAL_MAX);
        td.stkTo[ply + 1] = -1; td.stkPt[ply + 1] = 0;
        accSetNull(td, ply + 1);
        b.makeNullMove();
        int s = -negamax(td, b, depth - 1 - R, -beta, -beta + 1, ply + 1, false, false);
        b.unmakeNullMove();
        if (s >= beta) {
            if (s >= MATE_BOUND) s = beta;                        // don't trust unproven mates
            if (depth < NMP_VERIFY_DEPTH || td.nmpMinPly != 0) return s;
            // deep cutoff: verify without null moves (zugzwang)
            td.nmpMinPly = ply + 3 * (depth - 1 - R) / 4;
            const int v = negamax(td, b, depth - 1 - R, beta - 1, beta, ply, true, false);
            td.nmpMinPly = 0;
            if (v >= beta) return s;
        }
    }

    if (prunable && !singular && depth >= PROBCUT_MIN_DEPTH) {
        const int probBeta = beta + PROBCUT_MARGIN;
        if (!(ttDepth >= depth - 3 && ttFlag != LOWER && ttScore < probBeta)) {
            Movelist caps;
            movegen::legalmoves<movegen::MoveGenType::CAPTURE>(caps, b);
            orderMoves(td, b, caps, ttMove, ply, true, ph);
            for (int i = 0; i < caps.size(); ++i) {
                pickMove(caps, i);
                const Move m = caps[i];
                if (!b.isCapture(m) && m.typeOf() != Move::PROMOTION) continue;
                if (see(b,m) < probBeta - pruneEval) continue;
                const int movedPt = (int)b.at(m.from()).type();
                accSetMove(td, b, m, ply + 1);
                b.makeMove(m);
                td.stkPt[ply + 1] = movedPt; td.stkTo[ply + 1] = m.to().index(); td.stkQuiet[ply + 1] = false;
                int s = -quiesce(td, b, -probBeta, -probBeta + 1, ply + 1);
                if (s >= probBeta) {
                    s = -negamax(td, b, depth - PROBCUT_REDUCTION, -probBeta, -probBeta + 1, ply + 1, true, !cutNode);
                }
                b.unmakeMove(m);
                if (s >= probBeta) {
                    ttStore(key, m.move(), toTT(s, ply), depth - PROBCUT_REDUCTION + 1, LOWER);
                    return s - (probBeta - beta);
                }
            }
        }
    }

    Movelist moves;
    movegen::legalmoves(moves, b);
    if (moves.empty()) return inCheck ? -(MATE - ply) : 0;
    orderMoves(td, b, moves, ttMove, ply, true, ph);

    int best = tbFloor;
    Move bestMove{};
    Move quiets[64];
    int nq = 0;
    int alphaRaised = 0;                               // alpha raised earlier here, reduce later moves more
    Move captures[32];                                 // captures tried, for capture malus
    int ncap = 0;
    // lmr ctx: noisy tt move -> quiets probably worse, counter = quiet that already refuted the last move
    const bool ttNoisy = !(ttMove == Move()) && (b.isCapture(ttMove) || ttMove.typeOf() == Move::PROMOTION);
    const Move counter = td.stkTo[ply] >= 0 ? td.counterMove[td.stkPt[ply]][td.stkTo[ply]] : Move();
    for (int i = 0; i < moves.size(); ++i) {
        pickMove(moves, i);
        const Move m = moves[i];
        if (singular && m == excluded) continue;
        const bool quiet = !b.isCapture(m) && m.typeOf() != Move::PROMOTION;
        const bool isKiller = m == td.killers[ply][0] || m == td.killers[ply][1];
        const int movedPt = (int)b.at(m.from()).type();
        const int h = quiet ? histScore(td, side, ply, movedPt, m, ph)
                            : (b.isCapture(m) ? capHistScore(td, b, side, m) : 0);

        int ext = 0;
        bool dblUsed = false;                          // used up a double ext
        if (i == 0 && !singular && ply > 0 && depth >= SE_MIN_DEPTH && m == ttMove && ttFlag != UPPER && ttDepth >= depth - 3 && std::abs(ttScore) < MATE_BOUND) {
            const int sBeta = ttScore - SE_MARGIN * depth;
            const int sv = negamax(td, b, (depth - 1 ) / 2, sBeta - 1, sBeta, ply, false, cutNode, m);
            if (sv < sBeta) {
                ext = 1;
                if (sv < sBeta - (pvNode ? SE_DOUBLE_MARGIN_PV : SE_DOUBLE_MARGIN) && td.extCount[ply] < SE_MAX_DOUBLE) {   // very singular, extend twice
                    ext = 2;
                    ++td.extCount[ply];                // only this move's subtree sees it, undone below
                    dblUsed = true;
                }
            }
            else if (sBeta >= beta) return sBeta;
            else if (SE_NEG_CUT && cutNode) ext = -SE_NEG_CUT;   // cut node, not singular -> shallower
            else if (ttScore >= beta) ext = -1;
        }

        // SEE pruning: skip caps that clearly lose material at low depth. only ones orderMoves marked negative
        // never first move, pv, check nodes, or moves giving check
        const bool seeBad = prunable && i > 0 && depth <= SEE_PRUNE_MAX_DEPTH && best > -MATE_BOUND &&
                            !quiet && m.score() < 0 && see(b, m) < -SEE_PRUNE_MARGIN * depth;

        const bool capFut = prunable && i > 0 && !quiet && m.typeOf() != Move::PROMOTION && depth <= CFP_MAX_DEPTH &&
                            best > -MATE_BOUND && b.isCapture(m) &&
                            pruneEval + CFP_BASE + CFP_MARGIN * depth + PIECE_VALUE[capVictim(b, m)] <= alpha;

#ifdef GIVESCHECK_VERIFY
        {
            b.makeMove(m);
            const bool real = b.inCheck();
            b.unmakeMove(m);
            if (real != givesCheck(b, m)) {
                std::cerr << "givesCheck mismatch: real " << real << " fen " << b.getFen() << " move " << uci::moveToUci(m) << std::endl;
                std::abort();
            }
        }
#endif
        // prune before making the move
        if ((seeBad || capFut) && !givesCheck(b, m)) continue;

        // futility / lmp: skip hopeless quiets at low depth (not first move, killers, checks)
        if (prunable && i > 0 && quiet && !isKiller && best > -MATE_BOUND) {
            const bool lmp = depth <= LMP_MAX_DEPTH && i >= (3 + depth * depth) / (improving ? 1 : 2);
            const bool fut = depth <= FP_MAX_DEPTH && pruneEval + FP_BASE + FP_MARGIN * depth <= alpha;
            const bool hp  = depth <= HP_MAX_DEPTH && h < -HP_MARGIN * depth;      // history pruning
            if ((lmp || fut || hp) && !givesCheck(b, m)) continue;
        }

        accSetMove(td, b, m, ply + 1);
        b.makeMove(m);
        td.stkPt[ply + 1] = movedPt; td.stkTo[ply + 1] = m.to().index(); td.stkQuiet[ply + 1] = quiet;

        ttPrefetch(b.hash());                              // child probes TT right away
        int s;
        if (i == 0) {
            s = -negamax(td, b, depth - 1 + ext, -beta, -alpha, ply + 1, true, pvNode ? false : !cutNode);
        } else {
            // lmr: late quiets / bad caps get searched shallower
            // adjustments below push r up or down
            int r = 0;
            const bool badCapture = !quiet && m.score() < S_BAD_THRESH;
            if (depth >= LMR_MIN_DEPTH && i >= LMR_MIN_MOVES && (quiet || badCapture) && !inCheck && !b.inCheck()) {
                r = LMR[std::min(depth, 63)][std::min(i, 63)];
                if (pvNode) --r;                            // pv: reduce less
                if (isKiller) --r;
                if (m == counter) --r;                      // known refutation
                if (!improving) ++r;                        // eval stuck, reduce more
                if (cutNode) r += LMR_CUT_NODE;             // cut node
                if (cutNode && ttMove == Move()) ++r;       // ... and no info at all
                if (ttDepth >= depth) --r;                  // deep tt result, trust ordering less
                if (nodePv && !pvNode) --r;                 // was pv before
                if (td.cutoffCnt[ply + 1] > 3) ++r;         // replies keep failing high
                r += alphaRaised;
                if (LMR_EVAL_LOW_MARGIN  && !pvNode && staticEval + LMR_EVAL_LOW_MARGIN <= alpha) ++r;      // eval way below alpha
                if (LMR_EVAL_HIGH_MARGIN && cutNode && staticEval - LMR_EVAL_HIGH_MARGIN >= beta) ++r;      // eval way above beta at cut node
                if (LMR_STRONG_IMPROVE   && evalGain >= LMR_STRONG_IMPROVE) --r;                            // clearly improving
                if (LMR_TT_LOWER && !pvNode && ttFlag == LOWER && ttScore >= beta && ttDepth >= depth - 3) ++r;   // tt says fail high
                if (ttNoisy && quiet) ++r;                  // tt move is a capture, quiets rarely beat it
                if (badCapture) --r;                        // bad caps, careful
                r -= quiet ? h / 16384 : h / 4096;          // good hist reduce less, bad more (cap hist for captures)
                r = std::max(0, std::min(r, depth - 2));
            }
            s = -negamax(td, b, depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, !cutNode);
            if (s > alpha && r > 0) {                      // reduced search beat alpha, verify
                int nd = depth - 1;
                if (s > best + DEEPER_BASE + DEEPER_MUL * nd) ++nd;
                else if (s < best + nd) nd = std::max(nd - 1, depth - 1 - r);
                s = -negamax(td, b, nd, -alpha - 1, -alpha, ply + 1, true, !cutNode);
            }
            if (s > alpha && s < beta)                     // pvs re-search, full window
                s = -negamax(td, b, depth - 1, -beta, -alpha, ply + 1, true, false);
        }
        b.unmakeMove(m);
        if (dblUsed) --td.extCount[ply];

        if (s > best) { best = s; bestMove = m; }
        if (s > alpha) {
            alpha = s;
            if (depth > 2 && depth < 12 && std::abs(s) < MATE_BOUND) alphaRaised = 1;
            if (pvNode) {                                  // move + child pv = our pv
                td.pv[ply][ply] = m;
                const int cl = td.pvLen[ply + 1];
                for (int k = ply + 1; k < cl; ++k) td.pv[ply][k] = td.pv[ply + 1][k];
                td.pvLen[ply] = cl;
            }
        }
        if (alpha >= beta) {
            if (!singular) td.cutoffCnt[ply] += 1 + (ttMove == Move() ? 1 : 0) - (ext >= 2 ? 1 : 0);
            int bonus = std::min(HIST_B_MUL * depth - HIST_B_BASE, HIST_B_MAX);
            const int malus = std::min(HIST_M_MUL * depth - HIST_M_BASE, HIST_M_MAX);
            if (i > 0) bonus += bonus / 4;                  // came late, learn more
            if (quiet) {
                if (!(m == td.killers[ply][0])) { td.killers[ply][1] = td.killers[ply][0]; td.killers[ply][0] = m; }
                updateQuiet(td, b, side, ply, m, bonus, ph);
                for (int q = 0; q < nq; ++q) updateQuiet(td, b, side, ply, quiets[q], -malus, ph);   // punish failed quiets
                if (td.stkTo[ply] >= 0) td.counterMove[td.stkPt[ply]][td.stkTo[ply]] = m;
            } else if (b.isCapture(m)) {
                updateCapture(td, b, side, m, bonus);
            }
            // caps tried before the cutoff get a malus
            for (int c = 0; c < ncap; ++c) updateCapture(td, b, side, captures[c], -malus);
            break;
        }
        if (quiet && nq < 64) quiets[nq++] = m;
        else if (!quiet && ncap < 32 && b.isCapture(m)) captures[ncap++] = m;
    }

    if (pvNode && best > tbCeil) best = tbCeil;       // tb loss caps the score

    // fail low: reward the move into this node in cont tables
    if (!singular && best <= alphaOrig && depth >= 2 && td.stkTo[ply] >= 0 && td.stkQuiet[ply]) {
        const int pb = std::min(HIST_B_MUL * depth - HIST_B_BASE, HIST_B_MAX) / PRIOR_DIV;
        const int prevCur = curIdx(side ^ 1, td.stkPt[ply], td.stkTo[ply]);
        for (int k = 0; k < 3; ++k) {
            const int p = ply - 1 - CH_BACK[k];
            if (p >= 0 && td.stkTo[p] >= 0) gravity(td.contHist[k][td.stkPt[p] * 64 + td.stkTo[p]][prevCur], pb);
        }
    }

    // fail high blend: cutoff overshoots, pull toward beta (stays >= beta). not for mate/tb
    const int rawBest = best;
    if (best >= beta && depth >= FH_MIN_DEPTH && std::abs(best) < TB_WIN - 1000 && std::abs(beta) < TB_WIN - 1000)
        best = (best * (depth + 2) + beta) / (depth + 3);

    // fail low = no real best move, store none
    if (!singular) {
        int flag = best <= alphaOrig ? UPPER : (best >= beta ? LOWER : EXACT);
        if (flag == EXACT && tbFloor > -INF && best == tbFloor) flag = LOWER;   // tb floor is only a lower bound
        ttStore(key, flag == UPPER ? 0 : bestMove.move(), toTT(best, ply), depth, flag, inCheck ? TT_NO_EVAL : rawEval, nodePv ? 1 : 0);
        // learn unless best is noisy or the bound says nothing about eval error
        if (!inCheck && std::abs(rawBest) < TB_WIN - 1000 &&
            (bestMove == Move() || !(b.isCapture(bestMove) || bestMove.typeOf() == Move::PROMOTION)) &&
            !(flag == LOWER && rawBest <= rawEval) && !(flag == UPPER && rawBest >= rawEval))
            updateCorr(td, cix, side, depth, rawBest - rawEval);
    }
    return best;
}

static std::string scoreStr(int s) {
    if (s >= MATE_BOUND)  return "mate " + std::to_string((MATE - s + 1) / 2);
    if (s <= -MATE_BOUND) return "mate -" + std::to_string((MATE + s) / 2);
    return "cp " + std::to_string(s);
}

// pv string, falls back to best move
static std::string pvString(const Move* pv, int len, const Move& best) {
    std::string s;
    if (len > 0 && pv[0] == best) {
        for (int i = 0; i < len; ++i) s += (i ? " " : "") + uci::moveToUci(pv[i]);
    } else {
        s = uci::moveToUci(best);
    }
    return s;
}

// save finished iteration for picking best thread
static void recordDone(ThreadData& td, int depth, int score, const Move& best) {
    td.doneDepth = depth;
    td.doneScore = score;
    td.doneBest  = best;
    int n = 0;
    if (td.pvLen[0] > 0 && td.pv[0][0] == best) {
        n = td.pvLen[0];
        for (int i = 0; i < n; ++i) td.donePv[i] = td.pv[0][i];
    }
    td.donePvLen = n;
}

// ---- root search ----
static int rootSearch(ThreadData& td, Board& b, Movelist& moves, int depth, int alpha, int beta, Move& best) {
    td.posKey[0] = b.hash();
    td.nmpMinPly = 0;
    td.cutoffCnt[1] = td.cutoffCnt[2] = 0;            // root isn't a negamax node so clear these here
    // first pass: normal ordering, after that keep the order from last pass
    if (td.rootKey != td.posKey[0]) {
        td.rootKey = td.posKey[0];
        std::memset(td.rootNodes, 0, sizeof(td.rootNodes));
        orderMoves(td, b, moves, best, 0);
        std::stable_sort(moves.begin(), moves.end(),
                         [](const Move& x, const Move& y) { return x.score() > y.score(); });
    }
    int bestScore = -INF;
    bool first = true;
    const int alphaStart = alpha;
    const bool rootInCheck = b.inCheck();
    const int rootSide = sideIdx(b);
    const int rootPh = pawnHistIdx(b);
    int moveIdx = -1, bestIdx = -1;
    for (const Move& m : moves) {
        ++moveIdx;
        const uint64_t n0 = td.nodes;
        const int rootPt = (int)b.at(m.from()).type();
        const bool rootQuiet = !b.isCapture(m) && m.typeOf() != Move::PROMOTION;
        td.stkPt[1] = (int)b.at(m.from()).type(); td.stkTo[1] = m.to().index();
        td.stkQuiet[1] = !b.isCapture(m) && m.typeOf() != Move::PROMOTION;
        accSetMove(td, b, m, 1);
        b.makeMove(m);
        int s;
        if (first) {
            s = -negamax(td, b, depth - 1, -beta, -alpha, 1, true, false);
        } else {
            int r = 0;
            if (ROOT_LMR_MIN_DEPTH && depth >= ROOT_LMR_MIN_DEPTH && moveIdx >= ROOT_LMR_MIN_MOVES && rootQuiet &&
                !rootInCheck && !b.inCheck()) {
                r = LMR[std::min(depth, 63)][std::min(moveIdx, 63)] - 1;          // root matters, reduce less
                r -= histScore(td, rootSide, 0, rootPt, m, rootPh) / 16384;
                r = std::max(0, std::min(r, depth - 3));
            }
            s = -negamax(td, b, depth - 1 - r, -alpha - 1, -alpha, 1, true, true);
            if (s > alpha && r > 0)                                                // verify at full depth
                s = -negamax(td, b, depth - 1, -alpha - 1, -alpha, 1, true, true);
            if (s > alpha && s < beta)
                s = -negamax(td, b, depth - 1, -beta, -alpha, 1, true, false);
        }
        b.unmakeMove(m);
        first = false;
        td.rootNodes[m.from().index() * 64 + m.to().index()] = td.nodes - n0;

        if (s > bestScore) bestScore = s;
        if (s > alpha) {
            alpha = s;
            best = m;
            bestIdx = moveIdx;
            td.rootBestNodes = td.nodes - n0;
            td.pv[0][0] = m;
            const int cl = td.pvLen[1];
            for (int k = 1; k < cl; ++k) td.pv[0][k] = td.pv[1][k];
            td.pvLen[0] = std::max(cl, 1);
            if (alpha >= beta) break;
        }
    }
    // root hist: finished pass rewards best quiet, punishes earlier quiets
    if (ROOT_HIST_UPDATE && depth >= 4 && bestIdx >= 0 && bestScore > alphaStart && bestScore < beta &&
        !b.isCapture(best) && best.typeOf() != Move::PROMOTION) {
        const int bonus = std::min(HIST_B_MUL * depth - HIST_B_BASE, HIST_B_MAX) / 2;
        updateQuiet(td, b, rootSide, 0, best, bonus, rootPh);
        for (int k = 0; k < bestIdx && k < 16; ++k) {
            const Move q = moves[k];
            if (!b.isCapture(q) && q.typeOf() != Move::PROMOTION) updateQuiet(td, b, rootSide, 0, q, -bonus, rootPh);
        }
    }
    // next pass: best first, rest by nodes spent
    const Move bestNow = best;
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& x, const Move& y) {
        const uint64_t nx = x == bestNow ? UINT64_MAX : td.rootNodes[x.from().index() * 64 + x.to().index()];
        const uint64_t ny = y == bestNow ? UINT64_MAX : td.rootNodes[y.from().index() * 64 + y.to().index()];
        return nx > ny;
    });
    return bestScore;
}

// one iteration w/ aspiration, main + helpers
// iterNodes = nodes of the last pass
static int aspirationSearch(ThreadData& td, Board& b, Movelist& moves, int d, int lastScore,
                            Move& best, uint64_t& iterNodes) {
    int alpha = -INF, beta = INF, delta = ASP_DELTA;
    if (d >= ASP_MIN_DEPTH && std::abs(lastScore) < MATE_BOUND) {
        alpha = std::max(lastScore - delta, -INF);
        beta  = std::min(lastScore + delta,  INF);
    }
    int score, failHigh = 0;
    while (true) {
        const uint64_t n0 = td.nodes;
        const int sd = std::max(1, d - ASP_FH_REDUCE * std::min(failHigh, ASP_FH_MAX));
        score = rootSearch(td, b, moves, sd, alpha, beta, best);
        iterNodes = td.nodes - n0;
        if (score <= alpha) {
            beta  = (alpha + beta) / 2;
            alpha = std::max(score - delta, -INF);
            failHigh = 0;
        } else if (score >= beta) {
            beta = std::min(score + delta, INF);
            ++failHigh;
        } else {
            break;
        }
        delta += delta / 2;
        if (delta > ASP_MAX_DELTA) { alpha = -INF; beta = INF; failHigh = 0; }   // full window = full depth
    }
    return score;
}

// ---- lazy smp helper ----
// helpers share only the TT, own history
// skip iterations in different patterns so threads don't run in lock-step
// no printing, main thread stops them
static const int SKIP_SIZE[20]  = {1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4};
static const int SKIP_PHASE[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};

static void helperMain(ThreadData& td, Board b, Movelist moves, int maxDepth) {
    try {
        Move best = moves[0];
        td.stkTo[0] = -1; td.stkPt[0] = 0;
        td.pvLen[0] = 0;
        td.evalStack[0] = rootStaticEval(td, b);
        const int idx = (td.id - 1) % 20;
        int prevScore = 0;
        for (int d = 1; d <= maxDepth; ++d) {
            if (abortFlag.load(std::memory_order_relaxed) || stopFlag.load(std::memory_order_relaxed)) break;
            if (((d + SKIP_PHASE[idx]) / SKIP_SIZE[idx]) & 1) continue;
            uint64_t iterNodes = 0;
            prevScore = aspirationSearch(td, b, moves, d, prevScore, best, iterNodes);
            recordDone(td, d, prevScore, best);
            if (std::abs(prevScore) >= MATE_BOUND) break;
        }
    } catch (const Timeout&) {
        // main said stop
    }
}

// ---- picking the final move ----
// threads vote with (score - min score + 14) * depth
// main wins if it hit the requested depth
// mate scores: take the best score instead of voting
static int pickBestThread(int maxDepth) {
    const int n = (int)g_threads.size();
    int first = -1;
    for (int i = 0; i < n; ++i)
        if (g_threads[i]->doneDepth > 0) { first = i; break; }
    if (first < 0) return 0;
    const ThreadData& mainTd = *g_threads[0];
    if (mainTd.doneDepth >= maxDepth) return 0;

    bool mate = mainTd.doneDepth > 0 && mainTd.doneScore <= -MATE_BOUND;
    for (int i = 0; i < n; ++i)
        if (g_threads[i]->doneDepth > 0 && g_threads[i]->doneScore >= MATE_BOUND) mate = true;
    int best = first;
    if (mate) {
        for (int i = 0; i < n; ++i)
            if (g_threads[i]->doneDepth > 0 && g_threads[i]->doneScore > g_threads[best]->doneScore) best = i;
        return best;
    }

    int minScore = INF;
    for (int i = 0; i < n; ++i)
        if (g_threads[i]->doneDepth > 0) minScore = std::min(minScore, g_threads[i]->doneScore);

    struct Vote { uint16_t mv; int64_t v; };
    std::vector<Vote> votes;
    for (int i = 0; i < n; ++i) {
        const ThreadData& t = *g_threads[i];
        if (t.doneDepth <= 0) continue;
        const int64_t w = (int64_t)(t.doneScore - minScore + 14) * t.doneDepth;
        const uint16_t mv = t.doneBest.move();
        bool found = false;
        for (Vote& x : votes) if (x.mv == mv) { x.v += w; found = true; break; }
        if (!found) votes.push_back({mv, w});
    }
    auto votesFor = [&](const Move& m) -> int64_t {
        for (const Vote& x : votes) if (x.mv == m.move()) return x.v;
        return 0;
    };
    for (int i = 0; i < n; ++i) {
        const ThreadData& t = *g_threads[i];
        if (t.doneDepth <= 0) continue;
        if (votesFor(t.doneBest) > votesFor(g_threads[best]->doneBest)) best = i;   // ties keep lower index
    }
    return best;
}

// ---- iterative deepening + time mgmt ----
// hard = abort mid iteration, soft = checked after each iteration, scaled by stability
static const double STAB_SCALE[9] = {1.40, 1.20, 1.05, 0.95, 0.88, 0.82, 0.78, 0.75, 0.72};

// board is trashed on timeout, pass a copy
Move search(Board& b, const SearchLimits& lim) {
    if (!nnue::loaded()) {                             // no hce fallback, no net = no search
        std::cout << "info string ERROR: no NNUE net loaded, refusing to search" << std::endl;
        return Move();
    }
    initLMR();
    ensureThreads();
    const int64_t startNs = nowNs();
    {
        std::lock_guard<std::mutex> lk(g_timeMx);
        g_hardNs = lim.hardMs * 1000000LL;
        g_softNs = lim.softMs * 1000000LL;
        g_searchStartNs = startNs;
        // no hard limit while pondering
        g_deadlineNs = g_pondering.load() ? INT64_MAX : startNs + g_hardNs;
    }
    abortFlag = false;
    g_totalNodes = 0;
    g_lastPv.clear();                               // early returns = no pv
    const uint64_t rootKey = b.hash();              // b is mid-search after timeout
    ttAge = (ttAge + 1) & 15;                       // new tt generation
    for (auto& up : g_threads) {
        ThreadData& t = *up;
        t.nodes = 0;
        t.rootBestNodes = 0;
        t.rootKey = 0;                                  // fresh root order
        std::memset(t.lowPlyHist, 0, sizeof(t.lowPlyHist));
        t.pvLen[0] = 0;
        t.doneDepth = t.doneScore = t.donePvLen = 0;
        t.doneBest = Move();
        for (auto& k : t.killers) k[0] = k[1] = Move();
        for (auto& s : t.history) for (auto& f : s) for (int& h : f) h /= 2;   // age, don't wipe
        for (auto& a : t.capHist) for (auto& b2 : a) for (auto& c : b2) for (int& h : c) h /= 2;
    }

    Movelist moves;
    movegen::legalmoves(moves, b);
    if (moves.empty()) return Move();
    if (moves.size() == 1 && !lim.infinite) return moves[0];   // forced move, save the clock

    // ---- syzygy root probe (needs dtz files) ----
    // win/loss: play fathom's move. else only search moves that keep the result
    if (!lim.infinite && tbCanProbe(b)) {
        TBRoot r;
        if (tbProbeRoot(b, r)) {
            if (r.wdl >= 2 || r.wdl <= -2) {
                if (printInfo)
                    std::cout << "info depth 1 score cp " << (r.wdl > 0 ? TB_WIN : -TB_WIN) << " nodes 0 tbhits 1 pv " << uci::moveToUci(r.best) << std::endl;
                return r.best;
            }
            moves = r.keep;
            if (moves.size() == 1) return moves[0];
        }
    }

    ThreadData& td = *g_threads[0];
    Move best = moves[0], prevBest = best;
    int prevScore = 0, stability = 0;
    double moveChanges = 0.0;                          // decaying best move change count
    td.stkTo[0] = -1; td.stkPt[0] = 0;                            // nothing led into root
    td.evalStack[0] = rootStaticEval(td, b);        // read by plies 2 and 4

    // start helpers on root copies
    std::vector<std::thread> helpers;
    for (int i = 1; i < (int)g_threads.size(); ++i)
        helpers.emplace_back(helperMain, std::ref(*g_threads[i]), b, moves, lim.maxDepth);

    for (int d = 1; d <= lim.maxDepth; ++d) {
        try {
            const int lastScore = prevScore;
            uint64_t iterNodes = 0;
            const int score = aspirationSearch(td, b, moves, d, lastScore, best, iterNodes);
            recordDone(td, d, score, best);

            prevScore = score;
            const int64_t now = nowNs();
            const long long t = (now - startNs) / 1000000;
            if (printInfo) {
                const uint64_t total = totalNodes();
                std::cout << "info depth " << d << " score " << scoreStr(score)
                          << " nodes " << total << " nps " << (t > 0 ? total * 1000 / t : total)
                          << " hashfull " << hashfull() << " time " << t
                          << " pv " << pvString(td.pv[0], td.pvLen[0], best) << std::endl;
            }
            if (score >= MATE_BOUND) break;

            // ---- time mgmt ----
            moveChanges = moveChanges * TM_CHANGE_DECAY + ((d >= 2 && !(best == prevBest)) ? 1.0 : 0.0);
            stability = (d >= 2 && best == prevBest) ? std::min(stability + 1, 8) : 0;
            prevBest = best;
            // no clock while pondering, after ponderhit ponder time counts too
            if (!lim.infinite && !g_pondering.load()) {
                double target = (double)lim.softMs;
                if (lim.dynamic && d >= 6) {
                    double scale = STAB_SCALE[stability];                       // stable -> less time
                    const int drop = lastScore - score;
                    if (drop > 0)        scale *= 1.0 + std::min(drop, 80) * 0.006;   // dropping -> more time
                    else if (drop < -20) scale *= 0.95;                         // improving -> relax
                    scale *= 1.0 + TM_CHANGE_WEIGHT * std::min(moveChanges, 3.0);   // move keeps changing -> more time
                    if (d >= 8 && iterNodes > 0) {                              // dominant root move -> less time
                        double frac = (double)td.rootBestNodes / (double)iterNodes;
                        scale *= std::max(0.7, std::min(1.5, 1.7 - frac));
                        if (d >= 10 && stability >= 6 && frac > 0.9 && drop <= 10) scale *= 0.8;   // easy move
                    }
                    scale = std::max(0.4, std::min(scale, 2.5));
                    target = std::min(target * scale, (double)lim.hardMs);
                }
                if ((double)t >= target) break;
            }
        } catch (const Timeout&) {
            break;
        }
    }

    abortFlag = true;                                 // stop helpers, wait
    for (auto& h : helpers) h.join();
    g_totalNodes = totalNodes();

    // threads vote, main's live result used if it wins
    const int pick = pickBestThread(lim.maxDepth);
    Move result = best;
    g_lastPvKey = rootKey;
    if (pick == 0) {
        if (td.pvLen[0] > 0 && td.pv[0][0] == best) g_lastPv.assign(td.pv[0], td.pv[0] + td.pvLen[0]);
    } else {
        const ThreadData& t = *g_threads[pick];
        result = t.doneBest;
        g_lastPv.assign(t.donePv, t.donePv + t.donePvLen);
        if (printInfo)
            std::cout << "info depth " << t.doneDepth << " score " << scoreStr(t.doneScore)
                      << " nodes " << g_totalNodes << " pv " << pvString(t.donePv, t.donePvLen, t.doneBest) << std::endl;
    }
    return result;
}

// fixed depth + time budget overload
Move search(Board& b, int maxDepth, int ms) {
    setPondering(false);
    SearchLimits lim;
    lim.maxDepth = maxDepth;
    lim.hardMs = ms;
    lim.softMs = (int64_t)(ms * 0.6);
    lim.dynamic = true;
    return search(b, lim);
}

// last resort ponder move (forced/tb/book have no pv): tiny search, ~30ms. only when idle
static Move quickReply(const Board& after) {
    Movelist moves;
    movegen::legalmoves(moves, after);
    if (moves.empty()) return Move();
    if (moves.size() == 1) return moves[0];
    ensureThreads();
    ThreadData& td = *g_threads[0];
    Board b = after;
    td.stkTo[0] = -1; td.stkPt[0] = 0;
    td.evalStack[0] = rootStaticEval(td, b);
    abortFlag = false;                                // search left it set
    g_deadlineNs = nowNs() + 30000000LL;
    Move best = moves[0];
    int score = 0;
    try {
        for (int d = 1; d <= 8; ++d) {
            uint64_t iterNodes = 0;
            score = aspirationSearch(td, b, moves, d, score, best, iterNodes);
            if (std::abs(score) >= MATE_BOUND) break;
        }
    } catch (const Timeout&) {
        // time up, use what we have
    }
    return best;
}

// ponder move for bestmove: from pv, then TT, then quick search
// returns Move() if nothing found
Move ponderMoveFor(const Board& root, Move best) {
    if (best == Move() || !nnue::loaded()) return Move();
    Board b = root;
    b.makeMove(best);
    Movelist legal;
    movegen::legalmoves(legal, b);
    if (legal.empty()) return Move();
    auto legalMatch = [&](const Move& c) -> Move {
        for (const Move& m : legal)
            if (m == c) return m;
        return Move();
    };

    if (g_lastPvKey == root.hash() && g_lastPv.size() >= 2 && g_lastPv[0] == best) {
        const Move m = legalMatch(g_lastPv[1]);
        if (!(m == Move())) return m;
    }
    TTEntry e;
    if (ttProbe(b.hash(), e) && e.move != 0) {
        const Move m = legalMatch(Move(e.move));
        if (!(m == Move())) return m;
    }
    if (stopRequested()) return Move();
    return quickReply(b);
}
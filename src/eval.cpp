#include "eval.hpp"
#include "nnue.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

using namespace chess;

namespace {

typedef uint64_t U64;

inline int popcnt(U64 b) { return __builtin_popcountll(b); }
inline int popLsb(U64& b) { int s = __builtin_ctzll(b); b &= b - 1; return s; }

// material values for the endgame rules below (P N B R Q)
const int MG_VAL[5] = {100, 336, 338, 500, 966};
const int PHASE_W[5] = {0, 1, 1, 2, 4};      // total 24 at the start
const U64 LIGHT_SQ = 0x55AA55AA55AA55AAULL;

const PieceType PT[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                         PieceType::ROOK, PieceType::QUEEN, PieceType::KING};

} // namespace

// ---------- endgame knowledge on top of the net ----------
// the net has to learn mop-up and drawish endings from data with no mate scores and few positions under ten pieces,
// so these rules cover that. only fires in thin endings (phase <= 10), everything else costs one early exit
static int g_nnEgMask = 3;
static const int NN_EG_MOPUP = 1, NN_EG_DRAWISH = 2, NN_EG_OCB = 4;

int nnueAdjust(const Board& b, int stmScore) {
    const int mask = g_nnEgMask;
    if (!mask) return stmScore;

    int cnt[2][5];                                          // P N B R Q per colour
    U64 bishops[2];
    int phase = 0;
    for (int c = 0; c < 2; ++c) {
        const Color col = c == 0 ? Color::WHITE : Color::BLACK;
        for (int pt = 0; pt < 5; ++pt) {
            const U64 bb = b.pieces(PT[pt], col).getBits();
            cnt[c][pt] = popcnt(bb);
            if (pt == 2) bishops[c] = bb;
            if (pt > 0) phase += cnt[c][pt] * PHASE_W[pt];
        }
    }
    if (phase > 10) return stmScore;                        // not a thin ending: leave the net alone

    const int stm = b.sideToMove() == Color::WHITE ? 0 : 1;
    int white = stm == 0 ? stmScore : -stmScore;            // work from White's point of view

    int mat[2], np[2];                                      // material (pawns included) / non-pawn material
    for (int c = 0; c < 2; ++c) {
        np[c] = 0;
        for (int pt = 1; pt < 5; ++pt) np[c] += cnt[c][pt] * MG_VAL[pt];
        mat[c] = np[c] + cnt[c][0] * MG_VAL[0];
    }

    // mop-up: a clearly winning side in a thin endgame drives the enemy king to the edge
    if (mask & NN_EG_MOPUP) {
        const U64 wk = b.pieces(PieceType::KING, Color::WHITE).getBits();
        const U64 bk = b.pieces(PieceType::KING, Color::BLACK).getBits();
        const int diff = mat[0] - mat[1];
        if (std::abs(diff) >= 300 && wk && bk) {
            const int kingSq[2] = {__builtin_ctzll(wk), __builtin_ctzll(bk)};
            const int w = diff > 0 ? 0 : 1, l = w ^ 1;
            const int lf = kingSq[l] & 7, lr = kingSq[l] >> 3, wf = kingSq[w] & 7, wr = kingSq[w] >> 3;
            const int cornerDist = std::max(3 - lf, lf - 4) + std::max(3 - lr, lr - 4);   // 0 centre .. 6 corner
            const int kingDist = std::abs(lf - wf) + std::abs(lr - wr);
            const int bonus = (6 * cornerDist + 3 * (14 - kingDist)) * (24 - phase) / 24;   // endgame weight
            white += w == 0 ? bonus : -bonus;
        }
    }

    // drawish endings: shrink the score of the side that is ahead (sf out of 64, blended by phase)
    if (mask & (NN_EG_DRAWISH | NN_EG_OCB)) {
        const int strong = white > 0 ? 0 : 1, weak = strong ^ 1;
        const int pawnsStrong = cnt[strong][0], pawnsWeak = cnt[weak][0];
        const bool noNRQ = !(cnt[0][1] | cnt[1][1] | cnt[0][3] | cnt[1][3] | cnt[0][4] | cnt[1][4]);
        const bool noNQ  = !(cnt[0][1] | cnt[1][1] | cnt[0][4] | cnt[1][4]);
        const bool ocb = cnt[0][2] == 1 && cnt[1][2] == 1 &&
                         ((bishops[0] & LIGHT_SQ) != 0) != ((bishops[1] & LIGHT_SQ) != 0);
        int sf = 64;
        if ((mask & NN_EG_OCB) && ocb && noNRQ)
            sf = std::min(64, 22 + 3 * pawnsStrong);                         // opposite-coloured bishops only
        else if ((mask & NN_EG_OCB) && ocb && noNQ && cnt[0][3] == 1 && cnt[1][3] == 1)
            sf = 48;                                                         // ... with one rook each
        else if ((mask & NN_EG_DRAWISH) && pawnsStrong == 0 && pawnsWeak == 0 && np[strong] - np[weak] <= 340)
            sf = 6;                                                          // no pawns at all, at most a minor piece ahead
        else if ((mask & NN_EG_DRAWISH) && pawnsStrong == 0 && pawnsWeak == 0 && np[weak] == 0 &&
                 np[strong] == 2 * MG_VAL[1] && cnt[strong][1] == 2)
            sf = 6;                                                          // two knights against a bare king
        if (sf != 64) white = white * (phase * 64 + (24 - phase) * sf) / (24 * 64);
    }
    return stm == 0 ? white : -white;
}

namespace {
// ---------- NNUE ----------
// Score from the side to move's point of view, computed from scratch. The search keeps an incremental
// accumulator instead (search.cpp, evalAt); this path serves evaluate()/evalSide() outside the search.
static int nnueSide(const Board& b) {
    nnue::Accumulator acc;
    nnueRefresh(b, acc);
    const int bucket = nnue::outputBucket(popcnt(b.occ().getBits()));
    int s = nnue::evaluate(acc, b.sideToMove() == Color::WHITE ? 0 : 1, bucket);
    s = nnueAdjust(b, s);
    return std::max(-20000, std::min(20000, s));        // stay far below the tablebase / mate scores
}

// eval cache: index = low hash bits, check = high 32 bits. shared by all threads, one 64-bit slot
// (check << 32 | score) read and written atomically so a slot never mixes two positions
constexpr size_t EVAL_CACHE_SIZE = 1 << 20;                      // 8 MB
std::vector<uint64_t> evalCache(EVAL_CACHE_SIZE, 0);
}

void nnueRefresh(const Board& b, nnue::Accumulator& acc) {
    // Each perspective's features depend on where ITS OWN king stands (king bucket + left/right mirroring),
    // so find both kings first. A position without a king cannot occur in legal play; key 0 keeps it safe.
    int key[2] = {0, 0};
    for (int p = 0; p < 2; ++p) {
        const U64 k = b.pieces(PieceType::KING, p == 0 ? Color::WHITE : Color::BLACK).getBits();
        if (k) key[p] = nnue::kingKey(p, __builtin_ctzll(k));
    }
    nnue::reset(acc);
    for (int c = 0; c < 2; ++c) {
        const Color col = c == 0 ? Color::WHITE : Color::BLACK;
        for (int pt = 0; pt < 6; ++pt) {
            U64 bb = b.pieces(PT[pt], col).getBits();
            while (bb) {
                const int sq = popLsb(bb);
                nnue::addFeature(acc, 0, nnue::featureIndex(0, key[0], c, pt, sq));
                nnue::addFeature(acc, 1, nnue::featureIndex(1, key[1], c, pt, sq));
            }
        }
    }
}

// no net loaded = 0. callers must check nnueLoaded(), there is no fallback eval
int evaluate(const Board& b) {
    if (!nnue::loaded()) return 0;
    const int s = nnueSide(b);
    return b.sideToMove() == Color::WHITE ? s : -s;
}

int evalSide(const Board& b) {
    if (!nnue::loaded()) return 0;
    const uint64_t key = b.hash();
    uint64_t& slot = evalCache[key & (EVAL_CACHE_SIZE - 1)];
    const uint32_t check = (uint32_t)(key >> 32) | 1u;           // "| 1" keeps an empty slot from matching
    const uint64_t v = __atomic_load_n(&slot, __ATOMIC_RELAXED);
    if ((uint32_t)(v >> 32) == check) return (int)(int32_t)(uint32_t)v;
    int s = evaluate(b);
    s = b.sideToMove() == Color::WHITE ? s : -s;
    __atomic_store_n(&slot, ((uint64_t)check << 32) | (uint32_t)s, __ATOMIC_RELAXED);
    return s;
}

// ---------- net loading (call from the UCI "EvalFile" option, never while searching) ----------
bool loadNNUE(const std::string& path, std::string* err) {
    const bool ok = nnue::load(path, err);
    std::fill(evalCache.begin(), evalCache.end(), 0);   // cached scores belong to the old eval
    return ok;
}
#ifdef EMBED_NET
// the net compiled into the binary (embedded_net.cpp)
const unsigned char* embeddedNetData();
size_t embeddedNetSize();
bool loadEmbeddedNNUE(std::string* err) {
    const bool ok = nnue::loadFromMemory(embeddedNetData(), embeddedNetSize(), err);
    std::fill(evalCache.begin(), evalCache.end(), 0);
    return ok;
}
#else
bool loadEmbeddedNNUE(std::string* err) {
    if (err) *err = "this build has no embedded net (compile with -DEMBED_NET and add embedded_net.cpp)";
    return false;
}
#endif

void unloadNNUE() {
    nnue::unload();
    std::fill(evalCache.begin(), evalCache.end(), 0);
}
bool nnueLoaded() { return nnue::loaded(); }

void setNNUEEndgame(int mask) {
    g_nnEgMask = std::max(0, std::min(mask, 7));
    std::fill(evalCache.begin(), evalCache.end(), 0);   // cached scores were computed with the old setting
}
int nnueEndgame() { return g_nnEgMask; }
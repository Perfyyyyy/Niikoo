// test harness: ./nnue_test net.bin positions.txt  -> one score per line
#include "../nnue.hpp"
#include <cstdio>
int main(int argc, char** argv) {
    if (argc < 3) return 2;
    std::string err;
    if (!nnue::load(argv[1], &err)) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    FILE* f = std::fopen(argv[2], "r");
    int stm, n;
    while (std::fscanf(f, "%d %d", &stm, &n) == 2) {
        int col[40], pt[40], sq[40], king[2] = {0, 0};
        for (int i = 0; i < n; ++i) {
            if (std::fscanf(f, "%d %d %d", &col[i], &pt[i], &sq[i]) != 3) return 3;
            if (pt[i] == 5) king[col[i]] = sq[i];
        }
        nnue::Accumulator acc;
        for (int p = 0; p < 2; ++p) {
            nnue::resetPerspective(acc, p);
            const int key = nnue::kingKey(p, king[p]);
            for (int i = 0; i < n; ++i) nnue::addFeature(acc, p, nnue::featureIndex(p, key, col[i], pt[i], sq[i]));
        }
        std::printf("%d\n", nnue::evaluate(acc, stm, nnue::outputBucket(n)));
    }
    return 0;
}

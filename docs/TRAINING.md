# Training the network

The engine evaluates positions with a neural network (NNUE) trained on Stockfish evaluations from the
[Lichess evaluation database](https://database.lichess.org/) (CC0).

## Architecture (v3)
king-bucketed inputs (16 x 768) -> L1 (1024 by default) -> clamp + pairwise product -> 16 -> 32 -> 1,
one set of dense weights per output bucket (8 buckets by piece count).

## Pipeline
1. **Prepare data** (one pass over the database; about 3-4 hours on a laptop): converts the database into
   shuffled binary shards and drops unusable positions (mate scores, checks, best move is a capture, shallow evals).
   ```
   zstd -dc lichess_db_eval.jsonl.zst | python nnue_tools/prep_data.py --out data --shards 64 --workers 6
   ```
2. **Train** (a GPU such as a Colab T4 is recommended for L1 = 1024):
   ```
   python nnue_tools/train.py --data data --device cuda --epochs 4 --max-minutes 360 --out net.bin
   ```
   On a CPU use e.g. `--hidden 512`. The script prints a quantization check (float net vs the integer net the engine runs).
3. **Test the inference code** against the Python reference:
   ```
   g++ -O2 -std=c++17 -Isrc -o nnue_tools/nnue_test nnue_tools/test_main.cpp src/nnue.cpp
   python nnue_tools/test_nnue.py
   ```
4. **Load the net** in the engine: `setoption name EvalFile value path/to/net.bin`.
5. **Measure** with an SPRT: the same exe with and without `EvalFile` (see `scripts/cutechess_match.bat`).

## Notes
- The net file format is defined in `nnue_tools/nnue_format.py` and read by `src/nnue.cpp`.
- Training data and checkpoints are large and are not part of the repository (see `.gitignore`).

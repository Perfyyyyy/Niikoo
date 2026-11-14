# Niikoo

A UCI chess engine written in C++17 with a neural-network evaluation (NNUE) trained on Stockfish evaluations of
positions from the Lichess database. (Self play data will be added in the future)

## Features
- Alpha-beta search with iterative deepening, transposition table, late move reductions, mate distance pruning,
  history and capture-history move ordering, Lazy SMP (multi-threading) and pondering
- NNUE evaluation: 16 king-bucketed inputs, pairwise activation, two small dense layers, 8 output buckets
  (incremental accumulator updates, AVX2)
- Syzygy tablebases (Fathom) and Polyglot opening books.
- UCI protocol; works with cutechess-cli, Arena, Lichess-bot and others

## Strength
> +3356.7 ± 17.1 Elo vs. Akimbo 1.0.0 (3474), SmallBrain 7.0 (3441), Tunguska 2.2 (3362), Laser 1.7 (3338), Mr Bob 1.3.0 (3313) [20+0.2, 250 rounds, 2 games, -repeat (2500 total games)]

## Download
Windows x64 builds (AVX2) are on the [Releases](https://github.com/Perfyyyyy/Niikoo/releases) page.
Put the network file next to the exe and load it with `setoption name EvalFile value path/to/net.bin`.

## Build
Requires GNU Make and a C++17 compiler (MSYS2 UCRT64 on Windows).
```
make -j8                  # builds chess-ai.exe
make bench                # fixed-depth benchmark (node count must not change in refactors)
make bench NET=nets/net.bin
```

## UCI options
| Option | Default | Meaning |
|---|---|---|
| `Hash` | 64 | Transposition table size in MB |
| `Threads` | 1 | Search threads |
| `MoveOverhead` | 50 | Milliseconds kept in reserve for GUI/network lag |
| `EvalFile` | empty | NNUE network file (empty = handcrafted evaluation) |
| `Contempt` | 0 | Centipawns: how bad a draw is for the engine (0 = off) |
| `BookFile` | Perfect2023.bin | Polyglot opening book |
| `Ponder` | false | Think on the opponent's time |
| `SyzygyPath` | syzygy | Folder with Syzygy tablebases |
| `SyzygyProbeDepth` | 1 | Minimum depth for tablebase probes |
| `Clear Hash` | - | Clears the transposition table |

## Training the network
The training pipeline is in [`nnue_tools/`](nnue_tools): `prep_data.py` converts the Lichess evaluation database into
training shards and `train.py` trains the network (CPU or GPU) and writes the engine's net file.
See [`docs/TRAINING.md`](docs/TRAINING.md).

## Project layout
```
src/          engine source
nnue_tools/   NNUE data preparation, training and inference tests
nets/         released networks (small files)
scripts/      match scripts (cutechess-cli)
docs/         design notes, training guide, test results
```

## Credits
- [chess-library](https://github.com/Disservin/chess-library) by disservin (board and move generation), MIT
- [Fathom](https://github.com/jdart1/Fathom) (Syzygy probing), MIT
- [Lichess evaluation database](https://database.lichess.org/) (training data), CC0
- Piece-square tables from the Simplified Evaluation Function (Chess Programming Wiki)

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License
MIT (see `LICENSE`).

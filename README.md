# Niikoo

![CI](https://github.com/Perfyyyyy/Niikoo/actions/workflows/ci.yml/badge.svg)

A UCI chess engine written in C++17 with a neural-network evaluation (NNUE) trained on Stockfish evaluations of
positions from the Lichess database. (Self play data will be added in the future)

## Features
- Alpha-beta search with iterative deepening, transposition table, late move reductions, mate distance pruning,
  history and capture-history move ordering, Lazy SMP (multi-threading) and pondering
- NNUE evaluation: (16 king buckets x 768 piece-square features) -> 768 int16 accumulator per perspective ->
  clipped pairwise product -> dense 16 -> dense 32 -> 1, with 8 output buckets chosen by piece count
  (incremental accumulator updates, AVX2; the dense layer uses AVX-VNNI or AVX512-VNNI automatically when the CPU has it)
- Syzygy tablebases (Fathom) and Polyglot opening books.
- UCI protocol; works with cutechess-cli, Arena, Lichess-bot and others

## Strength
> +3388.4 ± 10.8 Elo vs. Akimbo 1.0.0 (3474), SmallBrain 7.0 (3441), Tunguska 2.2 (3362), Laser 1.7 (3338), Mr Bob 1.3.0 (3313) [20+0.2, 250 rounds, 2 games, -repeat (2500 total games)]

## Download
Windows x64 builds (AVX2) are on the [Releases](https://github.com/Perfyyyyy/Niikoo/releases) page.
They run on any AVX2 CPU; VNNI is detected at run time and used when available, so there is no separate VNNI build.
Put the network file next to the exe and load it with `setoption name EvalFile value path/to/net.bin`.

## Build
Requires GNU Make and a C++17 compiler (MSYS2 UCRT64 on Windows).
```
make -j8                  # builds niikoo.exe
make bench                # fixed-depth benchmark (node count must not change in refactors)
make bench NET=nets/net.bin
```
The SIMD level is chosen at run time (plain C++, AVX2, AVX-VNNI or AVX512-VNNI), so `ARCH=-mavx2` is enough for every
path. The AVX-VNNI path is only compiled with GCC 11+ or clang 12+; with an older compiler it is skipped and the
AVX512-VNNI or AVX2 path is used instead. All paths give identical evaluations.

## UCI options
| Option | Default | Meaning |
|---|---|---|
| `Hash` | 64 | Transposition table size in MB |
| `Threads` | 1 | Search threads |
| `MoveOverhead` | 50 | Milliseconds kept in reserve for GUI/network lag |
| `EvalFile` | empty | NNUE network file (empty = handcrafted evaluation) |
| `BookFile` | Perfect2023.bin | Polyglot opening book |
| `Ponder` | false | Think on the opponent's time |
| `SyzygyPath` | syzygy | Folder with Syzygy tablebases |
| `SyzygyProbeDepth` | 1 | Minimum depth for tablebase probes |
| `Clear Hash` | - | Clears the transposition table |

## Project layout
```
src/          engine source
nets/         released networks (small files)
scripts/      match scripts (cutechess-cli)
docs/         design notes, training guide, test results
```

## Credits
- [chess-library](https://github.com/Disservin/chess-library) by disservin (board and move generation), MIT
- [Fathom](https://github.com/jdart1/Fathom) (Syzygy probing), MIT
- [Lichess evaluation database](https://database.lichess.org/) (training data), CC0
- Piece-square tables from the Simplified Evaluation Function (Chess Programming Wiki)

See [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES.md).

## License
MIT See [`LICENSE`](LICENSE.md).
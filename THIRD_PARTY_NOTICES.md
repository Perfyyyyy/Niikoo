# Third-party material used by this engine

| What | Where | Licence / terms |
|---|---|---|
| chess-library v0.9.9 by disservin (board, move generation, rules) | `chess.hpp` | MIT. https://github.com/Disservin/chess-library |
| Fathom (Syzygy tablebase probing) | `fathom/` | MIT. https://github.com/jdart1/Fathom |
| Piece-square tables and piece values | `tables.hpp` | These match the well-known "Simplified Evaluation Function" published on the Chess Programming Wiki. Widely shared.|
| Training data: Lichess evaluation database (Stockfish evaluations of positions) | not in the repo (downloaded by `prep_data.py`) | CC0 (public domain). https://database.lichess.org |
| Opening book `Perfect2023.bin` | https://www.chessengeria.eu/post/perfect-2023-books | Made by Mr. Sedat Canbaz |

Techniques used here (alpha-beta, transposition table, late move reductions, mate distance pruning, history heuristics,
NNUE with king buckets / output buckets / pairwise activation) are general, widely published ideas. The code for them in this
repo was written for this project; no source files from Stockfish or other engines were copied in.
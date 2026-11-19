# Training the network

The engine evaluates positions with a neural network (NNUE) trained on Stockfish evaluations from the
[Lichess evaluation database](https://database.lichess.org/) (CC0).

## Architecture (v4)
king-bucketed inputs (16 x 768) -> L1 (1024 by default) -> clamp + pairwise product -> 16 -> 32 -> 1,
one set of dense weights per output bucket (8 buckets by piece count).
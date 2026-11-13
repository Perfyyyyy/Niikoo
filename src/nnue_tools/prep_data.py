#!/usr/bin/env python3
"""
Converts the Lichess evaluation database (Stockfish evals, CC0) into compact binary training data.

  zstd -dc lichess_db_eval.jsonl.zst | python3 prep_data.py --out data --workers 6          # ALL positions, 64 shuffled shards
  zstd -dc lichess_db_eval.jsonl.zst | python3 prep_data.py --out small.bin --max-positions 2000000 --stride 4   # one small file

--out - writes the records to stdout, so they can be piped straight into train.py --data - (no files at all).
--out NAME.bin writes a single file; any other name is a folder of --shards files. Every kept position goes to a
random shard, so each shard is a random sample of the whole set (train.py streams them one at a time, so the
data never has to fit in RAM). --workers N parses with N processes (the reader + zstd use about one more core).

Per position it keeps the deepest evaluation's first PV and skips: mate scores, |cp| above --max-cp,
evals shallower than --min-depth, positions where the side to move is in check, and positions whose best
move is a capture (noisy positions make poor static-eval training data).

Record (68 bytes): stm u8 (0 white / 1 black) | n u8 | score i16 (centipawns, WHITE's point of view)
                   | pieces u16[32]  = (colour*6 + piecetype)*64 + square, padded with 0xFFFF
"""
import argparse, itertools, json, os, sys, time
from collections import deque
from multiprocessing import Pool
import numpy as np

REC = np.dtype([("stm", "u1"), ("n", "u1"), ("score", "<i2"), ("pc", "<u2", 32)])
PT = {"p": 0, "n": 1, "b": 2, "r": 3, "q": 4, "k": 5}
VAL = [1, 3, 3, 5, 9, 0]
KN = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
ORTH = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def parse_board(placement):
    board, rank, file = [-1] * 64, 7, 0
    for ch in placement:
        if ch == "/":
            rank -= 1; file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            board[rank * 8 + file] = (0 if ch.isupper() else 1) * 6 + PT[ch.lower()]
            file += 1
    return board


def attacked(board, sq, by):
    """is square `sq` attacked by colour `by`?"""
    f, r = sq & 7, sq >> 3
    pr = r - 1 if by == 0 else r + 1                      # a white pawn attacks upward, so it sits one rank below
    for df in (-1, 1):
        ff = f + df
        if 0 <= ff < 8 and 0 <= pr < 8 and board[pr * 8 + ff] == by * 6:
            return True
    for df, dr in KN:
        ff, rr = f + df, r + dr
        if 0 <= ff < 8 and 0 <= rr < 8 and board[rr * 8 + ff] == by * 6 + 1:
            return True
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            ff, rr = f + df, r + dr
            if (df or dr) and 0 <= ff < 8 and 0 <= rr < 8 and board[rr * 8 + ff] == by * 6 + 5:
                return True
    for dirs, sliders in ((DIAG, (2, 4)), (ORTH, (3, 4))):
        for df, dr in dirs:
            ff, rr = f + df, r + dr
            while 0 <= ff < 8 and 0 <= rr < 8:
                p = board[rr * 8 + ff]
                if p != -1:
                    if p // 6 == by and p % 6 in sliders:
                        return True
                    break
                ff += df; rr += dr
    return False


def sq_of(name):
    return (ord(name[1]) - 49) * 8 + (ord(name[0]) - 97)


def convert(d, a):
    """-> (record fields) or None"""
    evals = d["evals"]
    best = max(evals, key=lambda e: e["depth"])
    if best["depth"] < a.min_depth:
        return None
    pv = best["pvs"][0]
    cp = pv.get("cp")
    if cp is None or abs(cp) > a.max_cp:
        return None
    parts = d["fen"].split()
    board = parse_board(parts[0])
    stm = 0 if parts[1] == "w" else 1
    ep = sq_of(parts[3]) if len(parts) > 3 and parts[3] != "-" else -1
    pieces = [(i, p) for i, p in enumerate(board) if p != -1]
    if len(pieces) > 32:
        return None
    king = [i for i, p in pieces if p == stm * 6 + 5]
    if len(king) != 1 or attacked(board, king[0], stm ^ 1):
        return None                                        # in check (or broken fen)
    line = pv.get("line", "")
    if line:
        mv = line.split()[0]
        to, frm = sq_of(mv[2:4]), sq_of(mv[0:2])
        if board[to] != -1 or (to == ep and board[frm] % 6 == 0):
            return None                                    # best move is a capture
    white_cp = cp if a.cp_pov == "white" else (cp if stm == 0 else -cp)
    return stm, white_cp, pieces, board


def material(board):
    return sum((1 if p // 6 == 0 else -1) * VAL[p % 6] for p in board if p != -1)


def check_pov(a):
    """Sanity check of the cp sign convention on the first positions."""
    agree = {0: [0, 0], 1: [0, 0]}
    for line in sys.stdin.buffer:
        r = convert(json.loads(line), argparse.Namespace(**{**vars(a), "cp_pov": "white", "max_cp": 10**6}))
        if r is None:
            continue
        stm, cp, _, board = r
        m = material(board)
        if abs(m) >= 3 and abs(cp) >= 150:
            agree[stm][0] += (m > 0) == (cp > 0)
            agree[stm][1] += 1
        if sum(v[1] for v in agree.values()) >= a.check:
            break
    for s, name in ((0, "white to move"), (1, "black to move")):
        ag, n = agree[s]
        print(f"{name}: cp sign agrees with material (as WHITE-pov) in {ag}/{n} = {ag / max(n, 1):.0%}")
    print("Both ~high -> data is white-pov (default). Black-to-move near 0% -> rerun with --cp-pov stm.")


def process_chunk(task):
    """worker: list of raw json lines -> numpy array of records"""
    lines, a = task
    buf = np.zeros(len(lines), dtype=REC)
    nb = 0
    for line in lines:
        try:
            r = convert(json.loads(line), a)
        except (ValueError, KeyError, IndexError):
            continue
        if r is None:
            continue
        stm, cp, pieces, _ = r
        rec = buf[nb]
        rec["stm"], rec["n"], rec["score"] = stm, len(pieces), cp
        rec["pc"][:] = 0xFFFF
        for k, (sq, p) in enumerate(pieces):
            rec["pc"][k] = p * 64 + sq
        nb += 1
    return buf[:nb]


def chunks(stream, size, stride):
    chunk, ln = [], 0
    for line in stream:
        ln += 1
        if stride > 1 and ln % stride:
            continue
        chunk.append(line)
        if len(chunk) == size:
            yield chunk, ln
            chunk = []
    if chunk:
        yield chunk, ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="NAME.bin = one file, '-' = write records to stdout (pipe into train.py --data -), otherwise a folder of shards")
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--max-positions", type=int, default=0, help="0 = everything")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth input line")
    ap.add_argument("--min-depth", type=int, default=20)
    ap.add_argument("--max-cp", type=int, default=1500)
    ap.add_argument("--cp-pov", choices=["white", "stm"], default="white", help="whose point of view the db's cp is")
    ap.add_argument("--check", type=int, default=0, help="only run the cp-sign sanity check on N positions")
    ap.add_argument("--chunk", type=int, default=20000, help="input lines per work item")
    a = ap.parse_args()
    if a.check:
        return check_pov(a)

    if a.out == "-":
        files = [sys.stdout.buffer]
    elif a.out.endswith(".bin"):
        files = [open(a.out, "wb")]
    else:
        os.makedirs(a.out, exist_ok=True)
        files = [open(os.path.join(a.out, f"shard_{i:03d}.bin"), "wb") for i in range(a.shards)]
    rng = np.random.default_rng(12345)
    total, lines_read, t0 = 0, 0, time.time()

    def write(arr):
        nonlocal total
        if len(files) == 1:
            files[0].write(arr.tobytes())          # (tofile() fails on pipes)
        else:
            sid = rng.integers(0, len(files), len(arr))
            for k in np.unique(sid):
                arr[sid == k].tofile(files[k])
        total += len(arr)

    def report():
        el = time.time() - t0
        print(f"{lines_read:,} lines read, {total:,} positions kept, {lines_read / max(el, 1):,.0f} lines/s, "
              f"{el / 60:.1f} min", file=sys.stderr, flush=True)

    stream = chunks(sys.stdin.buffer, a.chunk, a.stride)
    nchunks = 0
    if a.workers <= 1:
        for chunk, ln in stream:
            write(process_chunk((chunk, a))); lines_read = ln; nchunks += 1
            if nchunks % 25 == 0: report()
            if a.max_positions and total >= a.max_positions: break
    else:
        with Pool(a.workers) as pool:
            pending = deque()
            for chunk, ln in stream:
                pending.append(pool.apply_async(process_chunk, ((chunk, a),)))
                lines_read = ln; nchunks += 1
                if len(pending) >= a.workers * 3:            # bounded queue: never read far ahead of the workers
                    write(pending.popleft().get())
                    if nchunks % 25 == 0: report()
                if a.max_positions and total >= a.max_positions: break
            while pending:
                write(pending.popleft().get())
    for f in files:
        f.flush() if f is sys.stdout.buffer else f.close()
    report()
    print(f"done: {total:,} positions -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:        # the trainer finished (time budget) and closed the pipe
        os._exit(0)

#!/usr/bin/env python3
"""
Trains the v3 net and exports it for the engine:
  king-bucketed inputs (16 x 768) -> L1 per perspective -> clamp + pairwise product -> 16 -> 32 -> 1,
  with one set of dense weights per output bucket (8 buckets by piece count).
A GPU (Colab) is strongly recommended for L1 = 1024; on a CPU use --hidden 256 or 512.

--data can be a single .bin file, the folder of shards written by prep_data.py (streamed from disk one shard at a
time, so the dataset can be far bigger than RAM), or "-" to read records from stdin (no files at all):

  zstd -dc lichess_db_eval.jsonl.zst | python3 prep_data.py --out - | python3 train.py --data - --max-minutes 540 ...

In stdin mode the data is seen once (one pass), shuffled through an in-RAM buffer (--buffer positions), and the
LR schedule follows --max-minutes and/or --total-positions.

  python3 train.py --data data --hidden 256 --epochs 2 --max-minutes 540 --threads 4 --out net.bin
  python3 train.py --data data --resume checkpoint.pt ...        (continue a run)

The net is exported to --out every --save-minutes, so an interrupted run still leaves a usable net.
Loss: MSE between sigmoid(net output) and sigmoid(Stockfish cp / 400), both from the side to move's view.
The output is cp / 400, so the engine's score = output * 400.
"""
import argparse, copy, glob, math, os, queue, sys, threading, time
import numpy as np
import torch
import torch.nn as nn
from nnue_format import (quantize_and_write, read_net, reference_eval, QA, QW, SCALE, KING_BUCKET, KING_BUCKETS,
                         OUTPUT_BUCKETS, L2, L3)

REC = np.dtype([("stm", "u1"), ("n", "u1"), ("score", "<i2"), ("pc", "<u2", 32)])
PAD = KING_BUCKETS * 768                              # index of the all-zero padding row
KB = np.array(KING_BUCKET, dtype=np.int64)


def bucket_linear(x, W, b, ob):
    """out[i] = x[i] @ W[ob[i]].T + b[ob[i]]  (a different dense layer for each output bucket)"""
    out = x.new_zeros(x.shape[0], W.shape[1])
    for k in range(W.shape[0]):
        idx = (ob == k).nonzero(as_tuple=True)[0]
        if idx.numel():
            out = out.index_copy(0, idx, x[idx] @ W[k].t() + b[k])
    return out


class Net(nn.Module):
    """king-bucketed inputs -> L1 (shared by both perspectives) -> clamp + pairwise product -> dense 16 -> dense 32 -> 1.
    The dense layers have one weight set per output bucket."""
    def __init__(self, l1):
        super().__init__()
        assert l1 % 32 == 0 and l1 <= 1024, "hidden must be a multiple of 32 and at most 1024"
        self.l1 = l1
        OB = OUTPUT_BUCKETS
        self.ft = nn.EmbeddingBag(PAD + 1, l1, mode="sum", padding_idx=PAD)
        self.ft_bias = nn.Parameter(torch.zeros(l1))
        self.l2_w = nn.Parameter(torch.randn(OB, L2, l1) / math.sqrt(l1))
        self.l2_b = nn.Parameter(torch.zeros(OB, L2))
        self.l3_w = nn.Parameter(torch.randn(OB, L3, L2) / math.sqrt(L2))
        self.l3_b = nn.Parameter(torch.zeros(OB, L3))
        self.out_w = nn.Parameter(torch.randn(OB, 1, L3) / math.sqrt(L3))
        self.out_b = nn.Parameter(torch.zeros(OB, 1))
        with torch.no_grad():
            self.ft.weight.normal_(0.0, 0.05)
            self.ft.weight[PAD].zero_()

    def forward(self, us_idx, them_idx, ob):
        half = self.l1 // 2
        us = (self.ft(us_idx) + self.ft_bias).clamp(0.0, 1.0)
        them = (self.ft(them_idx) + self.ft_bias).clamp(0.0, 1.0)
        x = torch.cat([us[:, :half] * us[:, half:], them[:, :half] * them[:, half:]], dim=1)   # pairwise product, side to move first
        h2 = bucket_linear(x, self.l2_w, self.l2_b, ob).clamp(0.0, 1.0)
        h3 = bucket_linear(h2, self.l3_w, self.l3_b, ob).clamp(0.0, 1.0)
        return bucket_linear(h3, self.out_w, self.out_b, ob).squeeze(1)

    def clamp_(self):                                                    # keep everything representable in the net file
        with torch.no_grad():
            self.ft.weight.clamp_(-3.5, 3.5)                             # an accumulator of <= 33 terms can never overflow int16
            self.ft_bias.clamp_(-3.5, 3.5)
            self.l2_w.clamp_(-4.0, 4.0)                                  # int16 with QW = 512 and no int32 overflow in the engine


def make_batch(rec, idx):
    pc = rec["pc"][idx].astype(np.int64)
    stm = rec["stm"][idx].astype(np.int64)
    score = rec["score"][idx].astype(np.float32)
    valid = pc != 0xFFFF
    colour, pt, sq = pc // 384, (pc // 64) % 6, pc % 64
    npieces = valid.sum(1)
    obucket = np.clip((npieces - 2) // 4, 0, OUTPUT_BUCKETS - 1)
    feats = []
    for p in (0, 1):                                                     # perspective: 0 white, 1 black
        s_ = sq if p == 0 else sq ^ 56                                   # squares as seen from p's side
        own_king = valid & (pt == 5) & (colour == p)
        ks = np.take_along_axis(s_, own_king.argmax(1)[:, None], 1)[:, 0]
        mirror = (ks & 7) >= 4                                           # king on files e-h: mirror the board
        ks_m = np.where(mirror, ks ^ 7, ks)
        kb = KB[(ks_m >> 3) * 4 + (ks_m & 7)]
        s_ = np.where(mirror[:, None], s_ ^ 7, s_)
        rel = (colour != p).astype(np.int64)
        f = kb[:, None] * 768 + (rel * 6 + pt) * 64 + s_
        feats.append(np.where(valid, f, PAD))
    white = (stm == 0)[:, None]
    us, them = np.where(white, feats[0], feats[1]), np.where(white, feats[1], feats[0])
    s = np.where(stm == 0, score, -score)                                # side to move's view
    y = 1.0 / (1.0 + np.exp(-s / SCALE))
    return (torch.from_numpy(us), torch.from_numpy(them), torch.from_numpy(obucket.astype(np.int64)),
            torch.from_numpy(y.astype(np.float32)))


def export(net, path):
    with torch.no_grad():
        g = lambda t: t.detach().cpu().numpy().astype(np.float64)
        quantize_and_write(path, g(net.ft.weight)[:PAD], g(net.ft_bias), g(net.l2_w), g(net.l2_b), g(net.l3_w),
                           g(net.l3_b), g(net.out_w).reshape(OUTPUT_BUCKETS, L3), g(net.out_b).reshape(OUTPUT_BUCKETS))
    print(f"exported {path}", flush=True)


def quant_check(net, val_rec, path, n=400):
    """How far is the quantized/integer net (what the engine runs) from the float net that was trained?"""
    n = min(n, len(val_rec))
    ref = read_net(path)
    cpu = copy.deepcopy(net).cpu().eval()
    us, them, ob, y = make_batch(val_rec, np.arange(n))
    with torch.no_grad():
        flt = cpu(us, them, ob).numpy() * SCALE
    diffs = []
    for i in range(n):
        pcs = [(int(p) // 384, (int(p) // 64) % 6, int(p) % 64) for p in val_rec["pc"][i] if p != 0xFFFF]
        diffs.append(abs(float(flt[i]) - reference_eval(ref, pcs, int(val_rec["stm"][i]))))
    print(f"quantization check on {n} validation positions: float net vs engine net, mean |diff| {np.mean(diffs):.2f} cp, "
          f"max {np.max(diffs):.1f} cp (should be a few cp at most)", flush=True)


def evaluate(net, rec, batch, dev=None):
    net.eval(); tot = 0.0; cnt = 0
    with torch.no_grad():
        for i in range(0, len(rec), batch):
            idx = np.arange(i, min(i + batch, len(rec)))
            us, them, ob, y = [t.to(dev) if dev is not None else t for t in make_batch(rec, idx)]
            tot += ((torch.sigmoid(net(us, them, ob)) - y) ** 2).sum().item(); cnt += len(idx)
    net.train()
    return tot / max(cnt, 1)


def list_shards(path):
    files = sorted(glob.glob(os.path.join(path, "*.bin"))) if os.path.isdir(path) else [path]
    assert files, f"no .bin files found in {path}"
    return files, [os.path.getsize(f) // REC.itemsize for f in files]


def batch_producer(files, val_skip, batch, out_q, stop, seed=0):
    """Streams the shards in random order, one at a time; shuffles within each shard. The first `val_skip`
    records of the first shard are reserved for validation and never trained on. Runs forever (epoch after epoch)."""
    rng = np.random.default_rng(seed)
    while not stop.is_set():
        for fi in rng.permutation(len(files)):
            rec = np.fromfile(files[fi], dtype=REC)
            if fi == 0:
                rec = rec[val_skip:]
            perm = rng.permutation(len(rec))
            for b in range(len(rec) // batch):
                if stop.is_set():
                    return
                out_q.put(make_batch(rec, perm[b * batch:(b + 1) * batch]))


def read_records(stream, n):
    data = stream.read(n * REC.itemsize)
    return np.frombuffer(data[:len(data) // REC.itemsize * REC.itemsize], dtype=REC).copy()


def stream_producer(stream, val_n, batch, buf_size, out_q, val_q, stop, seed=0):
    """stdin mode: the first val_n records are held out for validation; the rest go through a shuffle buffer
    (each batch is a random sample of the buffer; the sampled slots are refilled from the stream)."""
    rng = np.random.default_rng(seed)
    val_q.put(read_records(stream, val_n))
    buf = read_records(stream, buf_size)
    ended = len(buf) < buf_size
    while not stop.is_set() and len(buf) >= batch:
        idx = rng.choice(len(buf), size=batch, replace=False, shuffle=False)
        out_q.put(make_batch(buf, idx))
        new = np.empty(0, dtype=REC) if ended else read_records(stream, batch)
        k = len(new)
        if k:
            buf[idx[:k]] = new
        if k < batch:                                   # stream ended: drain what is left in the buffer
            ended = True
            keep = np.ones(len(buf), dtype=bool)
            keep[idx[k:]] = False
            buf = buf[keep]
    out_q.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a .bin file, a folder of shards, or - for records on stdin")
    ap.add_argument("--buffer", type=int, default=4_000_000, help="stdin mode: shuffle buffer size in positions (68 bytes each)")
    ap.add_argument("--total-positions", type=int, default=0, help="stdin mode: expected positions to train on (LR schedule); else use --max-minutes")
    ap.add_argument("--out", default="net.bin")
    ap.add_argument("--hidden", type=int, default=0, help="L1 width (multiple of 32, max 1024). 0 = 1024 on a GPU, 512 on a CPU")
    ap.add_argument("--epochs", type=float, default=1.0, help="passes over the data (can be fractional)")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-end", type=float, default=1e-5)
    ap.add_argument("--val", type=int, default=200_000, help="positions held out for validation")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="torch threads (use the physical core count)")
    ap.add_argument("--resume", default="")
    ap.add_argument("--checkpoint", default="checkpoint.pt", help="where the checkpoint is written (e.g. a Google Drive path)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="auto = GPU if there is one")
    ap.add_argument("--producers", type=int, default=1, help="threads that build batches (shard/file mode only; use 2 on a GPU machine)")
    ap.add_argument("--max-minutes", type=float, default=0, help="time budget: the LR schedule is stretched over it and training stops cleanly")
    ap.add_argument("--save-minutes", type=float, default=15, help="export net.bin + checkpoint this often")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if (a.device == "cuda" or (a.device == "auto" and torch.cuda.is_available())) else "cpu")
    print(f"training on {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""), flush=True)
    stream_mode = a.data == "-"
    producers = 1 if stream_mode else max(1, a.producers)      # stdin can only be read by one thread
    q = queue.Queue(maxsize=6 * producers)
    stop = threading.Event()
    if stream_mode:
        assert a.max_minutes or a.total_positions, "stdin mode needs --max-minutes and/or --total-positions for the LR schedule"
        val_q = queue.Queue()
        threading.Thread(target=stream_producer, args=(sys.stdin.buffer, a.val, a.batch, a.buffer, q, val_q, stop, int(time.time())),
                         daemon=True).start()
        print(f"stdin mode: reading {a.val:,} validation positions, then filling a {a.buffer:,}-position shuffle buffer ...", flush=True)
        val_rec = val_q.get()
        steps_per_epoch = 0
        total_steps = a.total_positions // a.batch
        val_every = 2000
        print(f"{len(val_rec):,} val positions held out; training starts now", flush=True)
    else:
        files, counts = list_shards(a.data)
        total = sum(counts)
        first = np.fromfile(files[0], dtype=REC)
        val_n = min(a.val, len(first) // 2)
        val_rec = first[:val_n]
        del first
        n_train = total - val_n
        assert n_train > a.batch, "dataset too small"
        steps_per_epoch = n_train // a.batch
        total_steps = int(a.epochs * steps_per_epoch)
        print(f"{total:,} positions in {len(files)} file(s) ({n_train:,} train / {val_n:,} val), hidden {a.hidden}, "
              f"{a.threads} threads, {steps_per_epoch:,} steps/epoch, {total_steps:,} steps planned", flush=True)
        for i in range(producers):
            threading.Thread(target=batch_producer, args=(files, val_n, a.batch, q, stop, int(time.time()) + i), daemon=True).start()

    if a.hidden == 0:
        a.hidden = 1024 if dev.type == "cuda" else 512
    print(f"L1 = {a.hidden}", flush=True)
    net = Net(a.hidden).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    step = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
        step = ck.get("step", ck.get("epoch", 0) * steps_per_epoch)
        print(f"resumed at step {step:,}" + (" (stdin mode: the data stream starts again from the beginning)" if stream_mode else ""))

    checked = [False]

    def save(final=False):
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step}, a.checkpoint)
        export(net, a.out)
        if final or not checked[0]:                                 # check the quantized net once early, and at the end
            quant_check(net, val_rec, a.out)
            checked[0] = True

    print_every = max(1, min(200, steps_per_epoch // 4)) if steps_per_epoch else 50
    t0 = last_save = time.time()
    start_step, run, out_of_time = step, 0.0, False
    while not total_steps or step < total_steps:
        elapsed = (time.time() - t0) / 60
        progress = step / total_steps if total_steps else 0.0
        if a.max_minutes:                                                  # whichever runs out first drives the decay
            progress = max(progress, elapsed / a.max_minutes)
            if elapsed >= a.max_minutes:
                out_of_time = True
                break
        batch_data = q.get()
        if batch_data is None:                                             # stdin ended: all data used
            print("end of data stream")
            break
        us, them, ob, y = [t.to(dev, non_blocking=True) for t in batch_data]
        lr = a.lr_end + 0.5 * (a.lr - a.lr_end) * (1 + np.cos(np.pi * min(progress, 1.0)))   # cosine decay
        for g in opt.param_groups:
            g["lr"] = lr
        loss = ((torch.sigmoid(net(us, them, ob)) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        net.clamp_()
        run = run + loss.detach(); step += 1                                # stays on the GPU (no sync per step)
        if step % print_every == 0:
            pps = (step - start_step) * a.batch / max(time.time() - t0, 1e-9)
            left = f", ~{(total_steps - step) * a.batch / max(pps, 1) / 60:.0f} min left at this speed" if total_steps else ""
            ep = f" ({step / steps_per_epoch:.2f} epochs)" if steps_per_epoch else ""
            print(f"step {step:,}{ep} loss {float(run) / print_every:.6f} lr {lr:.2e} {pps:,.0f} pos/s, "
                  f"{elapsed:.0f} min elapsed{left}", flush=True)
            run = 0.0
        if (steps_per_epoch and step % steps_per_epoch == 0) or (not steps_per_epoch and step % val_every == 0):
            print(f"=== step {step:,}: validation loss {evaluate(net, val_rec, a.batch, dev):.6f}", flush=True)
        if time.time() - last_save > a.save_minutes * 60:
            save(); last_save = time.time()
    stop.set()
    print(f"final validation loss {evaluate(net, val_rec, a.batch, dev):.6f}" + (" (time budget reached)" if out_of_time else ""))
    save(final=True)


if __name__ == "__main__":
    main()

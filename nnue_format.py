"""Net file (v3) writer/reader + reference of the engine's inference (numpy only).

Architecture (per position):
  king-bucketed inputs (16 buckets x 768) -> L1 (int16 accumulator per perspective)
  -> clamp to [0,255], pairwise product of the two halves  -> concat(us, them)        (L1 values, 0..254)
  -> dense 16 (int weights, clipped ReLU) -> dense 32 (clipped ReLU) -> dense 1       (one weight set per output bucket)
"""
import struct
import numpy as np

MAGIC = b"CHSNNUE3"
QA, QW, SCALE = 255, 512, 400
KING_BUCKETS, OUTPUT_BUCKETS = 16, 8
L2, L3 = 16, 32
PSCALE = QA * QA / 256.0          # a pairwise product (0..255*255) >> 8 represents 1.0 as 253.996
KING_BUCKET = [0, 1, 2, 3,   4, 5, 6, 7,   8, 8, 9, 9,   8, 8, 9, 9,
               10, 10, 11, 11,   10, 10, 11, 11,   12, 12, 13, 13,   14, 14, 15, 15]
HEADER = "<8sIiiiIIII"            # magic, L1, qa, qw, scale, kingBuckets, outputBuckets, l2, l3


def write_net(path, ft_w, ft_b, l2_w, l2_b, l3_w, l3_b, out_w, out_b):
    """ft_w int16 [16*768, L1], ft_b int16 [L1], l2_w int16 [8, 16, L1], l2_b f32 [8,16],
       l3_w f32 [8, 32, 16], l3_b f32 [8, 32], out_w f32 [8, 32], out_b f32 [8]."""
    L1 = ft_w.shape[1]
    assert L1 % 2 == 0 and ft_w.shape[0] == KING_BUCKETS * 768 and ft_b.shape == (L1,)
    arrs = [(ft_w, "<i2", (KING_BUCKETS * 768, L1)), (ft_b, "<i2", (L1,)),
            (l2_w, "<i2", (OUTPUT_BUCKETS, L2, L1)), (l2_b, "<f4", (OUTPUT_BUCKETS, L2)),
            (l3_w, "<f4", (OUTPUT_BUCKETS, L3, L2)), (l3_b, "<f4", (OUTPUT_BUCKETS, L3)),
            (out_w, "<f4", (OUTPUT_BUCKETS, L3)), (out_b, "<f4", (OUTPUT_BUCKETS,))]
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER, MAGIC, L1, QA, QW, SCALE, KING_BUCKETS, OUTPUT_BUCKETS, L2, L3))
        for a, dt, shape in arrs:
            a = np.ascontiguousarray(a, dtype=dt)
            assert a.shape == shape, (a.shape, shape)
            f.write(a.tobytes())


def read_net(path):
    d = open(path, "rb").read()
    hs = struct.calcsize(HEADER)
    magic, L1, qa, qw, scale, kb, ob, l2, l3 = struct.unpack(HEADER, d[:hs])
    assert magic == MAGIC and (kb, ob, l2, l3) == (KING_BUCKETS, OUTPUT_BUCKETS, L2, L3), "not a v3 net"
    o = hs
    def take(dt, shape):
        nonlocal o
        n = int(np.prod(shape))
        a = np.frombuffer(d, dt, n, o).reshape(shape)
        o += n * np.dtype(dt).itemsize
        return a
    net = dict(L1=L1, qa=qa, qw=qw, scale=scale,
               ft_w=take("<i2", (kb * 768, L1)), ft_b=take("<i2", (L1,)),
               l2_w=take("<i2", (ob, l2, L1)), l2_b=take("<f4", (ob, l2)),
               l3_w=take("<f4", (ob, l3, l2)), l3_b=take("<f4", (ob, l3)),
               out_w=take("<f4", (ob, l3)), out_b=take("<f4", (ob,)))
    assert o == len(d), "net file has the wrong size"
    return net


def king_key(perspective, ksq):
    s = ksq if perspective == 0 else ksq ^ 56
    mirror = 1 if (s & 7) >= 4 else 0
    if mirror:
        s ^= 7
    return (KING_BUCKET[(s >> 3) * 4 + (s & 7)] << 1) | mirror


def feature_index(perspective, key, colour, pt, sq):
    s = sq if perspective == 0 else sq ^ 56
    if key & 1:
        s ^= 7
    return (key >> 1) * 768 + ((1 if colour != perspective else 0) * 6 + pt) * 64 + s


def output_bucket(n):
    return max(0, min((n - 2) // 4, OUTPUT_BUCKETS - 1))


def reference_eval(net, pieces, stm):
    """pieces: list of (colour, pt, sq). Integer arithmetic up to and including the first dense layer (bit-exact with
    the engine); the small float layers are evaluated in float64 (the engine uses float32: expect +-1 cp)."""
    L1, qa, qw, scale = net["L1"], net["qa"], net["qw"], net["scale"]
    king = {c: sq for (c, pt, sq) in pieces if pt == 5}
    acc = []
    for p in range(2):
        key = king_key(p, king[p])
        a = net["ft_b"].astype(np.int64).copy()
        for (c, pt, sq) in pieces:
            a += net["ft_w"][feature_index(p, key, c, pt, sq)].astype(np.int64)
        acc.append(a.astype(np.int16).astype(np.int64))              # int16 wrap, as in the engine
    half = L1 // 2
    parts = []
    for side in (stm, stm ^ 1):
        c = np.clip(acc[side], 0, qa)
        parts.append((c[:half] * c[half:] + 128) >> 8)               # pairwise product, 0..254
    x = np.concatenate(parts)
    b = output_bucket(len(pieces))
    z = net["l2_w"][b].astype(np.int64) @ x                          # int32 in the engine
    h2 = np.clip(z / (PSCALE * qw) + net["l2_b"][b].astype(np.float64), 0.0, 1.0)
    h3 = np.clip(net["l3_w"][b].astype(np.float64) @ h2 + net["l3_b"][b], 0.0, 1.0)
    o = float(net["out_w"][b].astype(np.float64) @ h3 + net["out_b"][b])
    return int(o * scale)                                            # truncates toward zero


# ---------- float (training) side: the same network in float64, and the quantizer ----------
def quantize_and_write(path, ft_w, ft_b, l2_w, l2_b, l3_w, l3_b, out_w, out_b):
    """Float parameters (as trained) -> quantized net file.
       ft_w [16*768, L1], ft_b [L1], l2_w [8, 16, L1], l2_b [8, 16], l3_w [8, 32, 16], l3_b [8, 32], out_w [8, 32], out_b [8]"""
    def q(a, scale):
        return np.clip(np.round(np.asarray(a, dtype=np.float64) * scale), -32768, 32767).astype(np.int16)
    write_net(path, q(ft_w, QA), q(ft_b, QA), q(l2_w, QW), np.asarray(l2_b, np.float32), np.asarray(l3_w, np.float32),
              np.asarray(l3_b, np.float32), np.asarray(out_w, np.float32), np.asarray(out_b, np.float32))


def float_forward(params, pieces, stm):
    """The training network evaluated in float64 on one position (what train.py's Net computes). Returns centipawns (float)."""
    ft_w, ft_b, l2_w, l2_b, l3_w, l3_b, out_w, out_b = params
    L1 = ft_b.shape[0]
    king = {c: sq for (c, pt, sq) in pieces if pt == 5}
    acc = []
    for p in range(2):
        key = king_key(p, king[p])
        a = ft_b.astype(np.float64).copy()
        for (c, pt, sq) in pieces:
            a += ft_w[feature_index(p, key, c, pt, sq)]
        acc.append(np.clip(a, 0.0, 1.0))
    half = L1 // 2
    x = np.concatenate([acc[s][:half] * acc[s][half:] for s in (stm, stm ^ 1)])
    b = output_bucket(len(pieces))
    h2 = np.clip(l2_w[b] @ x + l2_b[b], 0.0, 1.0)
    h3 = np.clip(l3_w[b] @ h2 + l3_b[b], 0.0, 1.0)
    return float(out_w[b] @ h3 + out_b[b]) * SCALE

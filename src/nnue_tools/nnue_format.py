"""Net file (v2) writer + bit-exact integer reference of the engine's inference (numpy only)."""
import struct
import numpy as np

MAGIC = b"CHSNNUE2"
QA, QB, SCALE = 255, 64, 400
KING_BUCKETS, OUTPUT_BUCKETS = 8, 8
KING_BUCKET = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 5, 5,
               6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7, 6, 6, 7, 7]


def write_net(path, ft_w, ft_b, out_w, out_b, qa=QA, qb=QB, scale=SCALE):
    """ft_w int16 [8*768, H], ft_b int16 [H], out_w int16 [8, 2H], out_b int32 [8]."""
    ft_w = np.ascontiguousarray(ft_w, dtype="<i2")
    ft_b = np.ascontiguousarray(ft_b, dtype="<i2")
    out_w = np.ascontiguousarray(out_w, dtype="<i2")
    out_b = np.ascontiguousarray(out_b, dtype="<i4")
    H = ft_w.shape[1]
    assert ft_w.shape[0] == KING_BUCKETS * 768 and ft_b.shape == (H,)
    assert out_w.shape == (OUTPUT_BUCKETS, 2 * H) and out_b.shape == (OUTPUT_BUCKETS,)
    with open(path, "wb") as f:
        f.write(struct.pack("<8sIiiiII", MAGIC, H, qa, qb, scale, KING_BUCKETS, OUTPUT_BUCKETS))
        f.write(ft_w.tobytes()); f.write(ft_b.tobytes()); f.write(out_w.tobytes()); f.write(out_b.tobytes())


def read_net(path):
    d = open(path, "rb").read()
    magic, H, qa, qb, scale, kb, ob = struct.unpack("<8sIiiiII", d[:32])
    assert magic == MAGIC and kb == KING_BUCKETS and ob == OUTPUT_BUCKETS, "not a v2 net"
    o = 32
    ft_w = np.frombuffer(d, "<i2", kb * 768 * H, o).reshape(kb * 768, H); o += kb * 768 * H * 2
    ft_b = np.frombuffer(d, "<i2", H, o); o += H * 2
    out_w = np.frombuffer(d, "<i2", ob * 2 * H, o).reshape(ob, 2 * H); o += ob * 2 * H * 2
    out_b = np.frombuffer(d, "<i4", ob, o)
    assert o + ob * 4 == len(d)
    return ft_w, ft_b, out_w, out_b, qa, qb, scale


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


def _tdiv(a, b):  # division truncating toward zero, like C++
    return -((-a) // b) if a < 0 else a // b


def reference_eval(net, pieces, stm):
    """net = read_net(...) tuple; pieces: list of (colour, pt, sq). Returns centipawns from the side to move's view."""
    ft_w, ft_b, out_w, out_b, qa, qb, scale = net
    H = ft_b.shape[0]
    king = {c: sq for (c, pt, sq) in pieces if pt == 5}
    acc = []
    for p in range(2):
        key = king_key(p, king[p])
        a = ft_b.astype(np.int64).copy()
        for (c, pt, sq) in pieces:
            a += ft_w[feature_index(p, key, c, pt, sq)].astype(np.int64)
        acc.append(a.astype(np.int16).astype(np.int64))      # int16 wrap, as in the engine
    us, them = acc[stm], acc[stm ^ 1]
    cu, ct = np.clip(us, 0, qa), np.clip(them, 0, qa)
    b = output_bucket(len(pieces))
    w = out_w[b].astype(np.int64)
    total = int(np.sum(cu * cu * w[:H]) + np.sum(ct * ct * w[H:]))
    total = _tdiv(total, qa) + int(out_b[b])
    return _tdiv(total * scale, qa * qb)

"""Checks the C++ inference (v3) against the numpy reference (exact up to the small float layers: +-1 cp)."""
import subprocess, sys, os, random, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnue_format import write_net, read_net, reference_eval, KING_BUCKETS, OUTPUT_BUCKETS, L2, L3

here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.gettempdir()
exe = os.path.join(here, "nnue_test.exe" if os.name == "nt" else "nnue_test")
if not os.path.exists(exe):
    sys.exit("nnue_test is missing - compile it first (see the g++ command in the README)")
rng = np.random.default_rng(1)
for L1, wscale in ((32, 1.0), (256, 1.0), (256, 0.2), (1024, 0.5)):
    ft_w = rng.integers(-40, 41, size=(KING_BUCKETS * 768, L1), dtype=np.int16)
    ft_b = rng.integers(-30, 120, size=L1, dtype=np.int16)
    l2_w = rng.integers(-300, 301, size=(OUTPUT_BUCKETS, L2, L1), dtype=np.int16)
    l2_b = (rng.normal(0, 0.5, size=(OUTPUT_BUCKETS, L2)) * wscale).astype(np.float32)
    l3_w = (rng.normal(0, 0.4, size=(OUTPUT_BUCKETS, L3, L2)) * wscale).astype(np.float32)
    l3_b = (rng.normal(0.3, 0.3, size=(OUTPUT_BUCKETS, L3))).astype(np.float32)
    out_w = (rng.normal(0, 0.5, size=(OUTPUT_BUCKETS, L3))).astype(np.float32)
    out_b = (rng.normal(0, 0.3, size=OUTPUT_BUCKETS)).astype(np.float32)
    net = os.path.join(tmp, f"test_net_{L1}.bin")
    write_net(net, ft_w, ft_b, l2_w, l2_b, l3_w, l3_b, out_w, out_b)
    ref_net = read_net(net)
    rows, pos = [], []
    for _ in range(300):
        n = random.randint(2, 32)
        sqs = random.sample(range(64), n)
        pcs = [(0, 5, sqs[0]), (1, 5, sqs[1])] + [(random.randint(0, 1), random.randint(0, 4), s) for s in sqs[2:]]
        stm = random.randint(0, 1)
        pos.append((pcs, stm))
        rows.append(f"{stm} {n} " + " ".join(f"{c} {p} {s}" for c, p, s in pcs))
    posfile = os.path.join(tmp, "test_pos.txt")
    open(posfile, "w").write("\n".join(rows) + "\n")
    out = subprocess.run([exe, net, posfile], capture_output=True, text=True, check=True).stdout.split()
    got = [int(x) for x in out]
    exp = [reference_eval(ref_net, p, s) for p, s in pos]
    diff = [abs(g - e) for g, e in zip(got, exp)]
    nonzero = sum(1 for e in exp if e != 0)
    print(f"L1={L1} wscale={wscale}: {len(exp)} positions, max |diff| = {max(diff)} cp, "
          f"{sum(d > 0 for d in diff)} differ by 1, non-zero outputs {nonzero}, range {min(exp)}..{max(exp)}")
    assert len(got) == len(exp) and max(diff) <= 1 and nonzero > 50
print("OK")

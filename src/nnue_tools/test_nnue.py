"""Checks the C++ inference (v2: king buckets + output buckets) against the numpy reference."""
import subprocess, sys, os, random, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnue_format import write_net, read_net, reference_eval

here = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.gettempdir()
exe = os.path.join(here, "nnue_test.exe" if os.name == "nt" else "nnue_test")
if not os.path.exists(exe):
    sys.exit("nnue_test is missing - compile it first (see the g++ command in the README)")
rng = np.random.default_rng(1)
for H in (32, 256):
    ft_w = rng.integers(-40, 41, size=(8 * 768, H), dtype=np.int16)
    ft_b = rng.integers(-30, 120, size=H, dtype=np.int16)
    out_w = rng.integers(-120, 121, size=(8, 2 * H), dtype=np.int16)
    out_b = rng.integers(-300000, 300000, size=8, dtype=np.int32)
    net = os.path.join(tmp, f"test_net_{H}.bin")
    write_net(net, ft_w, ft_b, out_w, out_b)
    ref_net = read_net(net)
    rows, pos = [], []
    for _ in range(400):
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
    bad = sum(g != e for g, e in zip(got, exp))
    print(f"H={H}: {len(exp)} positions, mismatches={bad}, score range {min(exp)}..{max(exp)}")
    assert bad == 0 and len(got) == len(exp)
print("OK")

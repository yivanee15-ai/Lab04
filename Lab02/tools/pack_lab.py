"""Run on YOUR machine: shrink train_lab.npy / val_lab.npy into small uploadable chunks.

Lab (float32) was derived from uint8 RGB, so converting back to RGB is lossless
(round-trip error < 1e-5).  uint8 is 4x smaller, and we split + compress it.

    python tools/pack_lab.py dataset/train_lab.npy out_dir [parts]
    python tools/pack_lab.py dataset/val_lab.npy   out_dir 1
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation as E

src, out = Path(sys.argv[1]), Path(sys.argv[2])
parts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
out.mkdir(parents=True, exist_ok=True)
lab = np.load(src)
rgb = E.lab_to_rgb(lab)
assert np.abs(E.rgb_to_lab(rgb) - lab).max() < 1e-3, "not lossless for this file"
for i, chunk in enumerate(np.array_split(rgb, parts)):
    f = out / f"{src.stem}_part{i}.npz"
    np.savez_compressed(f, rgb=chunk)
    print(f, f"{f.stat().st_size / 1e6:.1f} MB")

"""Rebuild <name>.npy (float32 Lab) from the chunks made by pack_lab.py.

    python tools/unpack_lab.py chunks_dir train_lab dataset/train_lab.npy
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluation as E

d, stem, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
files = sorted(d.glob(f"{stem}_part*.npz"), key=lambda p: int(p.stem.split("part")[-1]))
rgb = np.concatenate([np.load(f)["rgb"] for f in files])
lab = E.rgb_to_lab(rgb).astype(np.float32)
dst.parent.mkdir(parents=True, exist_ok=True)
np.save(dst, lab)
print(dst, lab.shape, lab.dtype)

"""CPU-friendly script version of the notebook's training cell (same model/loss/optimizer)."""
import sys; sys.path.insert(0, ".")
import sys, time, random
import numpy as np, torch, torch.nn as nn
from pathlib import Path
import evaluation as E
from model.cnn import ColorizationCNN

EPOCHS = int(sys.argv[1]); BS = 128; LR = 1e-3
torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.set_num_threads(4)
tr = np.load("dataset/train_lab.npy"); va = np.load("dataset/val_lab.npy")
L, ab = map(torch.from_numpy, E.normalize_lab(tr))
model = ColorizationCNN()
opt = torch.optim.Adam(model.parameters(), lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
crit = nn.L1Loss(); best = -1; t0 = time.time()
Path("model").mkdir(exist_ok=True)
for ep in range(1, EPOCHS + 1):
    model.train(); perm = torch.randperm(len(L)); run = seen = 0
    for i in range(0, len(L) - BS + 1, BS):
        idx = perm[i:i + BS]; x, y = L[idx], ab[idx]
        flip = torch.rand(BS) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x); y = torch.where(flip[:, None, None, None], y.flip(3), y)
        opt.zero_grad(set_to_none=True); loss = crit(model(x), y); loss.backward(); opt.step()
        run += loss.item() * BS; seen += BS
    sched.step()
    v = E.evaluate_model(model, va)
    flag = ""
    if v["psnr"] > best:
        best = v["psnr"]; flag = "  <- best"
        torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_psnr": v["psnr"]}, "model/ckpt_cnn.pth")
    print(f"epoch {ep:>3}/{EPOCHS} loss {run/seen:.4f} val PSNR {v['psnr']:.3f} dB chroma {v['pred_chroma']:.2f} ({time.time()-t0:.0f}s){flag}", flush=True)

"""
Lab02 official evaluation utilities.  *** DO NOT MODIFY THIS FILE ***

Your grade is computed with this exact file.  If you change it, your reported
numbers will not match the TA's and your submission is re-scored with the
original version.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

__all__ = [
    "L_SCALE", "AB_SCALE", "MAX_PIXEL_VALUE",
    "rgb_to_lab", "lab_to_rgb",
    "normalize_lab", "denormalize_ab", "reconstruct_rgb",
    "psnr", "psnr_from_ab", "mean_chroma", "colorfulness",
    "evaluate_arrays", "evaluate_model",
]

L_SCALE = 100.0
AB_SCALE = 110.0
MAX_PIXEL_VALUE = 255.0  # PSNR is computed on uint8 RGB

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_XYZ2RGB = np.linalg.inv(_M_RGB2XYZ)
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883])
_EPS = 216 / 24389
_KAPPA = 24389 / 27


def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """uint8 RGB (..., 3) -> float64 Lab (..., 3).  L*: 0~100, a*/b*: about -128~127."""
    c = np.asarray(rgb_u8, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE_D65
    f = np.where(xyz > _EPS, np.cbrt(xyz), (_KAPPA * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """float Lab (..., 3) -> uint8 RGB (..., 3).  Out-of-gamut colours are clipped."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > _EPS, f ** 3, (116 * f - 16) / _KAPPA) * _WHITE_D65
    lin = np.clip(xyz @ _M_XYZ2RGB.T, 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


# --- normalisation helpers ---------------------------------------------------
def normalize_lab(lab: np.ndarray):
    """Lab (N,H,W,3) -> (L_norm (N,1,H,W), ab_norm (N,2,H,W)) ready for the model."""
    lab = np.asarray(lab, dtype=np.float32)
    L = lab[..., :1].transpose(0, 3, 1, 2) / L_SCALE
    ab = lab[..., 1:].transpose(0, 3, 1, 2) / AB_SCALE
    return np.ascontiguousarray(L), np.ascontiguousarray(ab)


def denormalize_ab(ab_norm: np.ndarray) -> np.ndarray:
    """Normalised ab (N,2,H,W) or (N,H,W,2) -> real a*b* units, channel-last (N,H,W,2)."""
    ab = np.asarray(ab_norm, dtype=np.float32)
    if ab.ndim != 4:
        raise ValueError(f"expected a 4-D array, got shape {ab.shape}")
    if ab.shape[1] == 2 and ab.shape[-1] != 2:      # channel-first
        ab = ab.transpose(0, 2, 3, 1)
    elif ab.shape[-1] != 2:
        raise ValueError(f"cannot find the 2-channel axis in shape {ab.shape}")
    return ab * AB_SCALE


def reconstruct_rgb(L: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Ground-truth L* (N,H,W) or (N,H,W,1) + predicted a*b* (N,H,W,2), both in real
    Lab units, -> uint8 RGB (N,H,W,3)."""
    L = np.asarray(L, dtype=np.float32)
    if L.ndim == 3:
        L = L[..., None]
    return lab_to_rgb(np.concatenate([L, np.asarray(ab, dtype=np.float32)], axis=-1))


# --- metrics -----------------------------------------------------------------
def psnr(pred_rgb: np.ndarray, gt_rgb: np.ndarray, per_image: bool = False):
    """Mean per-image PSNR (dB) between two uint8 RGB batches, MAX = 255."""
    p = np.asarray(pred_rgb, dtype=np.float64)
    g = np.asarray(gt_rgb, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {g.shape}")
    mse = ((p - g) ** 2).mean(axis=tuple(range(1, p.ndim)))
    mse = np.maximum(mse, 1e-10)                      # guard against a perfect match
    values = 10 * np.log10(MAX_PIXEL_VALUE ** 2 / mse)
    return values if per_image else float(values.mean())


def psnr_from_ab(pred_ab: np.ndarray, gt_lab: np.ndarray,
                 normalized: bool = True, per_image: bool = False):
    """The official metric, straight from a prediction and the ground-truth Lab array.

    pred_ab  : (N,2,H,W) or (N,H,W,2); normalised by AB_SCALE unless normalized=False
    gt_lab   : (N,H,W,3) float, real Lab units
    """
    ab = denormalize_ab(pred_ab) if normalized else _as_channel_last_ab(pred_ab)
    gt_lab = np.asarray(gt_lab, dtype=np.float32)
    if ab.shape[:3] != gt_lab.shape[:3]:
        raise ValueError(f"prediction {ab.shape} does not match ground truth {gt_lab.shape}")
    pred_rgb = reconstruct_rgb(gt_lab[..., 0], ab)
    gt_rgb = lab_to_rgb(gt_lab)
    return psnr(pred_rgb, gt_rgb, per_image=per_image)


def _as_channel_last_ab(ab: np.ndarray) -> np.ndarray:
    ab = np.asarray(ab, dtype=np.float32)
    if ab.ndim == 4 and ab.shape[1] == 2 and ab.shape[-1] != 2:
        ab = ab.transpose(0, 2, 3, 1)
    return ab


def mean_chroma(ab: np.ndarray, normalized: bool = False) -> float:
    """Mean sqrt(a*^2 + b*^2).  A grey image scores 0; report it next to PSNR so a
    washed-out prediction is visible.  CIFAR-10 ground truth sits near 15.2."""
    x = denormalize_ab(ab) if normalized else _as_channel_last_ab(ab)
    return float(np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2).mean())


def colorfulness(rgb_u8: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness, averaged over the batch."""
    x = np.asarray(rgb_u8, dtype=np.float64)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    ax = tuple(range(1, rg.ndim))
    std = np.sqrt(rg.std(axis=ax) ** 2 + yb.std(axis=ax) ** 2)
    mean = np.sqrt(rg.mean(axis=ax) ** 2 + yb.mean(axis=ax) ** 2)
    return float((std + 0.3 * mean).mean())


def evaluate_arrays(pred_ab: np.ndarray, gt_lab: np.ndarray,
                    normalized: bool = True) -> dict:
    """Full report: the official PSNR plus the colour-saturation diagnostics."""
    ab = denormalize_ab(pred_ab) if normalized else _as_channel_last_ab(pred_ab)
    gt_lab = np.asarray(gt_lab, dtype=np.float32)
    pred_rgb = reconstruct_rgb(gt_lab[..., 0], ab)
    gt_rgb = lab_to_rgb(gt_lab)
    return {
        "psnr": psnr(pred_rgb, gt_rgb),
        "pred_chroma": mean_chroma(ab),
        "gt_chroma": mean_chroma(gt_lab[..., 1:]),
        "pred_colorfulness": colorfulness(pred_rgb),
        "gt_colorfulness": colorfulness(gt_rgb),
        "num_images": int(len(gt_lab)),
    }


# --- torch helper ------------------------------------------------------------
def evaluate_model(model, lab: np.ndarray, device="cpu", batch_size: int = 512) -> dict:
    """Run a torch model over a Lab array and return the evaluate_arrays report.

    model : nn.Module mapping (B,1,32,32) normalised L -> (B,2,32,32) normalised ab
    lab   : (N,32,32,3) float Lab array (e.g. the val or test split)
    """
    import torch

    L, _ = normalize_lab(lab)
    model = model.to(device).eval()
    out = np.empty((len(L), 2, L.shape[2], L.shape[3]), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(L), batch_size):
            x = torch.from_numpy(L[i:i + batch_size]).to(device)
            out[i:i + batch_size] = model(x).float().cpu().numpy()
    return evaluate_arrays(out, lab, normalized=True)


# --- command line ------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Lab02 official scorer")
    p.add_argument("--pred", type=Path, required=True,
                   help="predicted ab, .npy of shape (N,2,32,32) or (N,32,32,2)")
    p.add_argument("--gt", type=Path, required=True,
                   help="ground-truth Lab, .npy of shape (N,32,32,3)")
    p.add_argument("--raw-ab", action="store_true",
                   help="the prediction is already in real a*b* units (not divided by 110)")
    args = p.parse_args()

    pred = np.load(args.pred)
    gt = np.load(args.gt)
    if len(pred) != len(gt):
        raise SystemExit(f"error: {len(pred)} predictions for {len(gt)} images")

    r = evaluate_arrays(pred, gt, normalized=not args.raw_ab)
    print(f"images            : {r['num_images']}")
    print(f"PSNR (official)   : {r['psnr']:.3f} dB")
    print(f"mean chroma       : {r['pred_chroma']:.2f}  (ground truth {r['gt_chroma']:.2f})")
    print(f"colourfulness     : {r['pred_colorfulness']:.2f}  (ground truth {r['gt_colorfulness']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

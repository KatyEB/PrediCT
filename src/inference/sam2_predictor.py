"""
SAM2-based pericardium segmentation for COCA CT slices.

Strategy: for each axial slice where the heart mask is present, prompt SAM2
with a bounding box around the heart. SAM2 predicts a mask for the region
enclosed by the pericardial sac (heart + epicardial fat). Slices without
a heart mask are skipped.

The output is a (H, W, Z) binary mask of the pericardial region (everything
inside the pericardial sac), saved as a NIfTI alongside the heart mask.

  pericardial_mask & fat_hu_range  → epicardial fat
  cardiac_roi & fat_hu_range & ~pericardial_mask  → paracardial fat
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _get_box(mask_2d: np.ndarray, pad: int = 8) -> Optional[np.ndarray]:
    """Return SAM2 box [x1,y1,x2,y2] (col-row coords) from a 2D binary mask."""
    rows, cols = np.where(mask_2d > 0)
    if len(rows) == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h, w = mask_2d.shape
    return np.array([
        max(0, c0 - pad),
        max(0, r0 - pad),
        min(w - 1, c1 + pad),
        min(h - 1, r1 + pad),
    ], dtype=np.float32)


def ct_slice_to_rgb(
    ct_slice: np.ndarray,
    lo: float = -160.0,
    hi: float = 240.0,
) -> np.ndarray:
    """Convert HU slice to uint8 RGB (H, W, 3) for SAM2 input."""
    scaled = np.clip((ct_slice.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    uint8 = (scaled * 255).astype(np.uint8)
    return np.stack([uint8, uint8, uint8], axis=-1)


def predict_pericardium_volume(
    ct_crop: np.ndarray,
    heart_mask_crop: np.ndarray,
    predictor,
    box_pad: int = 8,
    min_heart_vox: int = 200,
    win_lo: float = -160.0,
    win_hi: float = 240.0,
) -> np.ndarray:
    """
    Run SAM2 slice-by-slice to predict the pericardial region mask.

    Args:
        ct_crop: (H, W, Z) float32 HU array (cardiac ROI)
        heart_mask_crop: (H, W, Z) uint8 binary heart mask
        predictor: SAM2ImagePredictor instance
        box_pad: pixels to pad around the heart mask bounding box
        min_heart_vox: minimum heart voxels in a slice to attempt prediction
        win_lo, win_hi: HU window for the SAM2 input image

    Returns:
        peri_mask: (H, W, Z) uint8 binary — 1 inside pericardial sac
    """
    H, W, Z = ct_crop.shape
    peri_mask = np.zeros((H, W, Z), dtype=np.uint8)

    for z in range(Z):
        heart_slice = heart_mask_crop[:, :, z]
        if heart_slice.sum() < min_heart_vox:
            continue

        box = _get_box(heart_slice, pad=box_pad)
        if box is None:
            continue

        img_rgb = ct_slice_to_rgb(ct_crop[:, :, z], lo=win_lo, hi=win_hi)

        with torch.inference_mode():
            predictor.set_image(img_rgb)
            masks, scores, _ = predictor.predict(
                box=box[None],          # SAM2 expects (1, 4) for box input
                multimask_output=True,
            )

        # Pick the mask with the highest score
        best = int(np.argmax(scores))
        peri_mask[:, :, z] = masks[best].astype(np.uint8)

    return peri_mask


def load_sam2_predictor(model_id: str = "facebook/sam2.1-hiera-base-plus", device: str = "mps"):
    """Load SAM2ImagePredictor from HuggingFace."""
    from sam2.build_sam import build_sam2_hf
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2_hf(model_id, device=device)
    return SAM2ImagePredictor(model)

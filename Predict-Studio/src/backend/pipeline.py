"""
pipeline.py — Pure imaging operations for CAC prediction.

Provides the core numerical routines: loading, resampling, cropping, and
running the PyTorch inference. Isolated here so orchestration and HTTP concerns
cannot bleed into the math.

Does NOT: handle file paths, read manifests, or calculate Agatston scores.
Called by: run.py.

Usage:
    image = load(folder)
    prob  = predict(x, weights_path, arch, patch, overlap, activation)
"""
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

# Set device automatically. Use CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(patient_folder: str | Path) -> sitk.Image:
    """Load DICOM series from folder into a SimpleITK Image."""
    print(f"Loading {patient_folder}...")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(patient_folder))
    if not dicom_names:
        raise ValueError(f"No DICOM files found in {patient_folder}")
    reader.SetFileNames(dicom_names)
    return reader.Execute()

def resample(image: sitk.Image, target_spacing_mm: tuple[float, float, float]) -> sitk.Image:
    """Resample volume to target spacing (sx, sy, sz) in mm."""
    print(f"Resampling to {target_spacing_mm}...")
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(orig_size, orig_spacing, target_spacing_mm)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_mm)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def crop_heart(image: sitk.Image, margin_mm: int, fast: bool = False) -> sitk.Image:
    """Crop the volume around the heart using TotalSegmentator."""
    print(f"Locating heart (fast={fast}) and cropping...")
    try:
        from totalsegmentator.python_api import totalsegmentator
        import tempfile
    except ImportError:
        raise RuntimeError(
            "TotalSegmentator not installed but crop=True. Models were trained on "
            "heart-cropped volumes; uncropped inference produces invalid scores. "
            "Install it, or pass crop=False for plumbing tests only."
        )
        
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "ct.nii.gz"
        tmp_out_dir = Path(tmpdir) / "masks"
        tmp_out_dir.mkdir()
        
        sitk.WriteImage(image, str(tmp_in))
        
        totalsegmentator(str(tmp_in), str(tmp_out_dir), roi_subset=["heart"], fast=fast)
        
        mask_img = sitk.ReadImage(str(tmp_out_dir / "heart.nii.gz"))
        mask_array = sitk.GetArrayFromImage(mask_img)

    if not mask_array.any():
        raise ValueError("Heart mask is empty. TotalSegmentator failed to find the heart.")

    z_idx, y_idx, x_idx = np.nonzero(mask_array)
    spacing = image.GetSpacing()
    
    z_min, z_max = z_idx.min(), z_idx.max()
    y_min, y_max = y_idx.min(), y_idx.max()
    x_min, x_max = x_idx.min(), x_idx.max()
    
    z_margin = int(round(margin_mm / spacing[2]))
    y_margin = int(round(margin_mm / spacing[1]))
    x_margin = int(round(margin_mm / spacing[0]))
    
    z_min, z_max = max(0, z_min - z_margin), min(mask_array.shape[0], z_max + z_margin + 1)
    y_min, y_max = max(0, y_min - y_margin), min(mask_array.shape[1], y_max + y_margin + 1)
    x_min, x_max = max(0, x_min - x_margin), min(mask_array.shape[2], x_max + x_margin + 1)
    
    return image[x_min:x_max, y_min:y_max, z_min:z_max]

def normalize(array: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    """Clip and scale array to [0, 1]. Array shape is arbitrary."""
    print(f"Normalizing HU {hu_window} -> [0, 1]...")
    clipped = np.clip(array, hu_window[0], hu_window[1])
    return (clipped - hu_window[0]) / (hu_window[1] - hu_window[0])

def predict(x: np.ndarray, weights_path: Path, arch: dict, patch: tuple[int, int, int], overlap: float, activation: str) -> np.ndarray:
    """Run inference. 
    
    Args:
        x: (X, Y, Z) float32 in [0, 1]. (Note: MONAI models expect spatial dims, not ZYX).
        weights_path: absolute path to PyTorch checkpoint.
        arch: dictionary of UNet architecture kwargs.
        patch: spatial size of sliding window.
        overlap: sliding window overlap.
        activation: "sigmoid" etc.
    """
    print(f"Running inference with weights: {weights_path}")
    net = UNet(spatial_dims=3, **arch).to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    net.eval()
    
    tensor = torch.from_numpy(x.astype(np.float32))[None, None].to(device)
    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=patch,
            sw_batch_size=1,
            predictor=net,
            overlap=overlap,
        )
        
    if activation == "sigmoid":
        probs = torch.sigmoid(logits)
    else:
        probs = logits
        
    return probs.squeeze().cpu().numpy().astype(np.float32)

def save_nifti(image: sitk.Image, out_path: Path):
    """Save SimpleITK image to NIfTI."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(out_path))

"""
models.py — blocks 9 to 12.

    read YAML + verify sha256    block 9
    contract gate                block 10
    window HU -> [0,1]           block 11
    sliding-window inference     block 12

THE POINT OF THIS FILE
    A checkpoint file contains only tensors. It does not record the HU window,
    the spacing, the ROI, or even whether its output is a binary mask or a
    coverage fraction — A1-ROI and A3-Coverage have byte-identical
    architectures. That information exists nowhere except the YAML beside it.

    So the YAML is not documentation. It is the missing half of the checkpoint,
    and block 10 is what stops a mismatch from becoming a confident wrong
    number instead of an error.

torch is imported lazily inside infer(). Everything up to and including the
gate runs without it, so the pipeline is testable on a machine with no GPU.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REQUIRED_KEYS = ("id", "weights", "output", "requires", "inference")


# ---------------------------------------------------------------------------
# Block 9 — read the manifest, find and verify the weights
# ---------------------------------------------------------------------------

@dataclass
class Model:
    """A manifest plus the resolved path to its weights."""

    id: str
    name: str
    manifest_path: Path
    weights_path: Path
    output_type: str          # "binary" | "soft"
    output_semantics: str
    requires: dict
    inference: dict
    framework: str
    architecture: dict
    sha256: str | None
    trained_on: str

    def describe(self) -> str:
        r = self.requires
        return (
            f"{self.id}  ({self.output_type})\n"
            f"    weights   {self.weights_path}\n"
            f"    requires  HU {r.get('hu_window')}  "
            f"spacing {r.get('spacing_mm')}  roi {r.get('roi')}\n"
            f"    trained   {self.trained_on}"
        )


def models_root() -> Path:
    """Where checkpoints live. Machine-specific, so it lives in ONE place.

    1. PREDICT_MODELS_ROOT environment variable
    2. models_root in ~/.predict/settings.yaml
    3. ./runs — which already works with no configuration at all
    """
    env = os.environ.get("PREDICT_MODELS_ROOT")
    if env:
        return Path(env)

    settings = Path.home() / ".predict" / "settings.yaml"
    if settings.exists():
        data = yaml.safe_load(settings.read_text()) or {}
        if data.get("models_root"):
            return Path(data["models_root"])

    return Path("runs")


def load_model(manifest_path: str | Path, root: Path | None = None,
               verify: bool = True) -> Model:
    """Read one model YAML and resolve its weights file."""
    manifest_path = Path(manifest_path)
    data = yaml.safe_load(manifest_path.read_text()) or {}

    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(f"{manifest_path.name} is missing: {', '.join(missing)}")

    root = Path(root) if root is not None else models_root()
    weights = Path(data["weights"])
    if not weights.is_absolute():
        weights = root / weights

    model = Model(
        id=data["id"],
        name=data.get("name", data["id"]),
        manifest_path=manifest_path,
        weights_path=weights,
        output_type=data["output"]["type"],
        output_semantics=data["output"].get("semantics", ""),
        requires=data["requires"],
        inference=data["inference"],
        framework=data.get("framework", "monai"),
        architecture=data.get("architecture", {}),
        sha256=data.get("sha256"),
        trained_on=data.get("trained_on", "not stated"),
    )

    if verify and model.framework != "dummy":
        verify_weights(model)
    return model


def verify_weights(model: Model) -> None:
    """Check the file exists and its hash matches the manifest.

    This is what makes the agatston_scoring_a3.py failure impossible to
    repeat: that script declared it was scoring v2 while loading v1, and
    nothing in the system could tell.
    """
    if not model.weights_path.exists():
        raise FileNotFoundError(
            f"{model.id}: weights not found at {model.weights_path}\n"
            f"Set PREDICT_MODELS_ROOT, or fix 'weights:' in "
            f"{model.manifest_path.name}"
        )

    if not model.sha256:
        return          # not yet recorded; run inspect_checkpoint.py

    digest = hashlib.sha256()
    with open(model.weights_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)

    actual = digest.hexdigest()
    if actual != model.sha256:
        raise ValueError(
            f"{model.id}: WRONG WEIGHTS FILE.\n"
            f"  manifest expects  {model.sha256}\n"
            f"  file on disk is   {actual}\n"
            f"  {model.weights_path}\n"
            "The file has been replaced or the manifest points at the wrong "
            "checkpoint. Refusing rather than scoring with unknown weights."
        )


def discover(*dirs: str | Path) -> list[Model]:
    """Every model YAML in the given directories. Missing dirs are skipped."""
    found: list[Model] = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for path in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            try:
                found.append(load_model(path, verify=False))
            except Exception as e:
                print(f"  [skip] {path.name}: {e}")
    return found


# ---------------------------------------------------------------------------
# Block 10 — the contract gate
# ---------------------------------------------------------------------------

def check_contract(provenance: dict, model: Model) -> list[str]:
    """Compare what preprocessing actually did against what the model needs.

    Returns a list of problems; empty means the volume is safe to infer on.
    Every check here guards a failure that produces a plausible wrong number
    rather than an exception.
    """
    problems: list[str] = []
    req = model.requires

    want = req.get("spacing_mm")
    if want:
        got = provenance.get("spacing_mm", [])
        if len(got) != 3 or any(abs(g - w) > 1e-3 for g, w in zip(got, want)):
            problems.append(
                f"spacing is {got} but the model needs {want}. Convolution "
                "receptive fields are fixed in voxels, so lesions would appear "
                "the wrong size."
            )

    want = req.get("orientation")
    if want and provenance.get("orientation") != want:
        problems.append(
            f"orientation is {provenance.get('orientation')} but the model "
            f"needs {want}. Left and right may be swapped."
        )

    want = req.get("roi", "none")
    if want != provenance.get("roi", "none"):
        problems.append(
            f"volume roi is {provenance.get('roi')} but the model was trained "
            f"on {want}. It would see anatomy it never saw in training."
        )

    need = req.get("min_slices", 0)
    have = provenance.get("n_slices", 0)
    if have < need:
        problems.append(
            f"{have} slices but the model needs at least {need} "
            "to fill one inference patch."
        )

    # A normalised volume would silently score zero everywhere downstream,
    # because every peak HU would fall below 130.
    lo, hi = provenance.get("hu_range", [0, 0])
    if hi <= 100:
        problems.append(
            f"HU range is {lo:.0f} to {hi:.0f}; this looks already normalised. "
            "Windowing happens at block 11 and needs raw Hounsfield Units."
        )

    return problems


def check_protocol(volume_warnings: list[str]) -> list[str]:
    """Domain-shift warnings the gate cannot catch.

    Contrast, kVp and gating satisfy every contract field yet still make the
    score meaningless. These are warnings, never refusals: the user decides.
    """
    return [w for w in volume_warnings
            if any(k in w.lower() for k in ("contrast", "kvp", "modality"))]


# ---------------------------------------------------------------------------
# Block 11 — windowing
# ---------------------------------------------------------------------------

def window(array: np.ndarray, hu_window: tuple[float, float]) -> np.ndarray:
    """Map raw HU into [0, 1] by clipping to a window, as MONAI's
    ScaleIntensityRanged(clip=True) does.

    THE SETTING THAT MATTERS MOST. At the project's [0, 1200], 130 HU (the
    Agatston threshold) maps to 0.108. At [100, 1000] it maps to 0.033 — three
    times darker at the exact value the biomarker is defined around, and
    everything below 100 HU flattens to zero. The model still produces a
    confident mask.

    This runs per model, in memory, on the cropped volume. It is never stored,
    because two models may want different windows from the same source.
    """
    lo, hi = float(hu_window[0]), float(hu_window[1])
    if hi <= lo:
        raise ValueError(f"invalid hu_window {hu_window}: max must exceed min")

    out = (np.asarray(array, dtype=np.float32) - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Block 12 — inference
# ---------------------------------------------------------------------------

def infer(volume_array: np.ndarray, model: Model, device: str = "auto",
          progress: bool = False) -> np.ndarray:
    """Run the model. Returns a (Z, Y, X) float array in [0, 1].

    Windowing happens here, so callers pass RAW HU and cannot accidentally
    window twice or with the wrong range.
    """
    if model.framework == "dummy":
        return _infer_dummy(volume_array, model)
    if model.framework == "monai":
        return _infer_monai(volume_array, model, device, progress)
    raise ValueError(
        f"{model.id}: unknown framework {model.framework!r}. "
        "Supported: monai, dummy."
    )


def _infer_dummy(volume_array: np.ndarray, model: Model) -> np.ndarray:
    """A model-shaped stand-in that needs no torch and no checkpoint.

    Thresholds raw HU at 130 — the textbook definition of calcium. It is not a
    segmentation model and will over-call bone and metal. It exists so the
    pipeline, the CLI, the API and the UI can all be built and tested before a
    real checkpoint is wired in, and so a broken checkpoint can be told apart
    from broken plumbing.
    """
    threshold = model.inference.get("threshold_hu", 130.0)
    return (np.asarray(volume_array) >= threshold).astype(np.float32)


def _infer_monai(volume_array: np.ndarray, model: Model, device: str,
                 progress: bool) -> np.ndarray:
    """MONAI UNet with sliding-window inference."""
    import torch
    from monai.inferers import sliding_window_inference
    from monai.networks.nets import UNet

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    arch = model.architecture
    net = UNet(
        spatial_dims=3,
        in_channels=arch.get("in_channels", 1),
        out_channels=arch.get("out_channels", 1),
        channels=tuple(arch.get("channels", (16, 32, 64, 128, 256))),
        strides=tuple(arch.get("strides", (2, 2, 2, 2))),
        num_res_units=arch.get("num_res_units", 2),
        dropout=arch.get("dropout", 0.1),
    ).to(device)

    state = torch.load(model.weights_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # strict=True: a silent partial load is worse than a crash.
    net.load_state_dict(state, strict=True)
    net.eval()

    windowed = window(volume_array, model.requires["hu_window"])
    tensor = torch.from_numpy(windowed)[None, None].to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=tuple(model.inference.get("patch_size", (96, 96, 32))),
            sw_batch_size=model.inference.get("sw_batch_size", 1),
            predictor=net,
            overlap=model.inference.get("sw_overlap", 0.5),
            progress=progress,
        )
        probs = torch.sigmoid(logits)

    return probs.squeeze().cpu().numpy().astype(np.float32)

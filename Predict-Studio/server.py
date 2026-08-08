"""
server.py — HTTP wrapper around pipeline.py. About 200 lines, no new logic.

    pip install fastapi uvicorn
    python server.py            then open http://127.0.0.1:8000

Every route calls the same functions run.py calls. If a number differs between
the CLI and the UI, that is a bug, not a feature — there is one code path.

Jobs run in a background thread and report progress by polling. Server-sent
events would be tidier; polling is four lines and cannot get stuck half-open,
which matters more when the audience is one person demonstrating it live.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import models as model_lib
import pipeline
from load import load_folder
from scoring import ScoringConfig

app = FastAPI(title="PrediCT")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

MODEL_DIRS = ("models", str(Path.home() / ".predict" / "models"))
OUT_DIR = Path("results")

# In-memory state. A restart loses jobs but never loses results, because
# results are on disk the moment each study finishes.
STUDIES: dict[str, dict] = {}
JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models")
def list_models():
    out = []
    for m in model_lib.discover(*MODEL_DIRS):
        ready = m.framework == "dummy" or m.weights_path.exists()
        out.append({
            "id": m.id, "name": m.name,
            "output_type": m.output_type,
            "requires": m.requires,
            "trained_on": m.trained_on,
            "ready": ready,
            "weights": str(m.weights_path),
        })
    return out


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

class ScanRequest(BaseModel):
    folder: str


@app.post("/api/scan")
def scan(req: ScanRequest):
    """Load a folder and register what was found. Does not run any model."""
    try:
        volumes = load_folder(req.folder)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    if not volumes:
        raise HTTPException(
            400,
            "No loadable scans found. Looked for DICOM (by magic bytes) and "
            ".nii/.nii.gz/.nrrd/.mha",
        )

    found = []
    for volume in volumes:
        study_id = uuid.uuid4().hex[:12]
        STUDIES[study_id] = {"volume": volume, "results": {}}
        found.append({
            "study_id": study_id,
            "patient_id": volume.patient_id,
            "shape": list(volume.array.shape),
            "spacing": list(volume.spacing),
            "n_files": volume.n_files,
            "warnings": volume.warnings,
        })
    return found


@app.get("/api/studies")
def list_studies():
    return [
        {"study_id": sid,
         "patient_id": s["volume"].patient_id,
         "shape": list(s["volume"].array.shape),
         "spacing": list(s["volume"].spacing),
         "warnings": s["volume"].warnings,
         "models_run": list(s["results"])}
        for sid, s in STUDIES.items()
    ]


@app.get("/api/studies/{study_id}/slice/{index}")
def get_slice(study_id: str, index: int, window: float = 1200, level: float = 300):
    """One slice as raw 8-bit greyscale bytes, window/level applied.

    Bytes rather than PNG so the browser can draw it straight into a canvas
    with no decode step, which keeps scrubbing smooth. The mask overlay is
    fetched separately and composited client-side, so opacity can change
    without refetching either.
    """
    study = STUDIES.get(study_id)
    if not study:
        raise HTTPException(404, "unknown study")

    array = study["volume"].array
    if not 0 <= index < array.shape[0]:
        raise HTTPException(404, f"slice {index} outside 0..{array.shape[0] - 1}")

    lo, hi = level - window / 2, level + window / 2
    plane = np.clip((array[index] - lo) / (hi - lo), 0, 1)
    return Response(
        (plane * 255).astype(np.uint8).tobytes(),
        media_type="application/octet-stream",
        headers={"X-Shape": f"{plane.shape[0]},{plane.shape[1]}"},
    )


@app.get("/api/studies/{study_id}/mask/{model_id}/{index}")
def get_mask_slice(study_id: str, model_id: str, index: int):
    study = STUDIES.get(study_id)
    if not study or model_id not in study["results"]:
        raise HTTPException(404, "no result for that study and model")

    mask = study["results"][model_id]["mask"]
    if not 0 <= index < mask.shape[0]:
        raise HTTPException(404, "slice outside volume")

    plane = np.clip(mask[index], 0, 1)
    return Response(
        (plane * 255).astype(np.uint8).tobytes(),
        media_type="application/octet-stream",
        headers={"X-Shape": f"{plane.shape[0]},{plane.shape[1]}"},
    )


@app.get("/api/studies/{study_id}/score/{model_id}")
def get_score(study_id: str, model_id: str):
    study = STUDIES.get(study_id)
    if not study or model_id not in study["results"]:
        raise HTTPException(404, "no result for that study and model")
    entry = study["results"][model_id]
    return {"summary": entry["score"].summary(), "lesions": entry["score"].ledger()}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    study_ids: list[str]
    model_ids: list[str]
    legacy: bool = False
    lesion_definition: str = "2d"
    force: bool = False


@app.post("/api/run")
def start_run(req: RunRequest):
    """Start a job. One study or two hundred — the same route, the same loop."""
    by_id = {m.id: m for m in model_lib.discover(*MODEL_DIRS)}
    chosen = [by_id[i] for i in req.model_ids if i in by_id]
    if not chosen:
        raise HTTPException(400, f"no such model(s): {req.model_ids}")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "running", "done": 0,
        "total": len(req.study_ids) * len(chosen),
        "results": [], "current": "",
    }

    config = (ScoringConfig.legacy() if req.legacy else ScoringConfig()).with_(
        lesion_definition=req.lesion_definition
    )

    def work():
        job = JOBS[job_id]
        try:
            for study_id in req.study_ids:
                study = STUDIES.get(study_id)
                if not study:
                    job["done"] += len(chosen)
                    continue

                for model in chosen:
                    volume = study["volume"]
                    job["current"] = f"{volume.patient_id} x {model.id}"

                    result = pipeline.process_study(
                        volume, model, scoring_config=config,
                        out_dir=OUT_DIR, cache_dir=Path(".cache"),
                        force=req.force,
                    )

                    if result.status == "ok":
                        prepared = pipeline.preprocess(volume, model.requires,
                                                       cache_dir=Path(".cache"))
                        study["results"][model.id] = {
                            "score": result.score,
                            "mask": model_lib.infer(prepared.array, model),
                        }

                    job["results"].append({
                        "study_id": study_id,
                        "patient_id": volume.patient_id,
                        "model_id": model.id,
                        "status": result.status,
                        "agatston": result.score.agatston if result.score else None,
                        "risk": result.score.risk_category if result.score else None,
                        "problems": result.problems,
                        "warnings": result.warnings,
                    })
                    job["done"] += 1

            job["status"] = "done"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = f"{type(e).__name__}: {e}"

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/results.csv")
def download_results():
    path = OUT_DIR / "results.csv"
    if not path.exists():
        raise HTTPException(404, "nothing scored yet")
    return FileResponse(path, filename="results.csv")


ui_dir = Path(__file__).parent / "ui"
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    print("\n  PrediCT  ->  http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

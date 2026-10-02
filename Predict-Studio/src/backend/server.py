import threading
import uuid
import shutil
import traceback
from pathlib import Path
from typing import List

from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.backend.paths import (study_id_from_series, upload_dir, raw_dir, out_dir,
                               study_dirs, safe_name, DATA)
from src.backend.registry import list_models
from src.backend.run import run
from src.backend.ingest import fix_extensions

app = FastAPI(title="PrediCT Server")

# In-memory job state
JOBS = {}

def checked(name: str) -> str:
    """paths.safe_name, reported to the client as a 400 rather than a crash."""
    try:
        return safe_name(name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def temp_upload_dir(temp_id: str) -> Path:
    """The staging folder of an upload; only ids this server issued are accepted."""
    if not checked(temp_id).startswith("temp_"):
        raise HTTPException(status_code=400, detail="Not an upload id.")
    return DATA / "uploads" / temp_id

@app.post("/studies")
async def upload_study(files: List[UploadFile] = File(...), custom_name: str = Form(None)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    # Validate the name before anything is written: it becomes data/raw/<name>.
    custom_name = checked(custom_name) if custom_name and custom_name.strip() else None

    temp_id = f"temp_{uuid.uuid4().hex}"
    temp_dir = DATA / "uploads" / temp_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        original_parents = {}
        for f in files:
            file_name = Path(f.filename).name
            file_path = temp_dir / file_name
            original_parents[file_name] = str(Path(f.filename).parent)
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
                
        fix_extensions(temp_dir)
        
        invalid_files = []
        dicom_dirs = set()
        for f_path in temp_dir.iterdir():
            if f_path.is_file():
                lower_name = f_path.name.lower()
                is_valid = lower_name.endswith(".dcm") or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")
                
                original_name = f_path.name
                if original_name not in original_parents and original_name.endswith(".dcm"):
                    original_name = original_name[:-4]
                    
                parent_dir = original_parents.get(original_name, "")
                
                if not is_valid:
                    invalid_files.append(f_path.name)
                elif lower_name.endswith(".dcm"):
                    dicom_dirs.add(parent_dir)
                
        if len(dicom_dirs) > 1:
            raise HTTPException(status_code=400, detail="Multiple folders contain DICOM files. Please upload the specific folder.")
        
        if invalid_files:
            return {"requires_cleaning": True, "temp_id": temp_id, "invalid_files": invalid_files}
            
        # Get study ID
        if custom_name:
            study_id = custom_name
        else:
            try:
                study_id = study_id_from_series(temp_dir)
            except StopIteration:
                raise HTTPException(status_code=400, detail="No DICOM or NIfTI files found in the upload.")

        final_dir = raw_dir(study_id)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(final_dir)
        
        return {"study_id": study_id, "requires_cleaning": False}
    except HTTPException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/studies/clean/{temp_id}")
def clean_and_commit_study(temp_id: str, custom_name: str = None):
    temp_dir = temp_upload_dir(temp_id)
    custom_name = checked(custom_name) if custom_name and custom_name.strip() else None
    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="Temp directory not found.")
        
    try:
        for f in temp_dir.iterdir():
            lower_name = f.name.lower()
            is_valid = lower_name.endswith(".dcm") or lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")
            if not is_valid:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
                    
        if custom_name:
            study_id = custom_name
        else:
            try:
                study_id = study_id_from_series(temp_dir)
            except StopIteration:
                raise HTTPException(status_code=400, detail="No DICOM or NIfTI files remained after cleaning.")

        final_dir = raw_dir(study_id)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_dir.rename(final_dir)
        
        return {"study_id": study_id}
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
        
@app.delete("/studies/clean/{temp_id}")
def abort_upload(temp_id: str):
    temp_dir = temp_upload_dir(temp_id)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return {"status": "aborted"}

@app.get("/studies")
def get_studies():
    out_root = DATA / "out"
    if not out_root.exists():
        return []

    results = []
    for d in out_root.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            found = False
            for md in d.iterdir():
                if md.is_dir():
                    results.append({"id": d.name, "model": md.name})
                    found = True
            if not found:
                results.append({"id": d.name, "model": "a1-roi"})
    # Sort results by id as a fallback
    results.sort(key=lambda x: (x["id"], x["model"]))
    return results

@app.get("/models")
def get_models():
    # Validated manifests only: a broken one fails loudly instead of being
    # listed. crop is the model's own default; a run may override it.
    models = [{"id": m["id"], "name": m["name"], "crop": m["crop"]} for m in list_models()]
    return sorted(models, key=lambda m: m["id"])

@app.get("/raw_patients")
def get_raw_patients():
    raw_root = DATA / "raw"
    if not raw_root.exists():
        return []

    patients = []
    for d in raw_root.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            uploaded = datetime.fromtimestamp(d.stat().st_mtime).isoformat(timespec="seconds")
            # Find the inner dicom directory
            # Usually it's the first directory inside the patient folder
            dicom_dirs = [sub for sub in d.iterdir() if sub.is_dir()]
            if dicom_dirs:
                dicom_dir = dicom_dirs[0]
                patients.append({"id": d.name, "path": str(dicom_dir.absolute()), "uploaded": uploaded})
            else:
                patients.append({"id": d.name, "path": str(d.absolute()), "uploaded": uploaded})

    patients.sort(key=lambda x: x["uploaded"], reverse=True)   # newest upload first
    return patients

class JobRequest(BaseModel):
    study_id: str | None = None
    input_path: str | None = None
    model_id: str
    crop: bool | None = None   # None = the model's manifest default

# One run at a time: two TotalSegmentator / nnUNet runs would compete for the
# same GPU memory. The lock makes check-and-start a single step.
JOBS_LOCK = threading.Lock()

@app.post("/jobs")
def start_job(req: JobRequest):
    if not req.study_id and not req.input_path:
        raise HTTPException(status_code=400, detail="Must provide study_id or input_path")

    if req.input_path and not req.study_id:
        # A scan uploaded through the UI is data/raw/<id>[/<series>]; anywhere
        # else the parent folder is usually the patient ID.
        p = Path(req.input_path)
        req.study_id = p.name if p.parent.resolve() == (DATA / "raw").resolve() else p.parent.name
    req.study_id = checked(req.study_id)   # becomes data/out/<study_id>
    req.model_id = checked(req.model_id)

    job_id = uuid.uuid4().hex[:12]

    with JOBS_LOCK:
        busy = [j for j in JOBS.values() if j["status"] == "running"]
        if busy:
            raise HTTPException(status_code=409, detail=(
                f"A run is already in progress ({busy[0]['study_id']} · {busy[0]['model_id']}). "
                "Start the next one when it finishes."))
        JOBS[job_id] = {
            "job_id": job_id,
            "study_id": req.study_id,
            "model_id": req.model_id,
            "crop": req.crop,
            "started": datetime.now().isoformat(timespec="seconds"),
            "status": "running",
            "stage": "started",
            "pct": 0.0,
            "error": None
        }

    def work():
        job = JOBS[job_id]
        import subprocess, sys
        from collections import deque
        tail = deque(maxlen=20)   # last output lines, reported if the run fails
        try:
            cmd = [sys.executable, "-m", "src.backend.run", "--model", req.model_id]
            if req.input_path:
                cmd.extend(["--input", req.input_path])
                if req.study_id:
                    cmd.extend(["--study", req.study_id])
            else:
                cmd.extend(["--study", req.study_id])
            if req.crop is not None:
                cmd.append("--crop" if req.crop else "--no-crop")

            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1
            )
            
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("[") and "%]" in line:
                    try:
                        pct_str, rest = line.split("%]", 1)
                        pct_val = float(pct_str.strip().strip("[")) / 100.0
                        stage_name = rest.strip()
                        job["pct"] = pct_val
                        job["stage"] = stage_name
                    except:
                        pass
                else:
                    tail.append(line)

            process.wait()
            if process.returncode == 0:
                job["status"] = "done"
                job["pct"] = 1.0
                job["stage"] = "done"
            else:
                job["status"] = "failed"
                # The traceback's last line names the actual problem.
                job["error"] = "\n".join(tail) or f"Process exited with code {process.returncode}"

        except Exception as e:
            job["status"] = "failed"
            job["error"] = traceback.format_exc()

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}

@app.get("/jobs")
def list_jobs():
    """Runs started since this server started, newest first. The UI reads this
    on every page load, so a run survives reloads and navigation."""
    return sorted(JOBS.values(), key=lambda j: j["started"], reverse=True)

@app.delete("/studies/{study_id}")
def delete_study(study_id: str, model: List[str] = Query(default=[]),
                 everything: bool = Query(False, alias="all")):
    """Delete some results of a study (?model=a&model=b), or everything the
    study owns (?all=true): uploaded scan, prep cache and all results."""
    study_id = checked(study_id)
    if any(j["status"] == "running" and j["study_id"] == study_id for j in JOBS.values()):
        raise HTTPException(status_code=409, detail=f"A run for {study_id} is in progress.")

    if everything:
        targets = study_dirs(study_id)
    elif model:
        targets = [out_dir(study_id, checked(m)) for m in model]
    else:
        raise HTTPException(status_code=400, detail="Say what to delete: ?model=<id> or ?all=true")

    existing = [p for p in targets if p.exists()]
    if not existing:
        raise HTTPException(status_code=404, detail=f"Nothing to delete for {study_id}.")
    try:
        for p in existing:
            shutil.rmtree(p)
        # a study whose last result was deleted should not remain as an empty folder
        results = DATA / "out" / study_id
        if results.exists() and not any(results.iterdir()):
            results.rmdir()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not delete: {e}")
    return {"deleted": [str(p.relative_to(DATA)) for p in existing]}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# Mount static files
data_dir = Path("data")
ui_dir = Path("ui")

if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
    
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

from fastapi.responses import RedirectResponse
@app.get("/")
def read_root():
    return RedirectResponse(url="/ui/index.html")

if __name__ == "__main__":
    import uvicorn
    print("\n  PrediCT Studio -> http://127.0.0.1:8001\n")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")

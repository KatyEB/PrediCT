import threading
import uuid
import shutil
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.backend.paths import study_id_from_series, upload_dir, DATA
from src.backend.registry import list_models
from src.backend.run import run

app = FastAPI(title="PrediCT Server")

# In-memory job state
JOBS = {}

@app.post("/studies")
async def upload_study(files: List[UploadFile] = File(...), custom_name: str = Form(None)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    temp_id = f"temp_{uuid.uuid4().hex}"
    temp_dir = DATA / "uploads" / temp_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        invalid_files = []
        for f in files:
            is_dcm = f.filename.lower().endswith(".dcm")
            if not is_dcm:
                invalid_files.append(Path(f.filename).name)
                
            file_path = temp_dir / Path(f.filename).name
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
        
        if invalid_files:
            return {"requires_cleaning": True, "temp_id": temp_id, "invalid_files": invalid_files}
            
        # Get study ID
        if custom_name:
            study_id = custom_name.strip()
        else:
            try:
                study_id = study_id_from_series(temp_dir)
            except StopIteration:
                raise HTTPException(status_code=400, detail="No files with .dcm extension found in the upload.")
            
        final_dir = DATA / "raw" / study_id
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
    temp_dir = DATA / "uploads" / temp_id
    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="Temp directory not found.")
        
    try:
        for f in temp_dir.iterdir():
            if not f.name.lower().endswith(".dcm"):
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
                    
        if custom_name:
            study_id = custom_name.strip()
        else:
            try:
                study_id = study_id_from_series(temp_dir)
            except StopIteration:
                raise HTTPException(status_code=400, detail="No DICOM files remained after cleaning.")
            
        final_dir = DATA / "raw" / study_id
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
    temp_dir = DATA / "uploads" / temp_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return {"status": "aborted"}

@app.get("/studies")
def get_studies():
    out_dir = DATA / "out"
    if not out_dir.exists():
        return []
    
    results = []
    for d in out_dir.iterdir():
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
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    return [{"id": d.name} for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

@app.get("/raw_patients")
def get_raw_patients():
    raw_dir = DATA / "raw"
    if not raw_dir.exists():
        return []
        
    patients = []
    for d in raw_dir.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            # Find the inner dicom directory
            # Usually it's the first directory inside the patient folder
            dicom_dirs = [sub for sub in d.iterdir() if sub.is_dir()]
            if dicom_dirs:
                dicom_dir = dicom_dirs[0]
                patients.append({"id": d.name, "path": str(dicom_dir.absolute())})
            else:
                patients.append({"id": d.name, "path": str(d.absolute())})
                
    patients.sort(key=lambda x: (0, int(x["id"])) if x["id"].isdigit() else (1, x["id"]))
    return patients

class JobRequest(BaseModel):
    study_id: str | None = None
    input_path: str | None = None
    model_id: str

@app.post("/jobs")
def start_job(req: JobRequest):
    if not req.study_id and not req.input_path:
        raise HTTPException(status_code=400, detail="Must provide study_id or input_path")
        
    if req.input_path and not req.study_id:
        # Infer study_id from the parent folder name (usually the patient ID)
        req.study_id = Path(req.input_path).parent.name

    job_id = uuid.uuid4().hex[:12]
    
    JOBS[job_id] = {
        "status": "running",
        "stage": "started",
        "pct": 0.0,
        "error": None
    }
    
    def work():
        job = JOBS[job_id]
        import subprocess, sys
        try:
            cmd = [sys.executable, "-m", "src.backend.run", "--model", req.model_id]
            if req.input_path:
                cmd.extend(["--input", req.input_path])
                if req.study_id:
                    cmd.extend(["--study", req.study_id])
            else:
                cmd.extend(["--study", req.study_id])
                
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
                        
            process.wait()
            if process.returncode == 0:
                job["status"] = "done"
                job["pct"] = 1.0
                job["stage"] = "done"
            else:
                job["status"] = "failed"
                job["error"] = f"Process exited with code {process.returncode}"
                
        except Exception as e:
            job["status"] = "failed"
            job["error"] = traceback.format_exc()
            
    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}

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

import threading
import uuid
import shutil
import traceback
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
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
async def upload_study(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Save to a temporary directory first
    temp_dir = DATA / "uploads" / f"temp_{uuid.uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for f in files:
            file_path = temp_dir / f.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(f.file, buffer)
        
        # Get study ID
        try:
            study_id = study_id_from_series(temp_dir)
        except StopIteration:
            raise HTTPException(status_code=400, detail="No files with .dcm extension found in the upload.")
            
        final_dir = upload_dir(study_id)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        temp_dir.rename(final_dir)
        
        return {"study_id": study_id}
    except HTTPException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    except Exception as e:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/studies")
def get_studies():
    uploads = DATA / "uploads"
    if not uploads.exists():
        return []
    return [d.name for d in uploads.iterdir() if d.is_dir() and not d.name.startswith("temp_")]

@app.get("/models")
def get_models():
    # Convert Path objects to strings before returning so FastAPI can serialize them
    models = list_models()
    for m in models:
        if 'weights_path' in m:
            m['weights_path'] = str(m['weights_path'])
    return models

class JobRequest(BaseModel):
    study_id: str
    model_id: str

@app.post("/jobs")
def start_job(req: JobRequest):
    job_id = uuid.uuid4().hex[:12]
    
    JOBS[job_id] = {
        "status": "running",
        "stage": "started",
        "pct": 0.0,
        "error": None
    }
    
    def work():
        job = JOBS[job_id]
        try:
            def progress(stage, pct):
                job["stage"] = stage
                job["pct"] = pct
                
            run(study_id=req.study_id, model_id=req.model_id, progress=progress)
            job["status"] = "done"
            job["pct"] = 1.0
            job["stage"] = "done"
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
ui_dir = Path("src/ui")

if data_dir.exists():
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
    
if ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(ui_dir), html=True), name="ui")

if __name__ == "__main__":
    import uvicorn
    print("\n  PrediCT Studio -> http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

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
    out_dir = DATA / "out"
    if not out_dir.exists():
        return []
    
    results = []
    for d in out_dir.iterdir():
        if d.is_dir() and not d.name.startswith("."):
            model = "a1-roi"
            for md in d.iterdir():
                if md.is_dir():
                    model = md.name
                    break
            results.append({"id": d.name, "model": model})
    # Sort results by id as a fallback
    results.sort(key=lambda x: x["id"])
    return results

@app.get("/models")
def get_models():
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    return [{"id": d.name} for d in models_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

class JobRequest(BaseModel):
    study_id: str | None = None
    input_path: str | None = None
    model_id: str

@app.post("/jobs")
def start_job(req: JobRequest):
    if not req.study_id and not req.input_path:
        raise HTTPException(status_code=400, detail="Must provide study_id or input_path")
        
    if req.input_path:
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
        try:
            def progress(stage, pct):
                job["stage"] = stage
                job["pct"] = pct
                
            run(study_id=req.study_id, model_id=req.model_id, progress=progress, 
                custom_input=Path(req.input_path) if req.input_path else None)
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
    print("\n  PrediCT Studio -> http://127.0.0.1:8080\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

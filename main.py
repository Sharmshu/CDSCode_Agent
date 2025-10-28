"""
main.py
FastAPI app + background job controller for modular AI agents.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Local modules
from utils.file_utils import get_job_dir, zip_outputs
from utils.job_utils import split_sections
from agents.cds.cds_agent import CdsAgent
# ------------------------------ CONFIG ------------------------------
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

from utils.logger_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

app = FastAPI(title="SAP ABAP Code Generator (AI Agents)")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# In-memory job store
jobs = {}


# ------------------------------ REQUEST MODEL ------------------------------
class RequirementPayload(BaseModel):
    REQUIREMENT: str


# ------------------------------ BACKGROUND JOB ------------------------------
def run_job(job_id: str, requirement_text: str):
    logger.info(f"Job {job_id} started")

    # Create job folder
    job_dir = get_job_dir()
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        # --- Split Sections ---
        sections = split_sections(requirement_text)
        logger.info(f"[{job_id}] Parsed sections: {list(sections.keys())}")

        # -------------------- Section Mapping --------------------
        def get_section_text(prefix: str):
            """Combine parent and child sections by number prefix."""
            matched = [v for k, v in sections.items() if k.startswith(prefix)]
            return "\n\n".join(matched).strip()

        cds_text = get_section_text("7")  # Section 7 (CDS view)
        
        logger.info(f"[{job_id}] Section 7 length: {len(cds_text)}")
    
        # -------------------- Run Agents --------------------
        cds_agent = CdsAgent(job_dir=job_dir)

        # --- Structure Agent ---
        logger.info(f"[{job_id}] Running cds...")
        cds_output = cds_agent.run(cds_text)
        path_structure = cds_output["path"]
        cds_purpose = cds_output["purpose"]
        cds_result = path_structure.read_text(encoding="utf-8") if path_structure.exists() else ""

        # Combine purposes dynamically
        purposes = {
            "CDS": cds_purpose,
        }
        logger.info(f"[{job_id}] Purposes received: {list(purposes.keys())}")
        # -------------------- Zip Results --------------------
        zip_path = zip_outputs(job_dir, [path_structure ], job_id)
        logger.info(f"[{job_id}] Finished successfully. ZIP: {zip_path}")

        # Update job record
        jobs[job_id].update({
            "status": "finished",
            "finished_at": datetime.utcnow().isoformat(),
            "zip_path": str(zip_path),
            "outputs": {
                "structure": str(path_structure.name),
            },
        })

        logger.info(f"✅ Job {job_id} completed. File ready at: {zip_path}")

    except Exception as e:
        logger.exception(f"❌ Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ------------------------------ ENDPOINTS ------------------------------
@app.post("/generate")
def create_job(payload: RequirementPayload, background_tasks: BackgroundTasks):
    """Start job with unified document text (in REQUIREMENT key)."""
    requirement_text = payload.REQUIREMENT.strip()
    if not requirement_text:
        raise HTTPException(status_code=400, detail="REQUIREMENT text is missing or empty")

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "created_at": datetime.utcnow().isoformat()}

    background_tasks.add_task(run_job, job_id, requirement_text)
    logger.info(f"Job {job_id} queued")

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Check current job status and download if ready."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") == "finished":
        zip_path = Path(job["zip_path"])
        if not zip_path.exists():
            raise HTTPException(status_code=500, detail="ZIP file missing")

        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip",
            headers={"X-Job-ID": job_id, "X-Status": "finished"},
        )

    return JSONResponse(job)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
"""
main.py
FastAPI app + background job controller for modular AI agents.
"""

import os,io,zipfile
import uuid
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

# Local modules
from utils.file_utils import get_job_dir, zip_outputs
from utils.job_utils import split_sections
from agents.cds.cds_agent import CdsAgent
from agents.ValueHelp.value_help_agent import ValueHelpAgent
# ------------------------------ CONFIG ------------------------------
load_dotenv()

# openai_api_key = os.getenv("OPENAI_API_KEY")

from utils.logger_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

app = FastAPI(title="SAP ABAP Code Generator (AI Agents)")
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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

        value_help_text = get_section_text("7")  #Section 7 (Value Help)
        cds_text = get_section_text("8")  # Section 8 (CDS view)
        
        logger.info(f"[{job_id}] Section 7 length: {len(value_help_text)}")
        logger.info(f"[{job_id}] Section 8 length: {len(cds_text)}")
        
        # -------------------- Initialize --------------------
        cds_result = ""
        value_help_result = ""
        value_help_entity = ""
        files_to_zip = []
    
        # -------------------- Run Agents --------------------
        # --- Value help Agent ---
        if value_help_text:
           logger.info(f"[{job_id}] Running value help...") 
           value_help_agent = ValueHelpAgent(job_dir=job_dir)
           value_help_output = value_help_agent.run(value_help_text)
           value_help_code = value_help_output.get("code", "")
           value_help_purpose = value_help_output.get("purpose", "")
           value_help_entity = value_help_output.get("entity", "")

           
           if value_help_code: 
               # Extract CDS entity name for reuse in main CDS
                import re
                match = re.search(r"define\s+view\s+entity\s+(\w+)", value_help_code)
                if match:
                    value_help_entity = match.group(1)
                    logger.info(f"[{job_id}] Extracted Value Help entity: {value_help_entity}")
                files_to_zip.append(("value_help_requirements.txt", value_help_code))
                logger.info(f"[{job_id}] ✅ Value Help CDS generated successfully.")
                logger.info(f"[{job_id}] 📘 Purpose: {value_help_purpose}")
           else:
                logger.warning(f"[{job_id}] ValueHelpAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No Value Help section found — skipping ValueHelpAgent.")
        
        # --- CDS Agent ---
        if cds_text:
           logger.info(f"[{job_id}] Running cds...")
           cds_agent = CdsAgent(job_dir=job_dir)
           cds_output = cds_agent.run(
                cds_text,
                metadata={
                    "value_help_entity": value_help_entity,
                    "value_help_purpose": value_help_purpose
                }
            )
           
           cds_code = cds_output.get("code", "")
           if cds_code: 
               files_to_zip.append(("cds_view.txt", cds_code))
           else:
                logger.warning(f"[{job_id}] CdsAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No CDS section found — skipping CdsAgent.")
        # cds_result = path_structure.read_text(encoding="utf-8") if path_structure.exists() else ""

        # -------------------- In memory Zip Results --------------------
        if not files_to_zip:
            raise ValueError("No valid sections found — no output generated.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_to_zip:
                zf.writestr(filename, content)
        zip_buffer.seek(0)

        # Update job record
        jobs[job_id].update({
            "status": "finished",
            "finished_at": datetime.utcnow().isoformat(),
            "zip_bytes": zip_buffer.getvalue(),
            "outputs": [f[0] for f in files_to_zip],
            },
        )

        logger.info(f"✅ Job {job_id} completed successfully (in-memory ZIP).")

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
    
    status = job.get("status")

    if status == "finished":
        # If in-memory ZIP is available
        if "zip_bytes" in job:
            zip_buffer = io.BytesIO(job["zip_bytes"])
            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{job_id}_results.zip"',
                    "X-Job-ID": job_id,
                    "X-Status": "finished"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="ZIP bytes not found in memory")

    return JSONResponse(job)

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
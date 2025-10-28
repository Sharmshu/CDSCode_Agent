import re
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger("file_utils")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] file_utils - %(message)s"
)


def extract_sections_from_text(text: str) -> Dict[str, str]:
    """
    Extracts numbered sections (e.g., SECTION 1, 2.1, etc.) from text.

    Args:
        text (str): Input text to parse.

    Returns:
        dict: Mapping of section numbers to their content.
    """
    sections = {}
    pattern = r"(?:SECTION\s+)?(\d+(?:\.\d+)*)[^\n]*\n(.*?)(?=(?:SECTION\s+\d+(?:\.\d+)*)|\Z)"
    matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        num = match.group(1).strip()
        content = match.group(2).strip()
        sections[num] = content

    logger.info(f"Extracted {len(sections)} sections: {list(sections.keys())}")
    return sections


def get_job_dir(base_dir: str = "jobs") -> Path:
    """
    Creates a unique job directory based on timestamp.

    Args:
        base_dir (str): Base directory to store job folders.

    Returns:
        Path: Path to the created job directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = Path(base_dir) / f"job_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created job directory: {job_dir}")
    return job_dir


def zip_outputs(job_dir: Path, file_paths: List[Path], job_id: str) -> Path:
    """
    Zips a list of files into a single archive inside the job directory.

    Args:
        job_dir (Path): Directory where the zip will be created.
        file_paths (list[Path]): List of file paths to include.
        job_id (str): Unique identifier for the zip name.

    Returns:
        Path: Path to the created zip file.
    """
    zip_path = job_dir / f"{job_id}_final.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            if fp and fp.exists():
                zf.write(fp, arcname=fp.name)
                logger.info(f"Added file to zip: {fp.name}")
            else:
                logger.warning(f"Skipped missing file: {fp}")

    logger.info(f"Created zip file: {zip_path}")
    return zip_path
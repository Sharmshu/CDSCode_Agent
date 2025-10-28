"""
utils/job_utils.py
Document splitting utilities.

This splits sections that look like:
  SECTION: 1. Purpose
  SECTION: 2. Scope
  SECTION: 3. Functional Requirements
  SECTION: 5. Technical Architecture
  SECTION: 5.2 Data Sources

It returns a dict with normalized keys like:
  '1. purpose' -> text
  '5.2 data sources' -> text
  'full' -> entire document (fallback)
"""

import re
from typing import Dict

SECTION_HEADER_RE = re.compile(r"SECTION:\s*([\d\.]+\s*\.\s*[^:\n\r]+)", re.IGNORECASE)

def _normalize_title(title: str) -> str:
    # Normalize: "1. Purpose" -> "1. purpose"
    return title.strip().lower()

def split_sections(document: str) -> Dict[str, str]:
    if not document:
        return {"full": ""}

    # Find header positions
    matches = list(SECTION_HEADER_RE.finditer(document))
    if not matches:
        return {"full": document.strip()}

    sections = {}
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(document)
        raw_title = m.group(1)
        key = _normalize_title(raw_title)
        content = document[start:end].strip()
        sections[key] = content

    # Always include full fallback
    sections["full"] = document.strip()
    return sections
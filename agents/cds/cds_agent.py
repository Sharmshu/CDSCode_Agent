# agents/cds/cds_agent.py

import logging
from pathlib import Path
from openai import OpenAI
from agents.base_agent import BaseAgent


class CdsAgent(BaseAgent):
    """
    Agent that generates RAP-ready CDS views from textual requirements.
    """

    def _init_llm(self):
        """
        Initialize the LLM client (e.g., OpenAI GPT).
        """
        return OpenAI()

    def run(self, section_text: str, metadata=None):
        """
        Generate a RAP-ready CDS view from requirement text.

        Returns:
            {
                "path": Path(...),
                "purpose": str
            }
        """
        self.logger.info("Running RAP CDS agent...")

        # -------------------- 1️⃣ SYSTEM PROMPT --------------------
        system_prompt = (
            "You are an expert SAP ABAP developer specializing in the RESTful ABAP Programming Model (RAP). "
            "Generate a RAP-ready ABAP CDS view entity definition (no SQL view name). "
            "Follow SAP conventions for annotations, naming, and structure. "
            "Output only the CDS code, without any markdown, explanations, or comments."
        )

        # -------------------- 2️⃣ USER PROMPT --------------------
        user_prompt = f"""
        Requirement:
        {section_text.strip()}

        Please produce a valid RAP CDS view entity definition:
        - Use `define root view entity` or `define view entity` depending on context.
        - Do NOT include `@AbapCatalog.sqlViewName`.
        - Place all annotations immediately **above** the define statement.
        - Include annotations like:
            @AccessControl.authorizationCheck: #CHECK
            @EndUserText.label
            @Metadata.ignorePropagatedAnnotations: true
            @ObjectModel:compositionRoot: true if applicable.
        - Include realistic field names, types, and join logic based on requirement.
        - Follow RAP conventions (e.g., keys, compositions, associations).
        """

        # -------------------- 3️⃣ LLM GENERATION --------------------
        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25,
        )

        cds_code = response.choices[0].message.content.strip()

        # -------------------- 4️⃣ SUMMARIZE PURPOSE --------------------
        purpose_prompt = (
            f"Summarize briefly what this CDS view entity is for:\n\n{cds_code}\n\n"
            "Return a single sentence (e.g., 'RAP root entity for sales order header and items')."
        )

        purpose_resp = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You summarize CDS entities."},
                {"role": "user", "content": purpose_prompt}
            ],
            temperature=0.2,
        )

        purpose = purpose_resp.choices[0].message.content.strip()

        # -------------------- 5️⃣ SAVE OUTPUT --------------------
        cds_file = self.job_dir / "rap_cds_view.abap"
        cds_file.write_text(cds_code, encoding="utf-8")

        self.logger.info(f"RAP CDS view saved to: {cds_file}")
        self.logger.info(f"Purpose: {purpose}")

        return {
            "path": cds_file,
            "purpose": purpose
        }
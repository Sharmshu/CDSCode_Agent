import logging
from pathlib import Path
from openai import OpenAI
from agents.base_agent import BaseAgent


class CdsAgent(BaseAgent):
    """
    Agent that generates RAP-ready CDS views using LLM and plain RAG file.
    """

    def _init_llm(self):
        """Initialize the LLM client (OpenAI GPT)."""
        api_key = None
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment.")
        return OpenAI(api_key=api_key)

    # ------------------- RAG CONTEXT -------------------
    def _get_rag_context(self) -> str:
        """
        Load the CDS reference (RAG) file and return content as string.
        """
        rag_file = Path(__file__).parent / "cds_requirements.txt"
        if not rag_file.exists():
            raise FileNotFoundError(f"RAG file not found: {rag_file}")
        return rag_file.read_text(encoding="utf-8").strip()

    # ------------------- MAIN RUN -------------------
    def run(self, section_text: str, metadata=None):
        """
        Generate a RAP-ready CDS view entity definition using requirement and RAG context.
        """
        self.logger.info("Running RAP CDS agent with RAG file context...")

        # Step 1: Load RAG file
        rag_context = self._get_rag_context()

        # Step 2: Prepare prompts
        system_prompt = (
            "You are an expert SAP ABAP developer specializing in the RESTful ABAP Programming Model (RAP). "
            "Generate a fully valid RAP-ready ABAP CDS view entity definition (without SQL view name). "
            "Follow SAP RAP conventions and mandatory annotations."
        )

        user_prompt = f"""
        Requirement:
        {section_text.strip()}

        Reference knowledge (from internal RAG file):
        {rag_context}

        Please produce a valid RAP CDS view entity definition:
        - Place all annotations immediately above the define statement.
        - Use correct RAP structure, key fields, and naming.
        - Ensure syntax correctness.
        """

        # Step 3: Generate CDS code
        response = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.25,
        )

        cds_code = response.choices[0].message.content.strip()

        # Step 4: Summarize purpose
        purpose_prompt = (
            f"Summarize briefly the purpose of this CDS view:\n\n{cds_code}\n\n"
            "Return one concise sentence (e.g., 'Root entity for Sales Order header in RAP BO.')."
        )

        purpose_resp = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You summarize SAP CDS entities."},
                {"role": "user", "content": purpose_prompt}
            ],
            temperature=0.2,
        )

        purpose = purpose_resp.choices[0].message.content.strip()

        # Step 5: Save result
        cds_file = self.job_dir / "rap_cds_view.abap"
        cds_file.write_text(cds_code, encoding="utf-8")

        self.logger.info(f"💾 RAP CDS view saved to: {cds_file}")
        self.logger.info(f"📝 Purpose: {purpose}")

        return {"path": cds_file, "purpose": purpose}

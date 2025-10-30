import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class ValueHelpAgent(BaseAgent):
    """
    Generates RAP-ready Value Help CDS views (F4 help) using LLM
    and optional RAG knowledge base.
    """

    # ------------------ LLM INIT ------------------
    def _init_llm(self):
        """Initialize the LLM client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        return ChatOpenAI(
            model_name=os.getenv("VALUE_HELP_MODEL_NAME", "gpt-4.1-mini"),
            temperature=float(os.getenv("VALUE_HELP_TEMPERATURE", "0.3")),
            openai_api_key=api_key
        )

    # ------------------ VECTORSTORE INIT ------------------
    def _init_vectorstore(self):
        """Load or build FAISS vector DB from value_help_requirements.txt."""
        kb_path = Path(__file__).parent / "value_help_requirements.txt"
        vs_path = Path(__file__).parent / "value_help_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Reuse existing vectorstore if KB not updated
        if vs_path.exists():
            kb_mtime = kb_path.stat().st_mtime if kb_path.exists() else 0
            vs_mtime = max((f.stat().st_mtime for f in vs_path.glob("**/*")), default=0)
            if kb_mtime <= vs_mtime:
                self.logger.info("📚 Loading existing FAISS vector DB for ValueHelpAgent...")
                return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
            else:
                self.logger.info("🔄 KB updated — rebuilding FAISS index...")

        if not kb_path.exists():
            self.logger.warning(f"⚠️ No KB file found at {kb_path}. Proceeding without RAG context.")
            return None

        kb_text = kb_path.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(kb_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created for ValueHelpAgent.")
        return vectorstore

    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieve top-k relevant KB chunks."""
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()
        if not self.vectorstore:
            return ""
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        combined = "\n\n".join([r.page_content for r in results])
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks for Value Help.")
        return combined

    # ------------------ MAIN RUN ------------------
    def run(self, field_description: str, metadata=None):
        """
        Generate a RAP Value Help CDS view definition for a standard field.
        """
        if not field_description:
            self.logger.warning("No field description provided to ValueHelpAgent.")
            return {"type": "value_help", "purpose": "", "code": ""}

        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("🚀 Running ValueHelpAgent with provided field description...")

        # --- Retrieve relevant RAG context ---
        rag_context = self._get_relevant_context(field_description)
        full_context = field_description.strip()
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # --- System and user prompts ---
        system_message = SystemMessage(
            content=(
                "You are an SAP ABAP expert specializing in the RAP model. "
                "Generate valid CDS view entities for Value Help (F4 help) with correct annotations, naming, and syntax. "
                "Use retrieved RAG context for reference."
            )
        )

        user_prompt = f"""
        Field requirement:
        {field_description.strip()}

        Reference knowledge (from RAG KB if available):
        {rag_context}

        Output strictly in JSON format with two keys:
        1) "value_help_code": RAP-ready ABAP CDS view entity code for value help.
        2) "value_help_purpose": Short description of the value help CDS view purpose.
        """

        # --- LLM generation ---
        resp = self.llm.invoke([system_message, HumanMessage(content=user_prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        # --- Parse JSON safely ---
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")
            data = json.loads(match.group())
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from LLM output: {e}")
            self.logger.debug(f"Raw LLM output:\n{raw}")
            raise

        value_help_code = data.get("value_help_code", "").strip()
        value_help_purpose = data.get("value_help_purpose", "").strip()
        value_help_entity = data.get("value_help_entity", "").strip()

        # --- Save CDS file ---
        vh_file = self.job_dir / "value_help_cds_view.abap"
        vh_file.write_text(value_help_code, encoding="utf-8")
        self.logger.info(f"💾 Value Help CDS view saved to: {vh_file}")
        self.logger.info(f"📝 Purpose: {value_help_purpose}")

        return {
            "type": "value_help",
            "purpose": value_help_purpose,
            "code": value_help_code,
            "entity": value_help_entity,
            # "path": cds_file
        }

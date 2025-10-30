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

class CdsAgent(BaseAgent):
    """
    Generates RAP-ready CDS view entity definitions using dynamic RAG context from vector database.
    """

    # ------------------ LLM INIT ------------------
    def _init_llm(self):
        """Initialize the LLM client."""
        return ChatOpenAI(
            model_name=os.getenv("CDS_MODEL_NAME", "gpt-4.1-mini"),
            temperature=float(os.getenv("CDS_TEMPERATURE", "0.25")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ------------------ VECTORSTORE INIT ------------------
    def _init_vectorstore(self):
        """
        Loads or builds FAISS vector DB from cds_requirements.txt file.
        """
        kb_path = Path(os.path.dirname(__file__)) / "cds_requirements.txt"
        vs_path = Path(os.path.dirname(__file__)) / "cds_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Rebuild vectorstore if KB updated after FAISS creation
        if vs_path.exists():
            kb_mtime = kb_path.stat().st_mtime if kb_path.exists() else 0
            vs_mtime = max(f.stat().st_mtime for f in vs_path.glob("**/*"))
            if kb_mtime <= vs_mtime:
                self.logger.info("📚 Loading existing FAISS vector DB for CdsAgent...")
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
        self.logger.info("✅ New FAISS vector DB created for CdsAgent.")
        return vectorstore

    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """
        Retrieves top-k relevant KB chunks from vector DB for given query.
        """
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()
        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        combined = "\n\n".join([r.page_content for r in results])
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks.")
        for i, r in enumerate( results, start=1):
            self.logger.info(f"\n--- RAG Chunk {i} ---\n{r.page_content}\n--- END CHUNK {i} ---")
        combined += r.page_content + "\n\n"
        return combined
    
    
    # ------------------ MAIN RUN METHOD ------------------
    def run(self, section_text: str, metadata=None) -> dict:
        if not section_text:
            self.logger.warning("No text provided to CdsAgent.")
            return {"type": "cds", "purpose": "", "path": Path()}

        self.logger.info("Running CdsAgent with provided section text...")
        print("\n🧠 [CdsAgent] Received text:\n", section_text[:500], "\n--- END ---")

        # --- Retrieve relevant RAG context ---
        rag_context = self._get_relevant_context(section_text)
        full_context = section_text.strip()
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # --- System message ---
        system_message = SystemMessage(
            content=(
                "You are a senior SAP ABAP developer specializing in the RESTful ABAP Programming Model (RAP). "
                "Follow SAP RAP naming conventions, annotations, and key field best practices. "
                "Ensure correct RAP syntax and clean formatting. "
                "Do not generate invalid annotations — rely on retrieved RAG context for reference."
            )
        )
        
        value_help_entity = metadata.get("value_help_entity") if metadata else None
        value_help_purpose = metadata.get("value_help_purpose") if metadata else None
        
        # --- Prompt to LLM ---
        prompt = f"""
        Using the following requirement/context, produce JSON with two keys:
        1) "cds_code": RAP-ready ABAP CDS view entity definition code that can be pasted directly into Eclipse.
        2) "cds_purpose": A short explanation of what this CDS view represents and its role in the data model.

        Requirement Context:
        {full_context}
        
        Value Help Metadata:
          - Entity Name: {value_help_entity}
          - Purpose: {value_help_purpose}
        
        Important:
         - If a value help CDS entity is provided, DO NOT use @ObjectModel.valueHelpDefinition or @UI.valueHelp annotations.
         - Instead, create an **association** to that value help CDS entity, using proper cardinality [0..1] and ON condition.
         - Use the format shown in the retrieved RAG under “#8. General instructions for Value help CDS”.

        Output format (strict JSON only, no markdown, no commentary):
        {{
            "cds_code": "...",
            "cds_purpose": "..."
        }}
        """

        # --- Get response ---
        resp = self.llm.invoke([system_message, HumanMessage(content=prompt)])
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

        # --- Extract values ---
        cds_code = data.get("cds_code", "").strip()
        cds_purpose = data.get("cds_purpose", "").strip()

        # --- Save result ---
        cds_file = self.job_dir / "RAP_CDS_View.abap"
        cds_file.write_text(cds_code, encoding="utf-8")

        self.logger.info(f"✅ RAP CDS view saved to: {cds_file}")
        self.logger.debug(f"Extracted CDS Purpose: {cds_purpose[:200]}...")

        return {
            "type": "cds",
            "purpose": cds_purpose,
            "code": cds_code,
            # "path": cds_file,
        }

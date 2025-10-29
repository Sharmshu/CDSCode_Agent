import logging
from pathlib import Path
from openai import OpenAI
from agents.base_agent import BaseAgent
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document


class ValueHelpAgent(BaseAgent):
    """
    Agent that generates RAP-ready Value Help CDS views (F4 helps)
    using LLM and an optional RAG knowledge base.
    """

    def __init__(self, job_dir: Path, logger=None):
        super().__init__(job_dir, logger)
        self.agent_dir = Path(__file__).parent        # <-- Points to /agents/ValueHelp
        self.llm = self._init_llm()
        self.vectorstore = self._init_vectorstore()
        self.logger.info("✅ ValueHelpAgent initialized successfully.")

    # ------------------- LLM INIT -------------------
    def _init_llm(self):
        """Initialize OpenAI client."""
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables.")
        return OpenAI(api_key=api_key)

    # ------------------- RAG CONTEXT -------------------
    def _init_vectorstore(self):
        """
        Load or create FAISS vector store from the Value Help knowledge base.
        """
        rag_file = self.agent_dir / "value_help_requirements.txt"
        vs_path = self.agent_dir / "value_help_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        if vs_path.exists():
            self.logger.info("📚 Loading existing FAISS vector DB for ValueHelpAgent...")
            return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

        if not rag_file.exists():
            self.logger.warning(f"⚠️ No KB file found at {rag_file}. Proceeding without RAG context.")
            return None

        rag_text = rag_file.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(rag_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created for ValueHelpAgent.")
        return vectorstore

    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """Retrieve top-k relevant KB chunks."""
        if not self.vectorstore:
            self.logger.warning("⚠️ No vector store available, skipping RAG context.")
            return ""
        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        combined = "\n\n".join([r.page_content for r in results])
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks for Value Help.")
        return combined

    # ------------------- MAIN RUN -------------------
    def run(self, field_description: str, metadata=None):
        """
        Generate a RAP Value Help CDS view definition for a standard field.
        """
        self.logger.info("🚀 Running ValueHelpAgent with RAG context...")

        # Step 1: Retrieve relevant context
        rag_context = self._get_relevant_context(field_description)
        full_context = field_description.strip()
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # Step 2: Build prompts
        system_prompt = (
            "You are an SAP ABAP expert specializing in the RAP model. "
            "Generate a valid CDS view entity that acts as a Value Help (F4 Help) for a standard field. "
            "Include necessary RAP annotations and make it syntax-correct."
        )

        user_prompt = f"""
        Field requirement:
        {field_description.strip()}

        Reference knowledge (from RAG KB if available):
        {rag_context}

        Please generate:
        - A valid CDS view entity for RAP Value Help (without SQL view name)
        - Follow SAP naming conventions and ensure RAP compatibility.
        """

        # Step 3: Generate CDS code
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            cds_code = response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"❌ LLM generation failed: {e}")
            raise

        # Step 4: Summarize purpose
        purpose_prompt = (
            f"Summarize briefly the purpose of this CDS view:\n\n{cds_code}\n\n"
            "Return one concise sentence (e.g., 'Value help CDS view for Customer field.')."
        )

        purpose_resp = self.llm.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You summarize SAP CDS entities."},
                {"role": "user", "content": purpose_prompt},
            ],
            temperature=0.2,
        )
        purpose = purpose_resp.choices[0].message.content.strip()

        # Step 5: Save result
        self.job_dir.mkdir(parents=True, exist_ok=True)
        cds_file = self.job_dir / "value_help_cds_view.abap"
        cds_file.write_text(cds_code, encoding="utf-8")

        self.logger.info(f"💾 Value Help CDS view saved to: {cds_file}")
        self.logger.info(f"📝 Purpose: {purpose}")

        return {"path": cds_file, "purpose": purpose}

import logging
from pathlib import Path
from openai import OpenAI
from agents.base_agent import BaseAgent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

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
    def _init_vectorstore(self) -> str:
        """
        Load the CDS reference (RAG) file and return content as string.
        """
        rag_file = Path(__file__).parent / "cds_requirements.txt"
        vs_path = Path(__file__).parent / "cds_vector_store"
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
        if vs_path.exists():
            self.logger.info("📚 Loading existing FAISS vector DB for CDSAgent...")
            return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
        
        if not rag_file.exists():
           self.logger.warning(f"⚠️ No KB file found at {rag_file}. Proceeding without RAG context.")
           return None
    
        rag_text = rag_file.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(rag_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created for TableAgent.")
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
        return combined
    # ------------------- MAIN RUN -------------------
    def run(self, section_text: str, metadata=None):
        """ 
        Generate a RAP-ready CDS view entity definition using requirement and RAG context.
        """
        self.logger.info("Running RAP CDS agent with RAG file context...")

        # Step 1: Load RAG file
         # --- Retrieve relevant RAG context ---
        rag_context = self._get_relevant_context(section_text)
        full_context = section_text.strip()
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

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

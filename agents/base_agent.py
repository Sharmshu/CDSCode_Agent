from pathlib import Path
import logging

class BaseAgent:
    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.llm = self._init_llm()

    def _init_llm(self):
        """
        To be overridden by each subclass to initialize its own LLM.
        """
        raise NotImplementedError("Each agent must implement its own LLM initialization method.")

    def run(self, section_text: str, metadata=None):
        raise NotImplementedError("Each agent must implement its own run()")
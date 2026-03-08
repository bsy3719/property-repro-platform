from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


class OpenAIService:
    def __init__(self, model_name: str | None = None) -> None:
        load_dotenv(DOTENV_PATH)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name or os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")

    def paper_to_markdown(self, raw_text: str) -> str:
        trimmed = raw_text[:120000]
        prompt = (
            "You are a scientific paper parsing agent for property prediction ML papers.\n"
            "Task:\n"
            "1) Read the paper text.\n"
            "2) Remove non-essential sections like references, acknowledgements, and appendix details.\n"
            "3) Produce concise Markdown with these sections exactly:\n"
            "# Paper Summary\n"
            "## Dataset\n"
            "## Preprocessing\n"
            "## Feature Engineering\n"
            "## Model\n"
            "## Hyperparameters\n"
            "## Training Strategy\n"
            "## Metrics\n"
            "## Reported Best Result (LFL)\n"
            "4) If a section is missing, write 'Not found in paper'.\n"
            "5) Focus on LFL-related experiment only, and best-performing model if multiple models exist.\n"
            "6) Return markdown only.\n\n"
            "Paper text:\n"
            f"{trimmed}"
        )

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=0.1,
        )
        return (response.output_text or "").strip()


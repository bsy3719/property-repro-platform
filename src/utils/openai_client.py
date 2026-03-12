from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_CHAT_MODEL = "gpt-5.2"


def create_openai_client(model_name: str | None = None) -> tuple[OpenAI, str]:
    load_dotenv(DOTENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    resolved_model = model_name or os.getenv("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL
    return OpenAI(api_key=api_key), resolved_model


def run_text_response(client: OpenAI, model_name: str, prompt: str, context: str) -> str:
    try:
        response = client.responses.create(model=model_name, input=prompt)
    except Exception as exc:
        raise RuntimeError(f"{context}: OpenAI 호출 실패: {exc}") from exc

    output_text = (response.output_text or "").strip()
    if not output_text:
        raise RuntimeError(f"{context}: OpenAI가 빈 응답을 반환했습니다.")
    return output_text

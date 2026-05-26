import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class JudgeResponse:
    content: str
    success: bool = True


class OpenAIJudge:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        timeout: int = 120,
        max_retries: int = 3,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        if not model:
            raise ValueError("judge model is empty")

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key or os.getenv("OPENAI_API_KEY", "EMPTY"),
            "timeout": timeout,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort

    @classmethod
    def from_env(cls, default_model: Optional[str] = None) -> "OpenAIJudge":
        model = os.getenv("OMTG_JUDGE_MODEL") or os.getenv("RM_NAME") or default_model
        if not model:
            raise ValueError("set OMTG_JUDGE_MODEL or RM_NAME for caption reward")

        return cls(
            model=model,
            api_key=os.getenv("OMTG_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OMTG_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            temperature=float(os.getenv("OMTG_JUDGE_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("OMTG_JUDGE_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("OMTG_JUDGE_TIMEOUT", "120")),
            max_retries=int(os.getenv("OMTG_JUDGE_MAX_RETRIES", "3")),
            reasoning_effort=os.getenv("OMTG_JUDGE_REASONING_EFFORT"),
        )

    def chat(self, messages: List[dict]) -> JudgeResponse:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort

                completion = self.client.chat.completions.create(**kwargs)
                message = completion.choices[0].message
                content = message.content or getattr(message, "reasoning_content", "") or ""
                return JudgeResponse(content=content, success=True)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"judge call failed: {last_error}") from last_error

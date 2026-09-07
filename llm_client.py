"""LLM Client module connecting to 9router proxy on port 20128."""

import json
import logging
import requests
from typing import Dict, Any, Optional
from config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = (base_url or settings.ROUTER_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.ROUTER_API_KEY
        self.model = model or settings.LLM_MODEL

    def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices returned by LLM endpoint")
            content = choices[0].get("message", {}).get("content", "")
            return content.strip()
        except Exception as e:
            logger.error(f"Error communicating with 9router LLM: {e}")
            raise


llm_client = LLMClient()

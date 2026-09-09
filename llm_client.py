"""LLM Client module connecting to 9router proxy on port 20128.

Supports:
- Chat completions (text/system prompt)
- Native Direct-Video completions (verified via 9router + Gemini 3.8 Flash)
- Robust parsing for both direct JSON and SSE stream chunks
"""

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import requests

from config import settings

logger = logging.getLogger(__name__)

# Mandatory Direct-Video Verification Flag (Blueprint Section 10)
# Verified on 2026-09-08: Gemini 3.8 Flash via 9router correctly comprehended ordered temporal visual states
DIRECT_VIDEO_VERIFIED: bool = True


class LLMClient:
    """Production client for Gemini 3.8 Flash multimodal decisions via 9router."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 240,
    ):
        self.base_url = (base_url or getattr(settings, "ROUTER_BASE_URL", "http://127.0.0.1:20128/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "ROUTER_API_KEY", "")
        self.model = model or getattr(settings, "LLM_MODEL", "ag/gemini-3.8-flash-high")
        self.timeout_sec = timeout_sec

    def _parse_response_text(self, resp: requests.Response) -> str:
        """Parses OpenAI-compatible response body handling direct JSON and SSE streams."""
        text_body = getattr(resp, "text", "") or ""
        if text_body.strip().startswith("data:"):
            parts = []
            for line in text_body.strip().split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    chunk_str = line[5:].strip()
                    if chunk_str and chunk_str != "[DONE]":
                        try:
                            c = json.loads(chunk_str)
                            delta = c.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                parts.append(delta["content"])
                        except Exception:
                            pass
            return "".join(parts).strip()

        try:
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return ""
            return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            return text_body.strip()

    def chat_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> str:
        """Executes text-based chat completion via 9router."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
            resp.raise_for_status()
            content = self._parse_response_text(resp)
            if not content:
                raise ValueError("No content returned by 9router LLM")
            return content
        except Exception as e:
            logger.error(f"Error in chat_completion via 9router: {e}")
            raise

    def video_completion(
        self,
        video_path: Union[str, Path],
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        timeout_sec: Optional[int] = None,
    ) -> str:
        """Executes native multimodal video completion via 9router + Gemini 3.8 Flash.

        Encodes video as base64 and attaches it as an image_url data URI payload.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        video_bytes = video_path.read_bytes()
        b64_video = base64.b64encode(video_bytes).decode("utf-8")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{b64_video}"}},
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        # Ensure video completion timeout is at least 240 seconds
        effective_timeout = max(timeout_sec if timeout_sec is not None else self.timeout_sec, 240)
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=effective_timeout)
            resp.raise_for_status()
            content = self._parse_response_text(resp)
            if not content:
                raise ValueError("No content returned by 9router video completion")
            return content
        except Exception as e:
            logger.error(f"Error in video_completion via 9router: {e}")
            raise

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON dictionary from LLM response text."""
        if not text:
            return None

        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None


llm_client = LLMClient()

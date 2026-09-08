"""Gemini Search Planner module (Auto Clipper V3.1 Stage A).

Consults Gemini 3.8 Flash via 9router at the start of each search cycle
to generate diverse, targeted Indonesian search queries, taking into account
recent pipeline rejections/failures to avoid repetitive searches.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from llm_client import llm_client

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_QUERIES = [
    "podcast bisnis indonesia",
    "podcast inspiratif indonesia",
    "interview tokoh indonesia",
    "curhat bang denny sumargo podcast",
    "raditya dika podcast",
    "bicara bicara podcast indonesia",
]


class SearchPlan(BaseModel):
    """Structured plan returned by the Gemini Search Planner."""
    queries: List[str] = Field(default_factory=list, description="List of search queries to execute")
    preferred_categories: List[str] = Field(
        default_factory=lambda: ["podcast", "interview", "talking-head"],
        description="Target content formats"
    )
    avoid: List[str] = Field(
        default_factory=lambda: ["music video", "gameplay", "shorts", "trailer"],
        description="Categories to filter out"
    )
    reasoning: Optional[str] = None


class GeminiSearchPlanner:
    """Plans video search campaigns via Gemini 3.8 Flash."""

    def __init__(self, client=None):
        self.client = client or llm_client

    def plan_searches(
        self,
        recent_failures_summary: Optional[List[Dict[str, Any]]] = None,
        max_queries: int = 5,
    ) -> SearchPlan:
        """Generates a search plan incorporating recent failure feedback."""
        system_prompt = (
            "Kamu adalah Content Scout & Acquisition Lead untuk YouTube Shorts otomatis.\n"
            "Tugasmu adalah merancang 3-5 query pencarian YouTube untuk menemukan video long-form "
            "berbahasa Indonesia (podcast, wawancara, vlog cerita, commentary, talking-head) yang "
            "kaya akan kutipan menarik, insight bernilai, atau momen lucu/emosional.\n"
            "Format jawaban HANYA valid JSON sesuai skema tanpa markdown tambahan."
        )

        failures_context = ""
        if recent_failures_summary:
            failures_context = (
                f"\n\nPerhatian: Beberapa kandidat baru-baru ini DITOLAK karena alasan berikut:\n"
                f"{json.dumps(recent_failures_summary[:5], indent=2)}\n"
                f"Sesuaikan query pencarian untuk menghindari pola video yang serupa!"
            )

        prompt = (
            f"Hasilkan rencana pencarian konten video YouTube Bahasa Indonesia:{failures_context}\n\n"
            f"Kembalikan JSON dengan format:\n"
            f"{{\n"
            f'  "queries": ["query 1", "query 2", "query 3"],\n'
            f'  "preferred_categories": ["podcast", "interview"],\n'
            f'  "avoid": ["music video", "gameplay", "full movie"],\n'
            f'  "reasoning": "Alasan pemilihan query ini"\n'
            f"}}"
        )

        try:
            raw = self.client.chat_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.4,
            )
            parsed = self.client.extract_json(raw)
            if parsed and isinstance(parsed.get("queries"), list) and len(parsed["queries"]) > 0:
                queries = [str(q).strip() for q in parsed["queries"] if str(q).strip()][:max_queries]
                return SearchPlan(
                    queries=queries,
                    preferred_categories=parsed.get("preferred_categories", ["podcast", "interview"]),
                    avoid=parsed.get("avoid", ["music video", "gameplay"]),
                    reasoning=parsed.get("reasoning", "Gemini generated search plan"),
                )
        except Exception as e:
            logger.warning(f"GeminiSearchPlanner invocation failed ({e}), using default search queries")

        return SearchPlan(
            queries=DEFAULT_SEARCH_QUERIES[:max_queries],
            reasoning="Default fallback queries due to offline/unresponsive LLM",
        )

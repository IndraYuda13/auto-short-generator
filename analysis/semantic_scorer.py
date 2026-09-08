"""Semantic Scorer module for Auto Short Generator Phase A.

Evaluates candidate transcript windows via Gemini 3.8 Flash through 9router proxy.
Applies rigorous viral and narrative completeness criteria:
- hook_score (0-100)
- payoff_score (0-100)
- self_contained_score (0-100) -> Answers: "Does this segment make sense without the rest of the video?"
- overall_score (0-100)
- good_clip (bool)
- suggested_start & suggested_end
Returns 'NO GOOD CLIP FOUND' when no candidate meets quality criteria.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import requests
from pydantic import BaseModel, Field

from config import settings
from analysis.candidate_generator import CandidateWindow

logger = logging.getLogger(__name__)


class SemanticScore(BaseModel):
    """Evaluation metrics for a candidate window."""
    candidate_id: str
    good_clip: bool
    score: float = Field(..., ge=0.0, le=100.0, description="Overall viral potential score")
    hook_score: float = Field(..., ge=0.0, le=100.0, description="Opening 3s retention/hook score")
    payoff_score: float = Field(..., ge=0.0, le=100.0, description="Resolution/punchline/payoff score")
    self_contained_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Self-contained score: Does this make sense without the rest of the video?"
    )
    reason: str
    suggested_start: float
    suggested_end: float


class SemanticScorer:
    """Evaluates short-form candidates using Gemini 3.8 Flash via 9router."""

    MIN_OVERALL_SCORE: float = 70.0
    MIN_SELF_CONTAINED_SCORE: float = 70.0
    MIN_HOOK_SCORE: float = 65.0

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout_sec: int = 45,
    ):
        self.base_url = (base_url or getattr(settings, "ROUTER_BASE_URL", "http://127.0.0.1:20128/v1")).rstrip("/")
        self.api_key = api_key or getattr(settings, "ROUTER_API_KEY", "")
        self.model = model or getattr(settings, "LLM_MODEL", "ag/gemini-3.8-flash-high")
        self.timeout_sec = timeout_sec

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Invokes 9router chat completions endpoint."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "stream": False,
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_sec)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No completion choices returned by 9router")
        return choices[0].get("message", {}).get("content", "").strip()

    def score_candidate(
        self,
        candidate: CandidateWindow,
        video_title: str = ""
    ) -> SemanticScore:
        """
        Evaluates a single candidate window against hook, payoff, and self-contained questions.
        """
        system_prompt = (
            "Kamu adalah Senior Content Editor & Viral Algorithm Specialist untuk YouTube Shorts dan TikTok. "
            "Kamu mengevaluasi apakah segmen video ini layak menjadi short viral yang mandiri dan memuaskan. "
            "Pertanyaan paling krusial yang WAJIB kamu jawab dengan jujur dan ketat adalah:\n"
            "'Does this segment make sense without the rest of the video?' (Apakah segmen ini punya konteks utuh yang bisa dipahami orang yang belum pernah nonton video aslinya?)"
        )

        prompt = f"""Judul Video Sumber: "{video_title}"
ID Kandidat: {candidate.candidate_id}
Rentang Waktu: {candidate.start_sec:.2f}s - {candidate.end_sec:.2f}s (Durasi: {candidate.duration:.2f}s)

Transkrip Kandidat:
\"\"\"{candidate.text}\"\"\"

Kriteria Penilaian Ketat (Skala 0 - 100):
1. `hook_score`: Seberapa kuat 3-5 detik pembuka memicu rasa penasaran, konflik, opini kontroversial, atau kelucuan?
2. `payoff_score`: Seberapa memuaskan kesimpulan, punchline, jawaban, atau klimaks emosi di akhir segmen?
3. `self_contained_score`: WAJIB evaluasi: 'Does this segment make sense without the rest of the video?'
   - Jika segmen terpotong di tengah argumen, butuh info sebelum/sesudahnya untuk paham, atau ambigu -> nilai < 60.
   - Jika segmen adalah cerita/argumen utuh dari premis sampai kesimpulan -> nilai >= 75.
4. `score` (Overall): Rata-rata terbobot potensi viral segmen ini (0-100).
5. `good_clip`: WAJIB boolean (true/false). HANYA bernilai true jika overall score >= 70, self_contained_score >= 70, dan hook_score >= 65.
6. `reason`: Alasan ringkas dan jelas dalam Bahasa Indonesia.
7. `suggested_start`: Detik start yang direkomendasikan (float, dalam rentang {candidate.start_sec:.2f} - {candidate.end_sec:.2f}).
8. `suggested_end`: Detik end yang direkomendasikan (float, dalam rentang {candidate.start_sec:.2f} - {candidate.end_sec:.2f}).

Keluarkan HANYA valid JSON object tanpa markdown penjelasan lain:
{{
  "candidate_id": "{candidate.candidate_id}",
  "good_clip": true,
  "score": 85.0,
  "hook_score": 88.0,
  "payoff_score": 82.0,
  "self_contained_score": 86.0,
  "reason": "Cerita unscripted tentang kegagalan bisnis pertama yang memancing penasaran di awal dan punya pesan moral jelas di akhir.",
  "suggested_start": {candidate.start_sec:.2f},
  "suggested_end": {candidate.end_sec:.2f}
}}"""

        try:
            raw_text = self._call_llm(prompt, system_prompt)
            data = self._extract_json(raw_text)

            score = float(data.get("score", 50.0))
            hook_score = float(data.get("hook_score", 50.0))
            payoff_score = float(data.get("payoff_score", 50.0))
            self_contained_score = float(data.get("self_contained_score", 50.0))

            # Strictly enforce good_clip conditions
            good_clip_raw = bool(data.get("good_clip", False))
            good_clip = (
                good_clip_raw
                and score >= self.MIN_OVERALL_SCORE
                and self_contained_score >= self.MIN_SELF_CONTAINED_SCORE
                and hook_score >= self.MIN_HOOK_SCORE
            )

            s_start = float(data.get("suggested_start", candidate.start_sec))
            s_end = float(data.get("suggested_end", candidate.end_sec))

            # Clamp suggested times within candidate window
            s_start = max(candidate.start_sec, min(candidate.end_sec - 10.0, s_start))
            s_end = min(candidate.end_sec, max(s_start + 10.0, s_end))

            return SemanticScore(
                candidate_id=candidate.candidate_id,
                good_clip=good_clip,
                score=score,
                hook_score=hook_score,
                payoff_score=payoff_score,
                self_contained_score=self_contained_score,
                reason=str(data.get("reason", "")),
                suggested_start=round(s_start, 2),
                suggested_end=round(s_end, 2)
            )
        except Exception as e:
            logger.error(f"Semantic scoring error for {candidate.candidate_id}: {e}")
            return SemanticScore(
                candidate_id=candidate.candidate_id,
                good_clip=False,
                score=0.0,
                hook_score=0.0,
                payoff_score=0.0,
                self_contained_score=0.0,
                reason=f"Scoring error: {e}",
                suggested_start=candidate.start_sec,
                suggested_end=candidate.end_sec
            )

    def score_candidates(
        self,
        candidates: List[CandidateWindow],
        video_title: str = ""
    ) -> List[SemanticScore]:
        """Scores a list of candidate windows sequentially."""
        results: List[SemanticScore] = []
        for cand in candidates:
            score = self.score_candidate(cand, video_title=video_title)
            results.append(score)
        return results

    def select_best_clip(
        self,
        candidates: List[CandidateWindow],
        scores: List[SemanticScore]
    ) -> Tuple[Optional[CandidateWindow], Optional[SemanticScore], str]:
        """
        Selects the top candidate that meets good_clip criteria.
        Returns (None, None, 'NO GOOD CLIP FOUND') if no candidate is deemed good.
        """
        score_map = {s.candidate_id: s for s in scores}
        valid_pairs: List[Tuple[CandidateWindow, SemanticScore]] = []

        for cand in candidates:
            sc = score_map.get(cand.candidate_id)
            if sc and sc.good_clip:
                valid_pairs.append((cand, sc))

        if not valid_pairs:
            logger.warning("No candidate satisfied good_clip quality thresholds.")
            return None, None, "NO GOOD CLIP FOUND"

        # Sort by overall score descending, then self_contained_score descending
        valid_pairs.sort(key=lambda pair: (pair[1].score, pair[1].self_contained_score), reverse=True)
        best_cand, best_score = valid_pairs[0]
        return best_cand, best_score, "GOOD_CLIP_SELECTED"

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """Extracts JSON dictionary from response text."""
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except Exception:
            # Match bracketed JSON object
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Unable to parse JSON from LLM output: {text[:200]}")

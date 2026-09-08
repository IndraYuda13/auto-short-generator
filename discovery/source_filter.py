"""Discovery Source Filter module for Auto Short Generator Phase A (Blueprint Stage 1).

Applies early validation criteria to video metadata without downloading full video media:
1. Duration constraints: 180s (3m) <= duration_sec <= 14400s (4h).
2. Rejection of live streams and upcoming premieres/broadcasts.
3. Rejection of private or restricted videos.
4. Duplicate rejection (against local SQLite database or in-memory processed set).
5. Speech presence validation (filtering out music videos, full movies, no-commentary gameplay, ASMR, etc.).
"""

import logging
import re
from typing import List, Optional, Set, Any
from pydantic import BaseModel, Field, model_validator

from db import db

logger = logging.getLogger(__name__)

# Patterns indicating video unlikely to have conversational spoken speech
NON_SPEECH_PATTERNS = [
    # Music & Audio
    r"\bofficial\s+music\s+video\b",
    r"\bofficial\s+(?:audio|mv|video)\b",
    r"\blyric(?:s)?\s+video\b",
    r"\bfull\s+album\b",
    r"\blofi\s+(?:hip\s*hop|beats|music)?\b",
    r"\binstrumental\b",
    r"\bkaraoke\b",
    r"\bmeditation\s+music\b",
    r"\bsleep\s+music\b",
    r"\bbgm\b",
    r"\bost\b",
    r"\baudio\s+only\b",
    r"\bcompilation\s+lagu\b",
    r"\bplaylist\s+lagu\b",
    r"\blagu\s+(?:terbaru|viral|galau|enak)\b",
    r"\bdj\s+remix\b",
    r"\bacoustic\s+cover\b",
    r"\bcover\s+lagu\b",
    # Full Movies
    r"\bfull\s+movie\b",
    r"\bfilm\s+bioskop\s+full\b",
    r"\bfilm\s+full\s+movie\b",
    r"\bstreaming\s+film\b",
    # Gameplay without commentary
    r"\bno\s+commentary\b",
    r"\bgameplay\s+(?:walkthrough\s+)?no\s+commentary\b",
    r"\bno\s+speech\b",
    r"\bno\s+talking\b",
    r"\bwalkthrough\s+no\s+voice\b",
    # ASMR & Ambient sounds
    r"\basmr\b",
    r"\bmukbang\s+asmr\b",
    r"\bambience\b",
    r"\bwhite\s+noise\b",
    r"\brelaxing\s+sound(?:s)?\b",
]


class EligibilityResult(BaseModel):
    """Result of source eligibility evaluation."""
    is_eligible: bool = True
    reason: str = "Video satisfies Phase A discovery requirements"
    accepted: bool = True
    rejection_code: Optional[str] = None
    video_id: str = ""
    title: str = ""
    duration_sec: float = 0.0
    duration: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def sync_verdict_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "is_eligible" in data and "accepted" not in data:
                data["accepted"] = bool(data["is_eligible"])
            elif "accepted" in data and "is_eligible" not in data:
                data["is_eligible"] = bool(data["accepted"])

            dur = data.get("duration")
            dur_sec = data.get("duration_sec")
            if dur is not None and dur_sec is None:
                data["duration_sec"] = float(dur)
            elif dur_sec is not None and dur is None:
                data["duration"] = float(dur_sec)
            elif dur is not None and dur_sec is not None:
                if float(dur_sec) != 0.0 and float(dur) == 0.0:
                    data["duration"] = float(dur_sec)
                elif float(dur) != 0.0 and float(dur_sec) == 0.0:
                    data["duration_sec"] = float(dur)
        return data


# Backward-compatible alias for existing test suites and consumers
SourceFilterVerdict = EligibilityResult


class SourceFilter:
    """Early stage metadata filter for candidate YouTube videos."""

    MIN_DURATION_SEC: float = 180.0    # 3 minutes minimum for long-form source
    MAX_DURATION_SEC: float = 14400.0  # 4 hours maximum (14,400s)

    def __init__(
        self,
        min_duration_sec: float = 180.0,
        max_duration_sec: float = 14400.0,
        processed_video_ids: Optional[Set[str]] = None,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.processed_video_ids: Set[str] = set(processed_video_ids) if processed_video_ids else set()

    def filter_video(self, metadata: Any) -> EligibilityResult:
        """
        Evaluate a single video's metadata against all Phase A source filters.
        Returns is_eligible=True (and accepted=True) only if the video passes all checks.
        """
        vid = getattr(metadata, "video_id", "")
        title = getattr(metadata, "title", "").strip()
        dur = float(getattr(metadata, "duration_sec", 0.0) or getattr(metadata, "duration", 0.0))
        is_live = bool(getattr(metadata, "is_live", False))
        is_upcoming = bool(getattr(metadata, "is_upcoming", False))
        is_private = bool(getattr(metadata, "is_private", False))
        desc = getattr(metadata, "description", "")

        # 1. Live stream / Upcoming / Private check
        if is_live:
            return EligibilityResult(
                is_eligible=False,
                accepted=False,
                reason="Live streams cannot be clipped reliably until finished",
                rejection_code="IS_LIVE",
                video_id=vid,
                title=title,
                duration_sec=dur,
                duration=dur,
            )

        if is_upcoming:
            return EligibilityResult(
                is_eligible=False,
                accepted=False,
                reason="Upcoming premiere or scheduled livestream cannot be clipped",
                rejection_code="IS_UPCOMING",
                video_id=vid,
                title=title,
                duration_sec=dur,
                duration=dur,
            )

        if is_private:
            return EligibilityResult(
                is_eligible=False,
                accepted=False,
                reason="Video is private or restricted",
                rejection_code="IS_PRIVATE",
                video_id=vid,
                title=title,
                duration_sec=dur,
                duration=dur,
            )

        # 2. Duration filter (must be >= 180s and <= 14400s)
        if dur < self.min_duration_sec:
            return EligibilityResult(
                is_eligible=False,
                accepted=False,
                reason=f"Video duration ({dur:.1f}s) is below minimum {self.min_duration_sec:.0f}s",
                rejection_code="DURATION_TOO_SHORT",
                video_id=vid,
                title=title,
                duration_sec=dur,
                duration=dur,
            )

        if dur > self.max_duration_sec:
            return EligibilityResult(
                is_eligible=False,
                accepted=False,
                reason=f"Video duration ({dur:.1f}s) exceeds maximum {self.max_duration_sec:.0f}s (4 hours)",
                rejection_code="DURATION_TOO_LONG",
                video_id=vid,
                title=title,
                duration_sec=dur,
                duration=dur,
            )

        # 3. Duplicate check (DB or provided in-memory set)
        if vid and vid in self.processed_video_ids:
            return EligibilityResult(
                is_eligible=False,
                accepted=False,
                reason=f"Video ID {vid} has already been processed in current session",
                rejection_code="DUPLICATE_VIDEO",
                video_id=vid,
                title=title,
                duration_sec=dur,
                duration=dur,
            )

        try:
            if vid and hasattr(db, "is_video_processed") and db.is_video_processed(vid):
                return EligibilityResult(
                    is_eligible=False,
                    accepted=False,
                    reason=f"Video ID {vid} is already marked processed in database",
                    rejection_code="DUPLICATE_VIDEO",
                    video_id=vid,
                    title=title,
                    duration_sec=dur,
                    duration=dur,
                )
        except Exception as e:
            logger.warning(f"Failed to check duplicate status from DB for {vid}: {e}")

        # 4. Speech presence check from title & description
        combined_text = f"{title} {desc}".lower()
        for pat in NON_SPEECH_PATTERNS:
            if re.search(pat, combined_text, re.IGNORECASE):
                return EligibilityResult(
                    is_eligible=False,
                    accepted=False,
                    reason=f"Detected non-speech pattern '{pat}' in video title/description",
                    rejection_code="NO_CLEAR_SPEECH",
                    video_id=vid,
                    title=title,
                    duration_sec=dur,
                    duration=dur,
                )

        # All checks passed
        return EligibilityResult(
            is_eligible=True,
            accepted=True,
            reason="Video satisfies Phase A discovery requirements",
            rejection_code=None,
            video_id=vid,
            title=title,
            duration_sec=dur,
            duration=dur,
        )

    def filter_batch(self, videos: List[Any]) -> List[EligibilityResult]:
        """Filters a list of video candidates and returns verdicts, tracking accepted IDs."""
        verdicts: List[EligibilityResult] = []
        for v in videos:
            verdict = self.filter_video(v)
            verdicts.append(verdict)
            if verdict.accepted or verdict.is_eligible:
                vid = getattr(v, "video_id", "")
                if vid:
                    self.processed_video_ids.add(vid)
        return verdicts

    def is_eligible(self, metadata: Any) -> bool:
        """Boolean convenience check for video candidate eligibility."""
        return self.filter_video(metadata).is_eligible

"""Boundary Refiner module for Auto Short Generator Phase A (Stage 7).

Refines candidate start and end timestamps:
1. Snaps start boundary before hook, avoiding mid-word cuts and avoiding start jitter near scene cuts.
2. Snaps end boundary after payoff, after sentence complete, adding natural breathing room (0.1-0.3s)
   without truncating reactions or laughter.
3. Strictly enforces final clip duration: target 30–55 seconds.
4. Rejects candidate if natural boundary within 30–55 seconds cannot be found:
   RefinementResult(is_valid: bool, refined_start: float, refined_end: float, duration: float, rejection_reason: Optional[str])
"""

import logging
from typing import List, Optional, Any
from pydantic import BaseModel, model_validator

from transcription.transcript_provider import TranscriptSegment
from transcription.whisper_aligner import WordToken

logger = logging.getLogger(__name__)


class RefinementResult(BaseModel):
    """Result of boundary refinement and duration validation."""
    is_valid: bool
    refined_start: float
    refined_end: float
    duration: float
    rejection_reason: Optional[str] = None
    snapped_to_scene_cut: bool = False
    laughter_buffer_added: float = 0.0

    @property
    def accepted(self) -> bool:
        """Backward compatibility alias for is_valid."""
        return self.is_valid

    @property
    def start_sec(self) -> float:
        """Backward compatibility alias for refined_start."""
        return self.refined_start

    @property
    def end_sec(self) -> float:
        """Backward compatibility alias for refined_end."""
        return self.refined_end

    @property
    def reason(self) -> str:
        """Backward compatibility alias for rejection_reason."""
        return self.rejection_reason or ""

    @model_validator(mode="before")
    @classmethod
    def _handle_compatibility(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "accepted" in data and "is_valid" not in data:
                data["is_valid"] = bool(data["accepted"])
            elif "is_valid" in data and "accepted" not in data:
                data["accepted"] = bool(data["is_valid"])

            if "start_sec" in data and "refined_start" not in data:
                data["refined_start"] = float(data["start_sec"])
            elif "refined_start" in data and "start_sec" not in data:
                data["start_sec"] = float(data["refined_start"])

            if "end_sec" in data and "refined_end" not in data:
                data["refined_end"] = float(data["end_sec"])
            elif "refined_end" in data and "end_sec" not in data:
                data["end_sec"] = float(data["refined_end"])

            if "reason" in data and "rejection_reason" not in data:
                data["rejection_reason"] = data["reason"]
            elif "rejection_reason" in data and "reason" not in data:
                data["reason"] = data["rejection_reason"]
        return data


# Backward-compatible alias
RefinedBoundaryResult = RefinementResult


class BoundaryRefiner:
    """Snaps clip boundaries to natural speech & scene cuts, enforcing 30–55s duration."""

    TARGET_MIN_DURATION: float = 30.0
    TARGET_MAX_DURATION: float = 55.0

    def __init__(
        self,
        min_duration_sec: float = 30.0,
        max_duration_sec: float = 55.0,
        laughter_buffer_sec: float = 0.2,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.laughter_buffer_sec = laughter_buffer_sec

    def refine(
        self,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        segments: Optional[List[TranscriptSegment]] = None,
        initial_start: Optional[float] = None,
        initial_end: Optional[float] = None,
        phrase_segments: Optional[List[TranscriptSegment]] = None,
        word_tokens: Optional[List[WordToken]] = None,
        scene_cuts: Optional[List[float]] = None,
    ) -> RefinementResult:
        """Convenience alias for refine_boundaries matching Orchestrator contract."""
        s = start_sec if start_sec is not None else (initial_start if initial_start is not None else 0.0)
        e = end_sec if end_sec is not None else (initial_end if initial_end is not None else 0.0)
        segs = segments if segments is not None else (phrase_segments or [])
        return self.refine_boundaries(
            initial_start=float(s),
            initial_end=float(e),
            phrase_segments=segs,
            word_tokens=word_tokens,
            scene_cuts=scene_cuts,
        )

    def refine_boundaries(
        self,
        initial_start: float,
        initial_end: float,
        phrase_segments: List[TranscriptSegment],
        word_tokens: Optional[List[WordToken]] = None,
        scene_cuts: Optional[List[float]] = None,
    ) -> RefinementResult:
        """
        Performs speech-boundary snapping, scene cut coordination,
        post-payoff laughter/reaction preservation, and strict 30-55s duration validation.
        """
        if initial_end <= initial_start:
            return RefinementResult(
                is_valid=False,
                refined_start=initial_start,
                refined_end=initial_end,
                duration=0.0,
                rejection_reason="Invalid boundary timestamps: end <= start",
            )

        if not phrase_segments and not word_tokens:
            return RefinementResult(
                is_valid=False,
                refined_start=initial_start,
                refined_end=initial_end,
                duration=round(initial_end - initial_start, 2),
                rejection_reason="No phrase segments or word tokens provided for boundary refinement",
            )

        refined_start = initial_start
        refined_end = initial_end
        snapped_cut = False

        # 1. Snap Start to Speech Boundary (Word or Phrase)
        if word_tokens:
            # Avoid cutting mid-word
            mid_word = next((w for w in word_tokens if w.start < initial_start < w.end), None)
            if mid_word:
                refined_start = max(0.0, mid_word.start - 0.05)
            else:
                matching_words = [w for w in word_tokens if w.start >= (initial_start - 1.5)]
                if matching_words:
                    first_word = matching_words[0]
                    refined_start = max(0.0, first_word.start - 0.05)
        elif phrase_segments:
            # Snap to closest phrase start
            matching_phrases = [p for p in phrase_segments if p.start >= (initial_start - 2.0)]
            if matching_phrases:
                refined_start = matching_phrases[0].start

        # Check scene cuts near start: avoid starting 100-300ms before a scene transition
        if scene_cuts:
            for sc in scene_cuts:
                if abs(sc - refined_start) <= 0.40:
                    refined_start = sc
                    snapped_cut = True
                    break

        # 2. Snap End to Speech Boundary & Preserve Laughter/Reaction
        buffer_added = 0.0
        if word_tokens:
            mid_end_word = next((w for w in word_tokens if w.start < initial_end < w.end), None)
            if mid_end_word:
                refined_end = mid_end_word.end + self.laughter_buffer_sec
                buffer_added = self.laughter_buffer_sec
            else:
                matching_end_words = [w for w in word_tokens if w.end <= (initial_end + 2.0)]
                if matching_end_words:
                    last_word = matching_end_words[-1]
                    refined_end = last_word.end + self.laughter_buffer_sec
                    buffer_added = self.laughter_buffer_sec
        elif phrase_segments:
            matching_end_phrases = [p for p in phrase_segments if p.end <= (initial_end + 3.0)]
            if matching_end_phrases:
                last_phrase = matching_end_phrases[-1]
                refined_end = last_phrase.end + self.laughter_buffer_sec
                buffer_added = self.laughter_buffer_sec

        refined_duration = refined_end - refined_start

        # 3. Micro-Adjustment to Fit [30.0, 55.0]s Target if Possible
        if refined_duration < self.min_duration_sec:
            # Try extending end to next phrase/word if available
            shortfall = self.min_duration_sec - refined_duration
            if phrase_segments:
                later_phrases = [p for p in phrase_segments if p.end > refined_end]
                for p in later_phrases:
                    if (p.end - refined_start) <= self.max_duration_sec:
                        refined_end = p.end + self.laughter_buffer_sec
                        refined_duration = refined_end - refined_start
                        if refined_duration >= self.min_duration_sec:
                            break

        elif refined_duration > self.max_duration_sec:
            # Try trimming to previous phrase end if available
            excess = refined_duration - self.max_duration_sec
            if phrase_segments:
                earlier_phrases = [
                    p for p in phrase_segments
                    if p.end < refined_end and (p.end - refined_start) >= self.min_duration_sec
                ]
                if earlier_phrases:
                    refined_end = earlier_phrases[-1].end + self.laughter_buffer_sec
                    refined_duration = refined_end - refined_start

        # Round to 2 decimals
        refined_start = round(refined_start, 2)
        refined_end = round(refined_end, 2)
        final_duration = round(refined_end - refined_start, 2)

        # 4. Strict Duration Check [30.0, 55.0]s
        if final_duration < self.min_duration_sec:
            return RefinementResult(
                is_valid=False,
                refined_start=refined_start,
                refined_end=refined_end,
                duration=final_duration,
                rejection_reason=(
                    f"Refined duration ({final_duration:.2f}s) is below minimum "
                    f"target {self.min_duration_sec:.1f}s"
                ),
                snapped_to_scene_cut=snapped_cut,
                laughter_buffer_added=buffer_added,
            )

        if final_duration > self.max_duration_sec:
            return RefinementResult(
                is_valid=False,
                refined_start=refined_start,
                refined_end=refined_end,
                duration=final_duration,
                rejection_reason=(
                    f"Refined duration ({final_duration:.2f}s) exceeds maximum "
                    f"target {self.max_duration_sec:.1f}s"
                ),
                snapped_to_scene_cut=snapped_cut,
                laughter_buffer_added=buffer_added,
            )

        return RefinementResult(
            is_valid=True,
            refined_start=refined_start,
            refined_end=refined_end,
            duration=final_duration,
            rejection_reason=None,
            snapped_to_scene_cut=snapped_cut,
            laughter_buffer_added=buffer_added,
        )

"""Boundary Refiner module for Auto Short Generator Phase A.

Refines candidate start and end timestamps:
1. Snaps start boundary to clean speech boundary (before hook, not mid-word, not mid-scene transition).
2. Snaps end boundary to speech completion (after payoff, sentence complete, preserving laughter/reaction).
3. Strictly enforces final clip duration: 30–55 seconds.
   Rejects candidate if duration cannot be satisfied within [30.0, 55.0] seconds.
"""

import logging
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, Field

from transcription.transcript_provider import TranscriptSegment
from transcription.whisper_aligner import WordToken

logger = logging.getLogger(__name__)


class RefinedBoundaryResult(BaseModel):
    """Result of boundary refinement and duration validation."""
    accepted: bool
    start_sec: float
    end_sec: float
    duration: float
    reason: str
    snapped_to_scene_cut: bool = False
    laughter_buffer_added: float = 0.0


class BoundaryRefiner:
    """Snaps clip boundaries to natural speech & scene cuts, enforcing 30–55s duration."""

    TARGET_MIN_DURATION: float = 30.0
    TARGET_MAX_DURATION: float = 55.0

    def __init__(
        self,
        min_duration_sec: float = 30.0,
        max_duration_sec: float = 55.0,
        laughter_buffer_sec: float = 0.5,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.laughter_buffer_sec = laughter_buffer_sec

    def refine_boundaries(
        self,
        initial_start: float,
        initial_end: float,
        phrase_segments: List[TranscriptSegment],
        word_tokens: Optional[List[WordToken]] = None,
        scene_cuts: Optional[List[float]] = None,
    ) -> RefinedBoundaryResult:
        """
        Performs speech-boundary snapping, scene cut coordination,
        post-payoff laughter preservation, and strict 30-55s duration validation.
        """
        if initial_end <= initial_start:
            return RefinedBoundaryResult(
                accepted=False,
                start_sec=initial_start,
                end_sec=initial_end,
                duration=0.0,
                reason="Invalid boundary timestamps: end <= start"
            )

        refined_start = initial_start
        refined_end = initial_end
        snapped_cut = False

        # 1. Snap Start to Speech Boundary (Word or Phrase)
        if word_tokens:
            # Find the word closest to initial_start
            matching_words = [w for w in word_tokens if w.start >= (initial_start - 1.5)]
            if matching_words:
                first_word = matching_words[0]
                refined_start = max(0.0, first_word.start - 0.05)  # 50ms breath pre-roll
        elif phrase_segments:
            # Snap to closest phrase start
            matching_phrases = [p for p in phrase_segments if p.start >= (initial_start - 2.0)]
            if matching_phrases:
                refined_start = matching_phrases[0].start

        # Check scene cuts near start: avoid starting 100-300ms before a scene transition
        if scene_cuts:
            for sc in scene_cuts:
                # If a scene cut occurs within 0.4s of refined_start, align with scene cut
                if abs(sc - refined_start) <= 0.40:
                    refined_start = sc
                    snapped_cut = True
                    break

        # 2. Snap End to Speech Boundary & Preserve Laughter/Reaction
        buffer_added = 0.0
        if word_tokens:
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
            return RefinedBoundaryResult(
                accepted=False,
                start_sec=refined_start,
                end_sec=refined_end,
                duration=final_duration,
                reason=(
                    f"Refined duration ({final_duration:.2f}s) is below minimum "
                    f"target {self.min_duration_sec:.1f}s"
                ),
                snapped_to_scene_cut=snapped_cut,
                laughter_buffer_added=buffer_added
            )

        if final_duration > self.max_duration_sec:
            return RefinedBoundaryResult(
                accepted=False,
                start_sec=refined_start,
                end_sec=refined_end,
                duration=final_duration,
                reason=(
                    f"Refined duration ({final_duration:.2f}s) exceeds maximum "
                    f"target {self.max_duration_sec:.1f}s"
                ),
                snapped_to_scene_cut=snapped_cut,
                laughter_buffer_added=buffer_added
            )

        return RefinedBoundaryResult(
            accepted=True,
            start_sec=refined_start,
            end_sec=refined_end,
            duration=final_duration,
            reason="Boundaries cleanly refined and satisfy 30-55s duration requirement",
            snapped_to_scene_cut=snapped_cut,
            laughter_buffer_added=buffer_added
        )

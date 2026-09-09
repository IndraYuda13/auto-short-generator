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
import re
from typing import List, Optional, Any, Tuple
from pydantic import BaseModel, model_validator

from transcription.transcript_provider import TranscriptSegment
from transcription.whisper_aligner import WordToken

logger = logging.getLogger(__name__)

# Indonesian dangling words and incomplete thought markers
DANGLING_CONNECTORS = {
    # Conjunctions & transitions
    "dan", "atau", "tapi", "tetapi", "karena", "sebab", "sehingga", "agar", "supaya",
    "bahwa", "jika", "kalau", "apabila", "ketika", "saat", "sementara", "sedangkan",
    "meskipun", "walaupun", "padahal", "makanya", "soalnya", "bahkan", "terus",
    "lalu", "kemudian", "melainkan", "namun",
    # Prepositions & specifiers
    "yang", "untuk", "dengan", "pada", "ke", "di", "dari", "tentang", "seperti",
    "oleh", "bagai", "bagaikan", "terhadap", "kepada", "sebagai", "mengenai",
    # Degree words & modifiers
    "sangat", "terlalu", "amat", "paling", "makin", "semakin", "kurang", "lebih",
    # Auxiliary verbs & modal words
    "masih", "sedang", "akan", "bisa", "harus", "mau", "belum", "pernah", "ingin",
    "sempat", "bakal",
    # Incomplete predicate / copula / discourse markers
    "sebenarnya", "adalah", "yaitu", "merupakan", "jadi", "yakni",
}

DANGLING_PHRASES = [
    "waktu itu masih",
    "karena sebenarnya",
    "jadi hidup",
    "dan kalau",
    "kalau sebenarnya",
    "tapi kalau",
    "tapi sebenarnya",
    "dan juga",
    "seperti yang",
    "yang sebenarnya",
    "hal yang",
    "bisa dibilang",
    "pada saat itu masih",
    "waktu itu",
    "pada saat",
    "di mana",
    "yang mana",
    "hal itu",
]


def is_sentence_complete(text: str) -> Tuple[bool, str]:
    """Check whether text ends with a complete sentence/thought.

    Returns (is_complete: bool, reason: str).
    """
    if not text or not text.strip():
        return True, "Empty text"

    cleaned = text.strip()

    # Check trailing ellipsis
    if cleaned.endswith("...") or cleaned.endswith("…"):
        return False, "Ends with trailing ellipsis '...'"

    # Check trailing mid-sentence punctuation
    if len(cleaned) > 1 and cleaned[-1] in (",", ";", ":", "-"):
        return False, f"Ends with mid-sentence punctuation '{cleaned[-1]}'"

    # Check dangling phrases (case-insensitive)
    lower_text = cleaned.lower()
    for phrase in DANGLING_PHRASES:
        if lower_text.endswith(phrase):
            return False, f"Ends with incomplete dangling phrase '{phrase}'"

    # Extract words without punctuation
    words = re.findall(r"\b\w+\b", lower_text)
    if not words:
        return True, "No words found"

    last_word = words[-1]
    if last_word in DANGLING_CONNECTORS:
        return False, f"Ends with dangling connector '{last_word}'"

    # If it ends with explicit sentence-ending punctuation (. ? !)
    if cleaned[-1] in (".", "?", "!"):
        return True, "Sentence ends with terminal punctuation"

    return True, "Sentence ending appears complete"


class RefinementResult(BaseModel):
    """Result of boundary refinement and duration validation."""
    is_valid: bool
    refined_start: float
    refined_end: float
    duration: float
    rejection_reason: Optional[str] = None
    snapped_to_scene_cut: bool = False
    laughter_buffer_added: float = 0.0
    sentence_complete: bool = True
    sentence_ending_reason: Optional[str] = None

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

    # Convenience alias for sentence completion checking
    is_sentence_complete = staticmethod(is_sentence_complete)

    def extend_to_sentence_boundary(
        self,
        start_sec: float,
        end_sec: float,
        segments: List[TranscriptSegment],
        max_duration_sec: Optional[float] = None,
        force_extend: bool = False,
    ) -> Tuple[bool, float, str]:
        """Extend end_sec to the end of the sentence/thought if new_duration <= max_duration_sec.

        Args:
            start_sec: Clip start time in seconds.
            end_sec: Clip end time in seconds.
            segments: Transcript segments to search across.
            max_duration_sec: Upper duration limit for the extended clip.
            force_extend: When True, bypasses 'already complete' check on current segments
                and actively searches later segments to reach the next sentence boundary.

        Returns:
            (success: bool, new_end_sec: float, reason: str)
        """
        max_dur = max_duration_sec or self.max_duration_sec
        if not segments:
            return False, end_sec, "No segments available for extension"

        # Find segments up to end_sec
        current_segs = [s for s in segments if s.start >= start_sec - 0.5 and s.end <= end_sec + 0.5]
        current_text = " ".join(s.text.strip() for s in current_segs if s.text)
        is_comp, comp_reason = is_sentence_complete(current_text)
        if is_comp and not force_extend:
            return True, end_sec, "Sentence ending is already complete"

        # Find subsequent segments ending after end_sec
        later_segs = [s for s in segments if s.end > end_sec]
        if not later_segs:
            return (
                False,
                end_sec,
                f"Cannot extend to sentence boundary: end of transcript reached with incomplete ending ({comp_reason})"
            )

        accumulated = list(current_segs)
        for s in later_segs:
            if s not in accumulated:
                accumulated.append(s)

            candidate_end = s.end + self.laughter_buffer_sec
            candidate_duration = round(candidate_end - start_sec, 2)
            if candidate_duration > max_dur:
                return (
                    False,
                    end_sec,
                    f"Cannot extend to complete sentence: duration {candidate_duration:.2f}s exceeds max {max_dur:.1f}s ({comp_reason})"
                )

            extended_text = " ".join(seg.text.strip() for seg in accumulated if seg.text)
            comp, r = is_sentence_complete(extended_text)
            if comp:
                return (
                    True,
                    round(candidate_end, 2),
                    f"Extended from {end_sec:.2f}s to {candidate_end:.2f}s ({candidate_duration:.2f}s) to complete sentence"
                )

        return (
            False,
            end_sec,
            f"Cannot extend to sentence boundary: no complete boundary found within {max_dur:.1f}s"
        )

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

        # 4. Check Sentence Boundary Completeness
        sentence_comp = True
        sentence_reason: Optional[str] = None
        if phrase_segments:
            included_segs = [p for p in phrase_segments if p.start >= refined_start - 0.5 and p.end <= refined_end + 0.5]
            if included_segs:
                full_text = " ".join(p.text.strip() for p in included_segs if p.text)
                if full_text:
                    is_comp, comp_reason = is_sentence_complete(full_text)
                    if not is_comp:
                        can_ext, new_end, ext_reason = self.extend_to_sentence_boundary(
                            start_sec=refined_start,
                            end_sec=refined_end,
                            segments=phrase_segments,
                            max_duration_sec=self.max_duration_sec,
                        )
                        if can_ext:
                            refined_end = new_end
                            final_duration = round(refined_end - refined_start, 2)
                            sentence_comp = True
                            sentence_reason = ext_reason
                        else:
                            return RefinementResult(
                                is_valid=False,
                                refined_start=refined_start,
                                refined_end=refined_end,
                                duration=final_duration,
                                rejection_reason=(
                                    f"Unfinished sentence ending: {comp_reason}. {ext_reason}"
                                ),
                                snapped_to_scene_cut=snapped_cut,
                                laughter_buffer_added=buffer_added,
                                sentence_complete=False,
                                sentence_ending_reason=ext_reason,
                            )

        # 5. Strict Duration Check [30.0, 55.0]s
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
                sentence_complete=sentence_comp,
                sentence_ending_reason=sentence_reason,
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
                sentence_complete=sentence_comp,
                sentence_ending_reason=sentence_reason,
            )

        return RefinementResult(
            is_valid=True,
            refined_start=refined_start,
            refined_end=refined_end,
            duration=final_duration,
            rejection_reason=None,
            snapped_to_scene_cut=snapped_cut,
            laughter_buffer_added=buffer_added,
            sentence_complete=sentence_comp,
            sentence_ending_reason=sentence_reason,
        )

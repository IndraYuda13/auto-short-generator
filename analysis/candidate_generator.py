"""Candidate Generator module for Auto Short Generator Phase A (Stage 4).

Segments long-form phrase-level transcripts into raw candidate windows of 25–70 seconds
based on:
1. Sentence boundaries (punctuation '.', '!', '?', or major speech pauses).
2. Natural pause boundaries (gap >= 0.5s between consecutive speech segments).
3. Sliding window traversal to ensure thorough coverage across the transcript.
4. Pre-context (1-2 sentences before) and Post-context (1-2 sentences after)
   for Gemini Semantic Scorer contextual evaluation.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

from transcription.transcript_provider import TranscriptSegment

logger = logging.getLogger(__name__)

SENTENCE_END_REGEX = re.compile(r'[.?!]+[\'"\s]*$')


def is_sentence_end(text: str) -> bool:
    """Returns True if text ends with sentence termination punctuation."""
    t = text.strip()
    return bool(SENTENCE_END_REGEX.search(t)) or t.endswith(("...", ":"))


class RawCandidate(BaseModel):
    """Raw candidate time window generated from transcript segments."""
    candidate_id: str
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str
    segments: List[Any] = Field(default_factory=list)
    pre_context: str = ""
    post_context: str = ""

    # Optional metadata fields for downstream pipeline compatibility
    segment_count: int = 0
    has_pause_before: bool = False
    has_pause_after: bool = False

    @property
    def duration(self) -> float:
        """Alias for duration_sec to maintain backward compatibility."""
        return self.duration_sec

    @model_validator(mode="before")
    @classmethod
    def _handle_compatibility(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "duration" in data and "duration_sec" not in data:
                data["duration_sec"] = float(data["duration"])
            elif "duration_sec" in data and "duration" not in data:
                data["duration"] = float(data["duration_sec"])

            if "segments" in data and isinstance(data["segments"], list):
                new_segs = []
                for seg in data["segments"]:
                    if hasattr(seg, "model_dump"):
                        new_segs.append(seg.model_dump())
                    elif isinstance(seg, dict):
                        new_segs.append(seg)
                    else:
                        new_segs.append(dict(seg))
                data["segments"] = new_segs
                if "segment_count" not in data or data["segment_count"] == 0:
                    data["segment_count"] = len(new_segs)
        return data


# Backward-compatible alias
CandidateWindow = RawCandidate


class CandidateGenerator:
    """Generates 25–70s raw candidate windows from phrase-level transcript segments."""

    def __init__(
        self,
        min_duration_sec: float = 25.0,
        max_duration_sec: float = 70.0,
        min_pause_sec: float = 0.5,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_pause_sec = min_pause_sec

    def _extract_pre_context(self, segments: List[TranscriptSegment], start_i: int) -> str:
        """Extracts 1-2 preceding sentences prior to start_i for context."""
        if start_i <= 0:
            return ""

        collected = []
        sentences_count = 0
        for k in range(start_i - 1, -1, -1):
            seg_text = segments[k].text.strip()
            collected.append(seg_text)
            if is_sentence_end(seg_text):
                sentences_count += 1
                if sentences_count >= 2:
                    break
            if len(collected) >= 4:
                break

        collected.reverse()
        return " ".join(collected).strip()

    def _extract_post_context(self, segments: List[TranscriptSegment], end_j: int) -> str:
        """Extracts 1-2 following sentences after end_j for context."""
        n = len(segments)
        if end_j >= n - 1:
            return ""

        collected = []
        sentences_count = 0
        for k in range(end_j + 1, n):
            seg_text = segments[k].text.strip()
            collected.append(seg_text)
            if is_sentence_end(seg_text):
                sentences_count += 1
                if sentences_count >= 2:
                    break
            if len(collected) >= 4:
                break

        return " ".join(collected).strip()

    def generate_candidates(
        self,
        segments: List[TranscriptSegment],
        max_candidates: int = 20,
    ) -> List[RawCandidate]:
        """
        Scans through segments, detects pauses and sentence ends,
        and constructs natural candidate windows of 25-70 seconds.
        Attaches pre_context and post_context for Gemini Semantic Scorer.
        """
        if not segments:
            return []

        n = len(segments)
        pauses_after = [False] * n
        sentence_ends = [False] * n

        for i in range(n):
            seg_text = segments[i].text.strip()
            if is_sentence_end(seg_text):
                sentence_ends[i] = True

            if i + 1 < n:
                gap = segments[i + 1].start - segments[i].end
                if gap >= self.min_pause_sec:
                    pauses_after[i] = True
                    sentence_ends[i] = True
            else:
                pauses_after[i] = True
                sentence_ends[i] = True

        candidates: List[RawCandidate] = []
        seen_ranges = []

        # Candidate start indices: start of video, or segments following pause/sentence end
        start_indices = [0]
        last_marked_start = 0
        for i in range(n - 1):
            if pauses_after[i] or sentence_ends[i]:
                start_indices.append(i + 1)
                last_marked_start = i + 1
            elif (segments[i + 1].start - segments[last_marked_start].start) >= 15.0:
                # Sliding window safeguard: ensure coverage even in unpunctuated runs
                start_indices.append(i + 1)
                last_marked_start = i + 1

        start_indices = sorted(list(set(start_indices)))

        cand_idx = 0
        for start_i in start_indices:
            start_time = segments[start_i].start
            pause_before = (start_i == 0) or pauses_after[start_i - 1]

            best_end_j = -1
            best_score = -1.0

            for end_j in range(start_i, n):
                end_time = segments[end_j].end
                dur = end_time - start_time

                if dur > self.max_duration_sec:
                    break

                if dur >= self.min_duration_sec:
                    is_sent_end = sentence_ends[end_j]
                    is_pause = pauses_after[end_j]

                    score = 0.0
                    if is_sent_end:
                        score += 3.0
                    if is_pause:
                        score += 3.0

                    # Proximity to ideal duration (~40-50s)
                    dur_diff = abs(dur - 45.0)
                    score += max(0.0, 5.0 - (dur_diff * 0.15))

                    if score > best_score:
                        best_score = score
                        best_end_j = end_j

            if best_end_j != -1:
                cand_segments = segments[start_i : best_end_j + 1]
                cand_start = round(cand_segments[0].start, 2)
                cand_end = round(cand_segments[-1].end, 2)
                cand_dur = round(cand_end - cand_start, 2)
                cand_text = " ".join(s.text.strip() for s in cand_segments)

                # Overlap deduplication (> 75% overlap check)
                is_duplicate = False
                for prev_start, prev_end in seen_ranges:
                    overlap_start = max(cand_start, prev_start)
                    overlap_end = min(cand_end, prev_end)
                    overlap_dur = max(0.0, overlap_end - overlap_start)
                    min_len = min(cand_dur, prev_end - prev_start)
                    if min_len > 0 and (overlap_dur / min_len) > 0.75:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    cand_idx += 1
                    seen_ranges.append((cand_start, cand_end))

                    pre_ctx = self._extract_pre_context(segments, start_i)
                    post_ctx = self._extract_post_context(segments, best_end_j)

                    serialized_segments: List[Dict[str, Any]] = []
                    for s in cand_segments:
                        if hasattr(s, "model_dump"):
                            serialized_segments.append(s.model_dump())
                        elif isinstance(s, dict):
                            serialized_segments.append(dict(s))
                        else:
                            serialized_segments.append(dict(vars(s)))

                    candidates.append(RawCandidate(
                        candidate_id=f"cand_{cand_idx:02d}",
                        start_sec=cand_start,
                        end_sec=cand_end,
                        duration_sec=cand_dur,
                        text=cand_text,
                        segments=serialized_segments,
                        pre_context=pre_ctx,
                        post_context=post_ctx,
                        segment_count=len(cand_segments),
                        has_pause_before=pause_before,
                        has_pause_after=pauses_after[best_end_j],
                    ))

                    if len(candidates) >= max_candidates:
                        break

        logger.info(f"Generated {len(candidates)} raw candidate windows (duration 25-70s)")
        return candidates

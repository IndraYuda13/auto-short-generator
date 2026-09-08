"""Candidate Generator module for Auto Short Generator Phase A.

Segments long-form phrase-level transcripts into candidate windows of 25–70 seconds
based on:
1. Silence/pause boundaries (gap >= 0.5s between consecutive speech segments).
2. Sentence boundaries (punctuation '.', '!', '?', or major speech pauses).
3. Natural speech phrase chunks.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from transcription.transcript_provider import TranscriptSegment

logger = logging.getLogger(__name__)


class CandidateWindow(BaseModel):
    """Raw candidate time window generated from transcript segments."""
    candidate_id: str
    start_sec: float
    end_sec: float
    duration: float
    text: str
    segment_count: int
    segments: List[TranscriptSegment] = Field(default_factory=list)
    has_pause_before: bool = False
    has_pause_after: bool = False


class CandidateGenerator:
    """Generates 25–70s candidate windows from phrase-level transcript segments."""

    def __init__(
        self,
        min_duration_sec: float = 25.0,
        max_duration_sec: float = 70.0,
        min_pause_sec: float = 0.5,
    ):
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.min_pause_sec = min_pause_sec

    def generate_candidates(
        self,
        segments: List[TranscriptSegment],
        max_candidates: int = 20,
    ) -> List[CandidateWindow]:
        """
        Scans through segments, detects pauses and sentence ends,
        and constructs natural candidate windows of 25-70 seconds.
        """
        if not segments:
            return []

        n = len(segments)
        # Compute boundary properties for each segment
        pauses_after = [False] * n
        sentence_ends = [False] * n

        for i in range(n):
            seg_text = segments[i].text.strip()
            # Punctuation check
            if seg_text.endswith((".", "!", "?", "...", ":")):
                sentence_ends[i] = True

            # Pause check to next segment
            if i + 1 < n:
                gap = segments[i + 1].start - segments[i].end
                if gap >= self.min_pause_sec:
                    pauses_after[i] = True
                    sentence_ends[i] = True
            else:
                pauses_after[i] = True
                sentence_ends[i] = True

        candidates: List[CandidateWindow] = []
        seen_ranges = []

        # Candidate start indices: start of video, or any segment following a pause/sentence end
        start_indices = [0]
        for i in range(n - 1):
            if pauses_after[i] or sentence_ends[i]:
                start_indices.append(i + 1)

        # Remove duplicate start indices
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
                    # Score how natural the stopping point is
                    is_sentence_end = sentence_ends[end_j]
                    is_pause_after = pauses_after[end_j]

                    # Bonus for stopping at natural sentence or pause boundary
                    score = 0.0
                    if is_sentence_end:
                        score += 2.0
                    if is_pause_after:
                        score += 3.0

                    # Proximity to ideal duration (~40-50s)
                    dur_diff = abs(dur - 45.0)
                    score += max(0.0, 5.0 - (dur_diff * 0.1))

                    if score > best_score:
                        best_score = score
                        best_end_j = end_j

            if best_end_j != -1:
                cand_segments = segments[start_i : best_end_j + 1]
                cand_start = round(cand_segments[0].start, 2)
                cand_end = round(cand_segments[-1].end, 2)
                cand_dur = round(cand_end - cand_start, 2)
                cand_text = " ".join(s.text.strip() for s in cand_segments)

                # Check overlap with already selected candidates (> 75% overlap check)
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
                    candidates.append(CandidateWindow(
                        candidate_id=f"cand_{cand_idx:02d}",
                        start_sec=cand_start,
                        end_sec=cand_end,
                        duration=cand_dur,
                        text=cand_text,
                        segment_count=len(cand_segments),
                        segments=cand_segments,
                        has_pause_before=pause_before,
                        has_pause_after=pauses_after[best_end_j],
                    ))

                    if len(candidates) >= max_candidates:
                        break

        logger.info(f"Generated {len(candidates)} raw candidate windows (duration 25-70s)")
        return candidates

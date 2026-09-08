"""Pacing engine: conservative silence removal with monotonic timeline remapping.

Removes dead air between words without cutting through speech boundaries.
Ensures monotonic timeline remapping for subtitles, audio, and visual edit events.
"""

from typing import List, Dict, Any, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class TimelineSegment:
    def __init__(self, src_start: float, src_end: float, dst_start: float, dst_end: float):
        self.src_start = src_start
        self.src_end = src_end
        self.dst_start = dst_start
        self.dst_end = dst_end

    def __repr__(self):
        return f"TimelineSegment(src=[{self.src_start:.2f}, {self.src_end:.2f}], dst=[{self.dst_start:.2f}, {self.dst_end:.2f}])"


class PacingEngine:
    """Calculates speech-preserving cut points and provides remapping functions."""

    def __init__(
        self,
        min_silence_gap: float = 0.65,    # Do not touch gaps shorter than 0.65s
        speech_padding: float = 0.15,     # Keep 0.15s padding before and after speech
        max_silence_removal: float = 0.8  # Cap maximum duration removed from any single gap
    ):
        self.min_silence_gap = min_silence_gap
        self.speech_padding = speech_padding
        self.max_silence_removal = max_silence_removal

    def build_pacing_timeline(
        self,
        word_segments: List[Dict[str, Any]],
        clip_duration: float
    ) -> Tuple[List[TimelineSegment], Callable[[float], float]]:
        """
        Takes word-level timestamps (clip-local: 0.0 to clip_duration).
        Identifies gaps between words > min_silence_gap.
        Shrinks the silence while preserving speech padding.
        Returns the list of retained source segments and a monotonic remap(t) function.
        """
        if not word_segments or clip_duration <= 0:
            # Identity timeline
            identity_seg = TimelineSegment(0.0, clip_duration, 0.0, clip_duration)
            return [identity_seg], lambda t: t

        # Clean and sort words by start time
        valid_words = []
        for w in word_segments:
            w_start = float(w.get("start", 0.0))
            w_end = float(w.get("end", w_start + 0.1))
            if w_end > w_start:
                valid_words.append((w_start, w_end))
        valid_words.sort(key=lambda x: x[0])

        if not valid_words:
            identity_seg = TimelineSegment(0.0, clip_duration, 0.0, clip_duration)
            return [identity_seg], lambda t: t

        # Merge overlapping speech words into continuous speech blocks
        speech_blocks: List[Tuple[float, float]] = []
        curr_s, curr_e = valid_words[0]
        for s, e in valid_words[1:]:
            if s <= curr_e:
                curr_e = max(curr_e, e)
            else:
                speech_blocks.append((curr_s, curr_e))
                curr_s, curr_e = s, e
        speech_blocks.append((curr_s, curr_e))

        # Build cuts in silence between speech blocks
        # Retained source intervals [src_start, src_end]
        retained_intervals: List[Tuple[float, float]] = []
        last_speech_end = 0.0

        for s_start, s_end in speech_blocks:
            gap = s_start - last_speech_end
            if gap > self.min_silence_gap:
                # Retain speech padding
                cut_start = last_speech_end + self.speech_padding
                cut_end = s_start - self.speech_padding
                actual_cut_duration = cut_end - cut_start

                if actual_cut_duration > 0.1:
                    # Apply max cut cap to prevent unnatural stutter
                    effective_cut = min(actual_cut_duration, self.max_silence_removal)
                    cut_end = cut_start + effective_cut

                    # Segment before cut
                    if cut_start > last_speech_end:
                        retained_intervals.append((last_speech_end, cut_start))
                    # Retain speech block from cut_end to s_end
                    retained_intervals.append((cut_end, s_end))
                    last_speech_end = s_end
                else:
                    # Gap too small after padding
                    retained_intervals.append((last_speech_end, s_end))
                    last_speech_end = s_end
            else:
                # Keep whole interval
                retained_intervals.append((last_speech_end, s_end))
                last_speech_end = s_end

        # Final tail
        if last_speech_end < clip_duration:
            retained_intervals.append((last_speech_end, clip_duration))

        # Merge contiguous retained intervals
        clean_retained: List[Tuple[float, float]] = []
        for r_start, r_end in retained_intervals:
            if clean_retained and abs(clean_retained[-1][1] - r_start) < 0.001:
                clean_retained[-1] = (clean_retained[-1][0], r_end)
            elif r_end > r_start:
                clean_retained.append((r_start, r_end))

        if not clean_retained:
            identity_seg = TimelineSegment(0.0, clip_duration, 0.0, clip_duration)
            return [identity_seg], lambda t: t

        # Build timeline segments and monotonic mapping
        segments: List[TimelineSegment] = []
        current_dst = 0.0
        for src_s, src_e in clean_retained:
            seg_dur = src_e - src_s
            segments.append(TimelineSegment(
                src_start=src_s,
                src_end=src_e,
                dst_start=current_dst,
                dst_end=current_dst + seg_dur
            ))
            current_dst += seg_dur

        def remap_timestamp(src_t: float) -> float:
            """Monotonically projects source timestamp to edited destination timeline."""
            if src_t <= segments[0].src_start:
                return segments[0].dst_start
            if src_t >= segments[-1].src_end:
                return segments[-1].dst_end

            for seg in segments:
                if seg.src_start <= src_t <= seg.src_end:
                    offset = src_t - seg.src_start
                    return seg.dst_start + offset
                if src_t < seg.src_start:
                    # In a cut gap, clamp to boundary
                    return seg.dst_start

            return segments[-1].dst_end

        return segments, remap_timestamp


pacing_engine = PacingEngine()

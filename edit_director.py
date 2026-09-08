"""Edit Director module: generates typed EditPlan for specific viral clips.

Transforms clip metadata and overlapping transcript into a clean, deterministic EditPlan.
Enforces calm editing rules: PODCAST_CLEAN profile, 0-3 punch-ins max, no random novelty hooks.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional

from edit_plan import (
    EditPlan,
    EditingProfile,
    FramingMode,
    EditEvent,
    EditEventType,
    SubtitleStyle,
    AudioProfile,
)
from llm_client import llm_client

logger = logging.getLogger(__name__)


class EditDirector:
    """Directs editing treatment for selected short-form video clips."""

    def __init__(self):
        pass

    def create_plan_for_clip(
        self,
        clip_id: str,
        start_sec: float,
        end_sec: float,
        transcript_segments: List[Dict[str, Any]],
        video_title: str = "",
        hook_reason: str = "",
        framing_mode: FramingMode = FramingMode.BLURRED_FALLBACK
    ) -> EditPlan:
        """
        Creates an EditPlan for the specified clip window.
        Attempts LLM analysis with low temperature (deterministic).
        Falls back to a safe default PODCAST_CLEAN plan on any failure or parsing error.
        """
        clip_duration = max(0.1, end_sec - start_sec)
        default_plan = EditPlan.create_default(
            clip_id=clip_id,
            duration=clip_duration,
            framing_mode=framing_mode
        )

        # Filter transcript segments overlapping the clip window
        overlapping_segments = []
        for s in transcript_segments:
            s_start = s.get("start", 0.0)
            s_end = s.get("end", s_start + s.get("duration", 0.0))
            if s_end > start_sec and s_start < end_sec:
                # Store with clip-local timing
                local_start = max(0.0, s_start - start_sec)
                local_end = min(clip_duration, s_end - start_sec)
                overlapping_segments.append({
                    "start": round(local_start, 2),
                    "end": round(local_end, 2),
                    "text": s.get("text", "").strip()
                })

        if not overlapping_segments:
            logger.info(f"No transcript segments overlapping clip {clip_id}, using default plan.")
            return default_plan

        try:
            plan = self._analyze_with_llm(
                clip_id=clip_id,
                clip_duration=clip_duration,
                video_title=video_title,
                hook_reason=hook_reason,
                segments=overlapping_segments,
                framing_mode=framing_mode
            )
            return plan
        except Exception as e:
            logger.warning(f"EditDirector LLM analysis failed: {e}. Falling back to default PODCAST_CLEAN plan.")
            return default_plan

    def _analyze_with_llm(
        self,
        clip_id: str,
        clip_duration: float,
        video_title: str,
        hook_reason: str,
        segments: List[Dict[str, Any]],
        framing_mode: FramingMode
    ) -> EditPlan:
        """Calls LLM to generate semantic emphasis moments and subtle punch-ins."""
        transcript_text = "\n".join([f"[{s['start']:.1f}s - {s['end']:.1f}s] {s['text']}" for s in segments])

        system_prompt = (
            "You are a master minimalist short-form video editor specializing in high-retention podcast shorts. "
            "Your philosophy: Do not clutter the screen. Never use novelty transitions (no camera shake, no paper tear, no glitch). "
            "Only add subtle punch-in cuts (115% zoom) on 1 to 3 critical punchlines or revelation moments. "
            "Pick 2 to 5 high-impact emphasis words for visual prominence."
        )

        prompt = f"""Clip ID: {clip_id}
Clip Duration: {clip_duration:.1f}s
Context Title: "{video_title}"
Viral Hook Angle: "{hook_reason}"

Transcript within this clip (timestamps are clip-local seconds 0.0 to {clip_duration:.1f}s):
---
{transcript_text}
---

Task:
Produce an Edit Plan in JSON format.
Rules:
1. `profile`: Must be "PODCAST_CLEAN" (or "COMEDY", "HIGH_ENERGY", "STORY", "NEWS").
2. `edit_events`: Up to 2 subtle PUNCH_IN events maximum. Each punch-in lasts between 1.2s and 2.5s.
   Never place events less than 3.0 seconds apart. Never exceed clip duration.
3. `emphasis_words`: 2 to 5 specific single words from the transcript that convey highest punch/climax.

Return ONLY valid JSON:
{{
  "profile": "PODCAST_CLEAN",
  "edit_events": [
    {{
      "time": 4.5,
      "type": "PUNCH_IN",
      "duration": 1.5,
      "intensity": 1.15
    }}
  ],
  "emphasis_words": ["kata1", "kata2"]
}}"""

        raw_response = llm_client.chat_completion(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Low variance deterministic
            json_mode=False
        )

        parsed = self._extract_json(raw_response)
        profile_str = parsed.get("profile", "PODCAST_CLEAN")
        try:
            profile = EditingProfile(profile_str)
        except Exception:
            profile = EditingProfile.PODCAST_CLEAN

        raw_events = parsed.get("edit_events", [])
        validated_events: List[EditEvent] = []
        last_event_time = -10.0

        for ev in raw_events:
            ev_time = float(ev.get("time", 0.0))
            ev_type_str = ev.get("type", "PUNCH_IN")
            ev_dur = float(ev.get("duration", 1.5))
            ev_int = float(ev.get("intensity", 1.15))

            # Enforce calm density: at least 2.5s spacing, max 2 punch-ins
            if ev_time >= 0.5 and (ev_time + ev_dur) <= clip_duration:
                if (ev_time - last_event_time) >= 2.5 and len(validated_events) < 3:
                    try:
                        ev_type = EditEventType(ev_type_str)
                    except Exception:
                        ev_type = EditEventType.PUNCH_IN
                    validated_events.append(EditEvent(
                        time=round(ev_time, 2),
                        type=ev_type,
                        duration=round(ev_dur, 2),
                        intensity=round(ev_int, 2),
                        text=None
                    ))
                    last_event_time = ev_time

        raw_words = parsed.get("emphasis_words", [])
        emphasis_words = [str(w).strip().lower() for w in raw_words if isinstance(w, str) and len(w.strip()) > 1][:6]

        return EditPlan(
            clip_id=clip_id,
            clip_duration=clip_duration,
            profile=profile,
            framing_mode=framing_mode,
            crop_keyframes=[],
            edit_events=validated_events,
            emphasis_words=emphasis_words,
            subtitle_style=SubtitleStyle(),
            audio_profile=AudioProfile()
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        clean = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE).strip()
        try:
            data = json.loads(clean)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {}


edit_director = EditDirector()

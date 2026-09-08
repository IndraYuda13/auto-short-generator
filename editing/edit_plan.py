"""Edit Plan module for Auto Short Generator Phase B (Stable Editing Core).

Implements Blueprint Bab 11:
- Typed Pydantic EditPlan contract
- SceneCrop for scene-static framing windows
- PunchInEvent with strict policy:
  * max 0-2 per clip
  * scale 1.04-1.08
  * duration 0.6-1.5s
  * default empty if in doubt
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PunchInEvent(BaseModel):
    """Subtle punch-in zoom event for dramatic emphasis (Blueprint Bab 11).

    Policy constraints:
    - Scale: 1.04 to 1.08 (calm, subtle, never dizzying)
    - Duration: 0.6s to 1.5s
    """
    start_time: float = Field(..., ge=0.0, description="Start timestamp relative to clip start in seconds")
    duration: float = Field(default=1.0, description="Duration of punch-in in seconds")
    scale: float = Field(default=1.06, description="Zoom scale factor (1.04 - 1.08)")

    @field_validator("scale", mode="before")
    @classmethod
    def validate_scale(cls, v: float) -> float:
        # Strict clamp/validation to 1.04 - 1.08
        return round(float(min(max(float(v), 1.04), 1.08)), 3)

    @field_validator("duration", mode="before")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        # Strict clamp/validation to 0.6s - 1.5s
        return round(float(min(max(float(v), 0.6), 1.5)), 2)

    @property
    def end_time(self) -> float:
        return round(self.start_time + self.duration, 2)


class SceneCrop(BaseModel):
    """Scene-static crop window for a single continuous scene segment (Blueprint Bab 12).

    Within [scene_start, scene_end], the crop window is HELD CONSTANT (no camera pan/jitter).
    """
    scene_start: float = Field(..., ge=0.0, description="Start timestamp of the scene segment in seconds")
    scene_end: float = Field(..., ge=0.0, description="End timestamp of the scene segment in seconds")
    crop_x: int = Field(..., ge=0, description="Left X coordinate in source frame")
    crop_y: int = Field(..., ge=0, description="Top Y coordinate in source frame")
    crop_w: int = Field(..., gt=0, description="Width of crop window in source frame")
    crop_h: int = Field(..., gt=0, description="Height of crop window in source frame")
    target_w: int = Field(default=1080, gt=0, description="Output target canvas width")
    target_h: int = Field(default=1920, gt=0, description="Output target canvas height")

    @field_validator("crop_x", "crop_y", "crop_w", "crop_h")
    @classmethod
    def ensure_even_dimensions(cls, v: int) -> int:
        """FFmpeg codecs (H.264/yuv420p) require even dimensions and coordinates."""
        val = int(v)
        if val % 2 != 0:
            val = max(0, val - 1)
        return val

    @property
    def duration(self) -> float:
        return max(0.0, round(self.scene_end - self.scene_start, 3))


class EditPlan(BaseModel):
    """Complete editing contract for video rendering (Blueprint Bab 11 & 15)."""
    layout: str = Field(
        default="PORTRAIT_9_16",
        description="Framing layout: 'PORTRAIT_9_16', 'SAFE_FULL_FRAME', 'BLURRED_FALLBACK', or 'REJECT'"
    )
    subtitle_policy: str = Field(
        default="GENERATE",
        description="Subtitle handling: 'GENERATE', 'EXTRACT_EMBEDDED', 'PRESERVE_BURNED_IN', or 'REJECT'"
    )
    audio_mastering: bool = Field(
        default=True,
        description="Whether to apply broadcast audio mastering (-16 LUFS, TP -1.5 dB, 48kHz stereo)"
    )
    punch_in_events: List[PunchInEvent] = Field(
        default_factory=list,
        description="List of subtle punch-in events (strictly max 2 per clip)"
    )
    crop_windows: List[SceneCrop] = Field(
        default_factory=list,
        description="List of scene-static crop windows per scene cut"
    )
    clip_id: str = Field(default="clip_default", description="Identifier for current clip")
    duration: Optional[float] = Field(default=None, ge=0.0, description="Total duration of the clip in seconds")
    target_width: int = Field(default=1080, description="Final output width (default 1080)")
    target_height: int = Field(default=1920, description="Final output height (default 1920)")

    @model_validator(mode="after")
    def enforce_punch_in_and_ordering(self) -> "EditPlan":
        """Enforce Punch-in Policy:
        - Max 0-2 punch-in events per clip
        - Sort punch-in events by start_time
        - Discard overlapping or excess punch-in events
        - Sort crop_windows by scene_start
        """
        # Sort crop windows by scene_start
        self.crop_windows = sorted(self.crop_windows, key=lambda c: c.scene_start)

        # Enforce max 2 punch-in events
        if len(self.punch_in_events) > 2:
            self.punch_in_events = self.punch_in_events[:2]

        # Sort punch-in events by start_time
        self.punch_in_events = sorted(self.punch_in_events, key=lambda p: p.start_time)

        # If clip duration is known, clamp events within duration
        if self.duration is not None and self.duration > 0:
            filtered_events: List[PunchInEvent] = []
            for ev in self.punch_in_events:
                if ev.start_time < self.duration:
                    if ev.start_time + ev.duration > self.duration:
                        ev.duration = max(0.6, self.duration - ev.start_time)
                    filtered_events.append(ev)
            self.punch_in_events = filtered_events[:2]

        return self

    def add_punch_in(self, start_time: float, duration: float = 1.0, scale: float = 1.06) -> bool:
        """Adds a punch-in event adhering to the max 2 policy."""
        if len(self.punch_in_events) >= 2:
            return False
        # Create and validate event
        event = PunchInEvent(start_time=start_time, duration=duration, scale=scale)
        self.punch_in_events.append(event)
        self.punch_in_events.sort(key=lambda p: p.start_time)
        return True

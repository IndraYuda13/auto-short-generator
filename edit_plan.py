"""Data models and validation for Editor V2 contracts."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EditingProfile(str, Enum):
    PODCAST_CLEAN = "PODCAST_CLEAN"
    HIGH_ENERGY = "HIGH_ENERGY"
    STORY = "STORY"
    NEWS = "NEWS"
    COMEDY = "COMEDY"


class FramingMode(str, Enum):
    FACE_TRACKED = "FACE_TRACKED"
    CENTER_CROP = "CENTER_CROP"
    BLURRED_FALLBACK = "BLURRED_FALLBACK"


class EditEventType(str, Enum):
    PUNCH_IN = "PUNCH_IN"
    PUNCH_OUT = "PUNCH_OUT"
    EMPHASIS = "EMPHASIS"
    REFRAME = "REFRAME"


class EditEvent(BaseModel):
    time: float = Field(..., ge=0.0, description="Clip-local timestamp in seconds")
    type: EditEventType = Field(..., description="Type of visual/audio event")
    duration: Optional[float] = Field(None, ge=0.0, description="Optional duration of the event")
    intensity: Optional[float] = Field(default=1.0, ge=0.0, le=2.0, description="Subtle intensity scalar (0.0 - 2.0)")
    text: Optional[str] = Field(default=None, description="Associated word or phrase for emphasis")


class CropKeyframe(BaseModel):
    time: float = Field(..., ge=0.0, description="Clip-local timestamp in seconds")
    crop_center_x: float = Field(..., ge=0.0, le=1.0, description="Normalized crop center X in source [0.0, 1.0]")
    crop_center_y: float = Field(..., ge=0.0, le=1.0, description="Normalized crop center Y in source [0.0, 1.0]")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence [0.0, 1.0]")


class SubtitleStyle(BaseModel):
    font_name: str = "Montserrat-Black"
    font_size: int = 52
    margin_v: int = 540
    highlight_color: str = "&H0000E6FF"  # Neon Yellow in ASS (&HAABBGGRR)
    primary_color: str = "&H00FFFFFF"    # White in ASS
    outline_color: str = "&H00000000"    # Black in ASS
    outline_width: int = 5
    shadow_width: int = 3
    max_words_per_line: int = 4
    active_word_scale: int = 100         # Calmer default (100% instead of aggressive 108%)


class AudioProfile(BaseModel):
    highpass_freq: int = 80              # Mild rumble removal at 80Hz
    compressor_enabled: bool = True
    loudness_target_lufs: float = -16.0  # Industry standard short-form speech
    true_peak_db: float = -1.5


class EditPlan(BaseModel):
    version: str = "2.0"
    clip_id: str = "default_clip"
    clip_duration: float = Field(..., gt=0.0, description="Total duration of the clip in seconds")
    profile: EditingProfile = EditingProfile.PODCAST_CLEAN
    framing_mode: FramingMode = FramingMode.BLURRED_FALLBACK
    crop_keyframes: List[CropKeyframe] = Field(default_factory=list)
    edit_events: List[EditEvent] = Field(default_factory=list)
    emphasis_words: List[str] = Field(default_factory=list)
    subtitle_style: SubtitleStyle = Field(default_factory=SubtitleStyle)
    audio_profile: AudioProfile = Field(default_factory=AudioProfile)

    @model_validator(mode="after")
    def validate_events_and_keyframes(self) -> "EditPlan":
        # Ensure edit events do not exceed clip duration; clamp or discard out-of-range events
        clamped_events: List[EditEvent] = []
        for event in self.edit_events:
            if event.time <= self.clip_duration:
                # Clamp event duration if it extends past clip end
                if event.duration is not None and event.time + event.duration > self.clip_duration:
                    event.duration = max(0.0, self.clip_duration - event.time)
                clamped_events.append(event)
        self.edit_events = sorted(clamped_events, key=lambda e: e.time)

        # Sort crop keyframes by timestamp
        self.crop_keyframes = sorted(self.crop_keyframes, key=lambda k: k.time)
        return self

    @classmethod
    def create_default(
        cls,
        clip_id: str,
        duration: float,
        framing_mode: FramingMode = FramingMode.BLURRED_FALLBACK
    ) -> "EditPlan":
        """Deterministic, calm default edit plan for PODCAST_CLEAN."""
        return cls(
            clip_id=clip_id,
            clip_duration=duration,
            profile=EditingProfile.PODCAST_CLEAN,
            framing_mode=framing_mode,
            crop_keyframes=[],
            edit_events=[],
            emphasis_words=[],
            subtitle_style=SubtitleStyle(),
            audio_profile=AudioProfile(),
        )

"""Editing package for Auto Short Generator Phase B (Stable Editing Core).

Implements Blueprint Bab 11, 12, 13, 14, 15:
- Bab 11: EditPlan, PunchInEvent, SceneCrop contracts
- Bab 12: Scene-static portrait framing (SceneStaticFraming)
- Bab 13: Subtitle policy classification and clean ASS generator (SubtitlePolicyClassifier)
- Bab 14: Broadcast audio mastering (-16 LUFS, TP -1.5 dB, 48kHz stereo, zero SFX)
- Bab 15: Clean FFmpeg filtergraph executor (CleanRenderer)
"""

from editing.edit_plan import EditPlan, PunchInEvent, SceneCrop
from editing.framing import SceneStaticFraming, FramingDecision
from editing.subtitle_policy import (
    SubtitlePolicyClassifier,
    SubtitleClassificationResult,
    SubtitleSourceType,
    SubtitleAction,
    generate_clean_ass_subtitles,
)
from editing.audio import (
    AudioMasterer,
    get_audio_filter_chain,
    build_audio_encoding_args,
    AUDIO_FILTER_CHAIN,
)
from editing.renderer import CleanRenderer, RenderResult

__all__ = [
    "EditPlan",
    "PunchInEvent",
    "SceneCrop",
    "SceneStaticFraming",
    "FramingDecision",
    "SubtitlePolicyClassifier",
    "SubtitleClassificationResult",
    "SubtitleSourceType",
    "SubtitleAction",
    "generate_clean_ass_subtitles",
    "AudioMasterer",
    "get_audio_filter_chain",
    "build_audio_encoding_args",
    "AUDIO_FILTER_CHAIN",
    "CleanRenderer",
    "RenderResult",
]

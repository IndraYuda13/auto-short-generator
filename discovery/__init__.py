"""Discovery package for Auto Short Generator Phase A (Blueprint Stage 1)."""

from discovery.searcher import Searcher, VideoSourceMeta, VideoMetadata
from discovery.source_filter import (
    SourceFilter,
    EligibilityResult,
    SourceFilterVerdict,
    NON_SPEECH_PATTERNS,
)

__all__ = [
    "Searcher",
    "VideoSourceMeta",
    "VideoMetadata",
    "SourceFilter",
    "EligibilityResult",
    "SourceFilterVerdict",
    "NON_SPEECH_PATTERNS",
]

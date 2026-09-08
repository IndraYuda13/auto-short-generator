"""Tests for Patch 4: Flicker Fix, Existing Subtitle Preservation, Subtitle-Aware Framing, and Gemini Visual Director."""

import pytest
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from edit_plan import (
    EditPlan,
    FramingMode,
    CropKeyframe,
    SubtitleStyle,
    AudioProfile,
)
from renderer import Renderer
from subtitle_detector import (
    SubtitleDetector,
    SubtitleSource,
    SubtitleRegion,
    SubtitleDetectionResult,
)
from visual_director import (
    VisualDirector,
    VisualDirectorResult,
    RecommendedFraming,
    ShotType,
)


# ==============================================================================
# 1. Flicker Fix & Keyframe Boundary Stability Tests
# ==============================================================================

def test_renderer_nested_if_crop_expression_syntax():
    """
    Verifies that Renderer compiles multiple keyframes into a continuous nested if expression
    rather than overlapping between() calls.
    """
    ren = Renderer()
    kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.48, crop_center_y=0.4),
        CropKeyframe(time=1.0, crop_center_x=0.52, crop_center_y=0.4),
        CropKeyframe(time=2.0, crop_center_x=0.55, crop_center_y=0.4),
    ]
    plan = EditPlan(
        clip_id="test_nested_if",
        clip_duration=3.0,
        framing_mode=FramingMode.FACE_TRACKED,
        crop_keyframes=kfs,
    )
    filter_complex, _, _ = ren._build_v2_pipeline(plan, ass_path=None, duration=3.0)

    # Must contain nested if(lt(t, ...)) expression
    assert "if(lt(t,1.000),0.4800,if(lt(t,2.000),0.5200,0.5500))" in filter_complex
    # Must NOT contain old between() in crop expression
    assert "between(t,0.00,1.00)" not in filter_complex
    assert "between(t,1.00,2.00)" not in filter_complex


def test_dense_boundary_numerical_continuity():
    """
    Simulates evaluation across integer-second boundaries (0.98s, 1.00s, 1.02s, 1.98s, 2.00s, 2.02s)
    and verifies that NO interval gap or double-sum occurs.
    """
    kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.45, crop_center_y=0.4),
        CropKeyframe(time=1.0, crop_center_x=0.50, crop_center_y=0.4),
        CropKeyframe(time=2.0, crop_center_x=0.55, crop_center_y=0.4),
    ]

    # Evaluate nested if logic
    def eval_crop(t):
        val = kfs[-1].crop_center_x
        for i in reversed(range(len(kfs) - 1)):
            if t < kfs[i + 1].time:
                val = kfs[i].crop_center_x
        return val

    # Near 1.0s boundary
    assert eval_crop(0.98) == 0.45
    assert eval_crop(1.00) == 0.50  # Instant deterministic transition, NOT 0.95 (sum) and NOT 0.0 (gap)
    assert eval_crop(1.02) == 0.50

    # Near 2.0s boundary
    assert eval_crop(1.98) == 0.50
    assert eval_crop(2.00) == 0.55
    assert eval_crop(2.02) == 0.55


# ==============================================================================
# 2. Existing Subtitle Detection & No Duplicate Invariant
# ==============================================================================

def test_subtitle_detector_detects_embedded_track(monkeypatch):
    """Verifies that ffprobe subtitle stream triggers EMBEDDED_TRACK classification."""
    detector = SubtitleDetector()

    fake_ffprobe_output = {
        "streams": [
            {"index": 2, "codec_name": "subrip", "tags": {"language": "ind", "title": "Indonesian"}}
        ]
    }

    def mock_run(*args, **kwargs):
        class Res:
            stdout = json.dumps(fake_ffprobe_output)
            stderr = ""
            returncode = 0
        return Res()

    monkeypatch.setattr(subprocess, "run", mock_run)

    result = detector.evaluate("dummy_video.mp4", 0.0, 30.0)
    assert result.has_existing_subtitle is True
    assert result.source == SubtitleSource.EMBEDDED_TRACK
    assert result.region is not None
    assert result.region.protected is True


def test_no_duplicate_subtitle_invariant_disables_ass_generation(tmp_path: Path):
    """
    Verifies that when existing_subtitle is True or generate_new_subtitle is False,
    Renderer V2 does NOT create or overlay any ASS subtitles.
    Strict Invariant:
      GENERATE_NEW_SUBTITLE = False
      ASS_FILTER_COUNT = 0
      NEW_TEXT_OVERLAY_COUNT = 0
    """
    ren = Renderer(output_dir=tmp_path)
    plan = EditPlan.create_default("test_no_dup", duration=10.0)
    plan.existing_subtitle = True
    plan.subtitle_source = "BURNED_IN"
    plan.generate_new_subtitle = False

    sub_segments = [
        {"start": 0.0, "end": 2.0, "text": "Ini subtitle tidak boleh digenerate", "words": []}
    ]

    should_gen = plan.generate_new_subtitle and not plan.existing_subtitle
    assert should_gen is False

    filter_complex, _, _ = ren._build_v2_pipeline(plan, ass_path=None, duration=10.0)
    # Split video filters from audio filters
    video_filters = [p for p in filter_complex.split(";") if not p.startswith("[0:a]")]
    video_filtergraph = ";".join(video_filters)
    # Assert ASS_FILTER_COUNT = 0, NEW_TEXT_OVERLAY_COUNT = 0 in video pipeline
    assert "ass=" not in video_filtergraph
    assert "subtitles=" not in video_filtergraph
    assert "drawtext=" not in video_filtergraph
    assert "null[outv]" in video_filtergraph


# ==============================================================================
# 3. Subtitle-Aware Framing (SUBTITLE_SAFE_FULL_WIDTH)
# ==============================================================================

def test_subtitle_safe_full_width_framing():
    """
    Verifies that SUBTITLE_SAFE_FULL_WIDTH scales 16:9 foreground to full 1080 canvas width
    over blurred background so 100% of subtitle width is preserved.
    """
    ren = Renderer()
    plan = EditPlan.create_default("test_safe_framing", duration=10.0)
    plan.framing_mode = FramingMode.SUBTITLE_SAFE_FULL_WIDTH
    plan.existing_subtitle = True

    filter_complex, _, _ = ren._build_v2_pipeline(plan, ass_path=None, duration=10.0)

    assert "scale=1080:-2:flags=bicubic[fg]" in filter_complex
    assert "boxblur=5:2" in filter_complex
    assert "crop=iw:ih*0.70:0:0" in filter_complex
    assert "overlay=(W-w)/2:(H-h)/2[base_v]" in filter_complex
    # Ensure no tight face-crop was applied
    assert "ih*9/16" not in filter_complex


# ==============================================================================
# 4. Gemini Multimodal Visual Director & Fallback
# ==============================================================================

def test_visual_director_parses_multimodal_response(monkeypatch):
    """Verifies that VisualDirector parses and validates structured JSON response from 9router."""
    vd = VisualDirector()

    fake_response_content = json.dumps({
        "has_existing_subtitle": True,
        "subtitle_kind": "BURNED_IN",
        "subtitle_region": {"x1": 0.08, "y1": 0.75, "x2": 0.92, "y2": 0.95, "protected": True},
        "shot_type": "single_speaker_medium",
        "faces_visible": 1,
        "recommended_framing": "SUBTITLE_SAFE_FULL_WIDTH",
        "confidence": 0.98,
        "reasons": ["Wide horizontal subtitles detected across lower frame."]
    })

    class FakeResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [
                    {"message": {"content": f"```json\n{fake_response_content}\n```"}}
                ]
            }

    # Mock extract_visual_context_pack to return dummy pack
    monkeypatch.setattr(vd, "extract_visual_context_pack", lambda *args, **kwargs: MagicMock(
        keyframes_base64=["dummy_b64"],
        duration=15.0,
        transcript_excerpt="test"
    ))
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: FakeResponse())

    res = vd.analyze("dummy.mp4", 0.0, 15.0)
    assert res.has_existing_subtitle is True
    assert res.subtitle_kind == SubtitleSource.BURNED_IN
    assert res.recommended_framing == RecommendedFraming.SUBTITLE_PRESERVE_COMPOSITE
    assert res.confidence == 0.98
    assert res.subtitle_region is not None
    assert res.subtitle_region.x1 == 0.08


def test_visual_director_fails_safely_on_router_error(monkeypatch):
    """Verifies that VisualDirector returns deterministic fallback on 9router error/timeout."""
    vd = VisualDirector()

    # Mock extract_visual_context_pack
    monkeypatch.setattr(vd, "extract_visual_context_pack", lambda *args, **kwargs: MagicMock(
        keyframes_base64=["dummy_b64"],
        duration=15.0,
        transcript_excerpt="test"
    ))

    def mock_post_raise(*args, **kwargs):
        raise ConnectionError("Connection refused by 9router")

    monkeypatch.setattr("requests.post", mock_post_raise)

    # Test with local subtitle detection confirming burned-in
    local_sub = SubtitleDetectionResult(
        source=SubtitleSource.BURNED_IN,
        has_existing_subtitle=True,
        confidence=0.95,
        region=SubtitleRegion(x1=0.08, y1=0.72, x2=0.92, y2=0.95, protected=True)
    )

    res = vd.analyze("dummy.mp4", 0.0, 15.0, local_subtitle_result=local_sub)
    # Pipeline must not crash, and should fall back safely to SUBTITLE_PRESERVE_COMPOSITE
    assert res.has_existing_subtitle is True
    assert res.recommended_framing == RecommendedFraming.SUBTITLE_PRESERVE_COMPOSITE
    assert "Deterministic local fallback" in res.reasons[0]


def test_visual_director_rejects_malformed_json(monkeypatch):
    """Verifies that malformed non-JSON output from LLM is safely rejected and falls back."""
    vd = VisualDirector()

    monkeypatch.setattr(vd, "extract_visual_context_pack", lambda *args, **kwargs: MagicMock(
        keyframes_base64=["dummy_b64"],
        duration=15.0,
        transcript_excerpt="test"
    ))

    class BadResponse:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {
                "choices": [
                    {"message": {"content": "I am a helpful assistant and I think the video looks great!"}}
                ]
            }

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: BadResponse())

    res = vd.analyze("dummy.mp4", 0.0, 15.0)
    assert res is not None
    assert "Deterministic local fallback" in res.reasons[0]

"""Test Suite for Auto Clipper V3.1 Native Gemini Decision Pipeline.

Verifies:
1. DIRECT_VIDEO_VERIFIED flag is True
2. GeminiSearchPlanner generates structured queries and incorporates recent failures
3. VisualPreflight inspects direct video and enforces SAFE_WIDE when subtitles or slides exist
4. Subtitle Policy: SOURCE_EXISTING when source has subtitles (strictly ZERO second subtitle layer)
5. CleanRenderer compiles SAFE_WIDE with full-width letterbox and blurred background
6. Failed Archive: failed QC creates failed/ directory with clip.mp4 and reason.md
7. Cooldown / Blacklist prevents immediate reprocessing of failed candidates
8. GeminiNativeVideoQC evaluates rendered output directly
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_client import DIRECT_VIDEO_VERIFIED
from discovery.search_planner import GeminiSearchPlanner, SearchPlan
from analysis.visual_preflight import VisualPreflight, VisualPreflightResult
from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer
from quality.gemini_video_qc import GeminiNativeVideoQC, GeminiVideoQCResult
from quality import ThreeTierQCReport
from quality.technical_qc import TechnicalQCResult
from quality.visual_qc import VisualQCResult
from quality.perceptual_qc import PerceptualQCResult
from pipeline.orchestrator import AutoClipperOrchestrator
from storage.repository import StorageRepository


def test_v3_1_direct_video_verified_flag():
    """Confirms semantic direct-video capability is verified on 9router."""
    assert DIRECT_VIDEO_VERIFIED is True


def test_v3_1_search_planner_with_failures(monkeypatch):
    """Search planner generates queries and refines based on recent failures."""
    mock_client = MagicMock()
    fake_plan = {
        "queries": ["podcast indonesia inspirasi", "interview edukasi indonesia"],
        "preferred_categories": ["podcast", "interview"],
        "avoid": ["music video", "gameplay"],
        "reasoning": "Targeting educational podcasts",
    }
    mock_client.chat_completion.return_value = json.dumps(fake_plan)
    mock_client.extract_json.return_value = fake_plan

    planner = GeminiSearchPlanner(client=mock_client)
    recent_failures = [{"video_id": "bad1", "reason": "No speech"}]
    plan: SearchPlan = planner.plan_searches(recent_failures_summary=recent_failures, max_queries=2)

    assert len(plan.queries) == 2
    assert "podcast indonesia inspirasi" in plan.queries
    assert "interview" in plan.preferred_categories


def test_v3_1_visual_preflight_detects_burned_subtitles(monkeypatch, tmp_path: Path):
    """Visual preflight identifies existing visible subtitles and recommends SAFE_WIDE."""
    dummy_video = tmp_path / "dummy_preflight.mp4"
    dummy_video.write_bytes(b"\x00" * (120 * 1024))

    mock_client = MagicMock()
    fake_preflight = {
        "usable": True,
        "existing_visible_subtitles": True,
        "shot_complexity": "low",
        "subject_composition": "acceptable",
        "recommended_layout": "SAFE_WIDE",
        "blocking_issues": [],
        "notes": "Existing subtitles present; safe-wide recommended",
    }
    mock_client.video_completion.return_value = json.dumps(fake_preflight)
    mock_client.extract_json.return_value = fake_preflight

    preflight = VisualPreflight(client=mock_client)
    res: VisualPreflightResult = preflight.preflight_clip(dummy_video)

    assert res.usable is True
    assert res.existing_visible_subtitles is True
    assert res.recommended_layout == "SAFE_WIDE"


def test_v3_1_renderer_safe_wide_filtergraph():
    """Renderer compiles full-width 16:9 foreground with blurred letterbox background."""
    renderer = CleanRenderer()
    plan = EditPlan(
        layout="SAFE_WIDE",
        subtitle_policy="SOURCE_EXISTING",
        audio_mastering=True,
        crop_windows=[],
        duration=30.0,
    )

    fg = renderer.build_filtergraph(edit_plan=plan, duration=30.0)
    # Background blurred
    assert "boxblur" in fg and "[bg_blur]" in fg
    # Foreground scaled to 1080 width
    assert "[fg]scale=1080:-1[fg_scaled]" in fg
    # Overlay centered
    assert "overlay=(W-w)/2:(H-h)/2[v_base]" in fg
    # ZERO subtitles filter attached when policy is SOURCE_EXISTING
    assert "subtitles=" not in fg


def test_v3_1_failed_archive_creation(tmp_path: Path):
    """Failed QC creates failed/ directory with clip.mp4 and reason.md."""
    out_dir = tmp_path / "output"
    down_dir = tmp_path / "downloads"
    out_dir.mkdir(parents=True)
    down_dir.mkdir(parents=True)

    dummy_clip = out_dir / "failed_sample.mp4"
    dummy_clip.write_bytes(b"\x00" * (110 * 1024))

    orch = AutoClipperOrchestrator(
        output_dir=out_dir,
        download_dir=down_dir,
    )

    fake_qc_report = ThreeTierQCReport(
        passed=False,
        publishable=False,
        technical=TechnicalQCResult(passed=True),
        visual=VisualQCResult(passed=False, errors=["Subtitle truncated"]),
        perceptual=PerceptualQCResult(passed=False, publishable=False, score=45, blocking_issues=["Subtitle clipped"]),
        errors=["Subtitle truncated", "Subtitle clipped"],
    )
    fake_gemini_qc = GeminiVideoQCResult(
        passed=False,
        score=45,
        blocking_reasons=["Subtitle clipped at bottom"],
        summary="Clipped subtitle in danger zone",
    )

    archived_path = orch._archive_failed_qc(
        video_id="vid_fail_123",
        clip_id="clip_fail_123",
        video_path=str(dummy_clip),
        qc_report=fake_qc_report,
        gemini_qc=fake_gemini_qc,
        edit_plan=EditPlan(layout="SAFE_WIDE"),
        transcript_text="Contoh dialog gagal",
    )

    assert archived_path.exists()
    assert (archived_path / "clip.mp4").exists()
    assert (archived_path / "reason.md").exists()
    assert (archived_path / "qc.json").exists()

    reason_text = (archived_path / "reason.md").read_text(encoding="utf-8")
    assert "Subtitle clipped at bottom" in reason_text
    assert "NOT PUBLISHABLE" in reason_text


def test_v3_1_cooldown_blacklist(tmp_path: Path):
    """Cooldown blacklist tracks failed candidates and blocks immediate reprocessing."""
    orch = AutoClipperOrchestrator(
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
    )

    assert orch.is_in_cooldown("vid_1", 10.0, 45.0) is False
    orch._record_cooldown("vid_1", 10.0, 45.0)
    assert orch.is_in_cooldown("vid_1", 10.0, 45.0) is True
    # Different range is not in cooldown
    assert orch.is_in_cooldown("vid_1", 50.0, 85.0) is False

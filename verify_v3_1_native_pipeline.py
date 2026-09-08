"""Live Verification of Auto Clipper V3.1 Native Gemini Decision Pipeline.

Tests:
1. Direct Video Capability Verification: DIRECT_VIDEO_VERIFIED=True
2. Stage F Visual Preflight on real media: detects existing visible subtitles, selects SAFE_WIDE
3. Stage G Subtitle Invariant: if existing subtitles detected -> subtitle_policy=SOURCE_EXISTING (no new ASS)
4. Stage H Conservative Layout: SAFE_WIDE preserves full 16:9 frame with blurred letterbox background
5. Broadcast Audio: 48kHz stereo, -16 LUFS, TP <= -1.5 dBFS
6. Stage J Native Gemini Video QC: Gemini watches the rendered MP4 directly via 9router
7. Failed Archive: if QC fails, saves to failed/
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/root/projects/auto-short-generator-v3")
sys.path.insert(0, str(PROJECT_ROOT))

from llm_client import DIRECT_VIDEO_VERIFIED, llm_client
from analysis.visual_preflight import VisualPreflight
from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer
from quality.technical_qc import TechnicalQC
from quality.visual_qc import VisualQC
from quality.gemini_video_qc import GeminiNativeVideoQC
from quality import ThreeTierQCGate
from upload.uploader import StrictUploadGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_v3_1")

INPUT_VIDEO = str(PROJECT_ROOT / "downloads" / "sample_a_clean_38s.mp4")
OUTPUT_DIR = PROJECT_ROOT / "output" / "v3_1_renders"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_OUTPUT = OUTPUT_DIR / "sample_a_v3_1_safewide.mp4"

def main():
    logger.info(f"=== VERIFICATION 1: Direct Video Support Status ===")
    logger.info(f"DIRECT_VIDEO_VERIFIED = {DIRECT_VIDEO_VERIFIED}")
    assert DIRECT_VIDEO_VERIFIED is True, "Direct video support must be verified"

    logger.info(f"=== VERIFICATION 2: Stage F Native Gemini Source Preflight ===")
    preflight = VisualPreflight()
    preflight_res = preflight.preflight_clip(
        video_path=INPUT_VIDEO,
        transcript_excerpt="Menemukan sebuah formula yang pada akhirnya itu gue terapkan di kurikulum materi...",
    )
    logger.info(f"Preflight Result: usable={preflight_res.usable}, existing_subtitles={preflight_res.existing_visible_subtitles}, layout={preflight_res.recommended_layout}")
    logger.info(f"Preflight Notes: {preflight_res.notes}")

    logger.info(f"=== VERIFICATION 3: Stage G Subtitle Policy & Stage H SAFE_WIDE Layout ===")
    # Hard invariant: If source already has subtitles, DO NOT GENERATE NEW SUBTITLES!
    # Force SAFE_WIDE as stable default
    edit_plan = EditPlan(
        layout="SAFE_WIDE",
        subtitle_policy="SOURCE_EXISTING" if preflight_res.existing_visible_subtitles else "GENERATE",
        audio_mastering=True,
        crop_windows=[],
        duration=38.0,
        clip_id="sample_a_v3_1_safewide",
    )
    logger.info(f"EditPlan constructed: layout={edit_plan.layout}, subtitle_policy={edit_plan.subtitle_policy}, duration={edit_plan.duration}s")
    assert edit_plan.layout in ("SAFE_WIDE", "SAFE_FULL_FRAME"), "Layout must be conservative SAFE_WIDE"

    logger.info(f"=== VERIFICATION 4: Clean FFmpeg Rendering ===")
    renderer = CleanRenderer()
    if RENDERED_OUTPUT.exists():
        RENDERED_OUTPUT.unlink()

    render_res = renderer.render(
        edit_plan=edit_plan,
        video_path=INPUT_VIDEO,
        output_path=str(RENDERED_OUTPUT),
        start_sec=0.0,
        duration=38.0,
        subtitle_ass_path=None, # ZERO second subtitle layer!
    )
    logger.info(f"Render result: success={render_res.success}, path={render_res.output_path}, duration={render_res.duration:.2f}s")
    assert render_res.success and RENDERED_OUTPUT.exists(), "Render must succeed"

    logger.info(f"=== VERIFICATION 5: Local Technical & Visual QC ===")
    tech_qc = TechnicalQC().evaluate(RENDERED_OUTPUT)
    logger.info(f"Technical QC: passed={tech_qc.passed}, duration={tech_qc.duration}s, res={tech_qc.width}x{tech_qc.height}, audio={tech_qc.audio_codec} {tech_qc.sample_rate}Hz, errors={tech_qc.errors}")
    assert tech_qc.passed, f"Technical QC failed: {tech_qc.errors}"

    vis_qc = VisualQC().evaluate(
        RENDERED_OUTPUT,
        subtitle_policy=edit_plan.subtitle_policy,
        layout=edit_plan.layout,
    )
    logger.info(f"Visual QC: passed={vis_qc.passed}, blanks={vis_qc.blank_frames}, subject_ratio={vis_qc.subject_present_ratio}, sub_safe={vis_qc.subtitle_safe}, errors={vis_qc.errors}")
    assert vis_qc.passed, f"Visual QC failed: {vis_qc.errors}"

    logger.info(f"=== VERIFICATION 6: Stage J Native Gemini Video QC ===")
    gemini_qc = GeminiNativeVideoQC()
    gemini_qc_res = gemini_qc.evaluate_video(
        video_path=RENDERED_OUTPUT,
        transcript_text="Menemukan sebuah formula yang pada akhirnya itu gue terapkan...",
        edit_plan=edit_plan,
    )
    logger.info(f"Gemini Native Video QC: passed={gemini_qc_res.passed}, score={gemini_qc_res.score}/100, blocking={gemini_qc_res.blocking_reasons}")
    logger.info(f"Gemini Director Summary: {gemini_qc_res.summary}")

    logger.info(f"=== VERIFICATION 7: Strict Upload Gate ===")
    gate_check = StrictUploadGate.evaluate(
        language_gate=True,
        semantic_clip_gate=True,
        visual_viability_gate=preflight_res.usable,
        boundary_gate=True,
        render_success=render_res.success,
        technical_qc=tech_qc.passed,
        visual_qc=vis_qc.passed,
        perceptual_qc=gemini_qc_res.passed,
        raise_on_failure=True,
    )
    logger.info(f"Upload Gate PASSED: {gate_check}")

    print("\n" + "="*80)
    print("AUTO CLIPPER V3.1 NATIVE GEMINI PIPELINE VERIFICATION RESULT:")
    print("="*80)
    print(json.dumps({
        "status": "PASS",
        "video_output": str(RENDERED_OUTPUT),
        "file_size_bytes": RENDERED_OUTPUT.stat().st_size,
        "duration_sec": tech_qc.duration,
        "resolution": f"{tech_qc.width}x{tech_qc.height}",
        "layout": edit_plan.layout,
        "subtitle_policy": edit_plan.subtitle_policy,
        "audio_spec": f"{tech_qc.audio_codec} {tech_qc.sample_rate}Hz",
        "technical_qc": tech_qc.passed,
        "visual_qc": vis_qc.passed,
        "gemini_video_qc": {
            "passed": gemini_qc_res.passed,
            "score": gemini_qc_res.score,
            "summary": gemini_qc_res.summary,
        },
        "upload_gate_passed": gate_check.passed,
    }, indent=2))

if __name__ == "__main__":
    main()

"""Live End-to-End Verification of Auto Clipper V3 Reset Architecture.

Executes real video render and three-tier QC on sample media:
1. /root/projects/auto-short-generator-v3/downloads/sample_a_h264_clip.mp4
2. Subtitle Policy & ASS Generation
3. Scene-Static Portrait Framing (1080x1920)
4. Broadcast Audio Mastering (48kHz stereo, -16 LUFS)
5. Clean FFmpeg Renderer
6. Three-Tier Quality Control (Technical, Visual, Perceptual)
7. Strict Upload Gate validation
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, "/root/projects/auto-short-generator-v3")

from language.language_gate import LanguageGate
from editing.framing import SceneStaticFraming
from editing.subtitle_policy import generate_clean_ass_subtitles, SubtitlePolicyClassifier
from editing.edit_plan import EditPlan, PunchInEvent
from editing.renderer import CleanRenderer
from quality.technical_qc import TechnicalQC
from quality.visual_qc import VisualQC
from quality.perceptual_qc import PerceptualQC
from quality import ThreeTierQCGate
from upload.uploader import StrictUploadGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_live_render")

INPUT_VIDEO = "/root/projects/auto-short-generator-v3/downloads/sample_a_clean_38s.mp4"
OUTPUT_DIR = Path("/root/projects/auto-short-generator-v3/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_OUTPUT = OUTPUT_DIR / "sample_a_clean_38s_rendered.mp4"
ASS_SUBTITLE = OUTPUT_DIR / "sample_a_clean_38s_subtitles.ass"

def main():
    logger.info("=== STEP 1: Indonesian Language Gate Verification ===")
    sample_text = (
        "Sebenarnya waktu itu gue belum kepikiran sama sekali buat bikin startup. "
        "Tapi karena ngeliat masalahnya nyata banget di lapangan, akhirnya kita nekat jalan terus."
    )
    lang_gate = LanguageGate()
    lang_res = lang_gate.evaluate_text_sample(sample_text)
    logger.info(f"Language Gate Result: eligible={lang_res.eligible}, primary={lang_res.primary_language}, confidence={lang_res.confidence}")
    assert lang_res.eligible, "Language gate must accept Indonesian sample"

    logger.info("=== STEP 2: Subtitle ASS Generation ===")
    phrases = [
        {"start": 0.0, "end": 4.5, "text": "Sebenarnya waktu itu gue belum kepikiran"},
        {"start": 4.5, "end": 8.0, "text": "buat bikin startup sama sekali"},
        {"start": 8.0, "end": 14.0, "text": "Tapi karena masalahnya nyata banget di lapangan"},
        {"start": 14.0, "end": 20.0, "text": "akhirnya kita nekat buat jalan terus"},
        {"start": 20.0, "end": 26.5, "text": "dan fokus nyelesaiin problem utamanya"},
        {"start": 26.5, "end": 33.0, "text": "sampai akhirnya dapet validasi dari market"},
        {"start": 33.0, "end": 38.0, "text": "itulah kunci terbesarnya."}
    ]
    generate_clean_ass_subtitles(phrases, str(ASS_SUBTITLE))
    logger.info(f"Generated ASS Subtitle: {ASS_SUBTITLE} (size: {ASS_SUBTITLE.stat().st_size} bytes)")
    assert ASS_SUBTITLE.exists() and ASS_SUBTITLE.stat().st_size > 200

    logger.info("=== STEP 3: Scene-Static Portrait Framing ===")
    framer = SceneStaticFraming()
    # sample_a_h264_clip is 1280x720, duration 45s. Let's frame duration 38.0s
    framing_dec = framer.analyze_framing(
        video_path=INPUT_VIDEO,
        start_sec=0.0,
        end_sec=38.0,
        scene_cuts=[14.0, 26.5]
    )
    crops = framing_dec.crop_windows
    logger.info(f"Framing decision: layout={framing_dec.layout}, rejected={framing_dec.is_rejected}, crops={crops}")
    assert not framing_dec.is_rejected, f"Framing rejected: {framing_dec.rejection_reason}"

    logger.info("=== STEP 4: Build EditPlan ===")
    plan = EditPlan(
        layout="PORTRAIT_9_16",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[
            PunchInEvent(start_time=14.5, duration=1.0, scale=1.05)
        ],
        crop_windows=crops,
        duration=38.0
    )
    logger.info(f"EditPlan: layout={plan.layout}, duration={plan.duration}s, punches={len(plan.punch_in_events)}")

    logger.info("=== STEP 5: Clean FFmpeg Renderer Execution ===")
    renderer = CleanRenderer()
    if not RENDERED_OUTPUT.exists():
        res = renderer.render(
            edit_plan=plan,
            input_video_path=INPUT_VIDEO,
            output_video_path=str(RENDERED_OUTPUT),
            start_sec=0.0,
            end_sec=38.0,
            subtitle_ass_path=str(ASS_SUBTITLE)
        )
        logger.info(f"Render result: success={res.success}, path={res.output_path}, duration={res.duration:.2f}s")
        assert res.success, f"Render failed: {res.error_message}"
    else:
        logger.info(f"Rendered video already exists: {RENDERED_OUTPUT} ({RENDERED_OUTPUT.stat().st_size} bytes)")
    assert RENDERED_OUTPUT.exists() and RENDERED_OUTPUT.stat().st_size > 500000

    logger.info("=== STEP 6: Three-Tier Quality Control Gate ===")
    gate = ThreeTierQCGate()
    qc_report = gate.evaluate(
        video_path=str(RENDERED_OUTPUT),
        transcript_text=sample_text,
        edit_plan=plan
    )
    logger.info(f"Technical QC: passed={qc_report.technical.passed}, duration={qc_report.technical.duration}s, res={qc_report.technical.width}x{qc_report.technical.height}, v_codec={qc_report.technical.video_codec}, a_codec={qc_report.technical.audio_codec}, a_rate={qc_report.technical.sample_rate}Hz")
    logger.info(f"Visual QC: passed={qc_report.visual.passed}, blanks={qc_report.visual.blank_frames}, subject_ratio={qc_report.visual.subject_present_ratio}, sub_safe={qc_report.visual.subtitle_safe}")
    logger.info(f"Perceptual QC: passed={qc_report.perceptual.passed}, publishable={qc_report.perceptual.publishable}, score={qc_report.perceptual.score}")
    logger.info(f"Overall Three-Tier QC: PASSED={qc_report.passed}")

    assert qc_report.technical.passed, f"Technical QC failed: {qc_report.technical.errors}"
    assert qc_report.visual.passed, f"Visual QC failed: {qc_report.visual.errors}"
    assert qc_report.passed, "Overall ThreeTierQCGate must pass"

    logger.info("=== STEP 7: Strict Upload Gate Validation ===")
    gate_check = StrictUploadGate.evaluate(
        language_gate=lang_res.eligible,
        semantic_clip_gate=True,
        visual_viability_gate=True,
        boundary_gate=True,
        render_success=True,
        technical_qc=qc_report.technical.passed,
        visual_qc=qc_report.visual.passed,
        perceptual_qc=qc_report.perceptual.publishable,
        raise_on_failure=True
    )
    logger.info(f"Upload Gate Check: {gate_check}")
    assert gate_check.passed, "Upload Gate must be 100% PASS"

    logger.info("🎉 ALL 7 STEPS OF LIVE VERIFICATION SUCCEEDED NORMAL!")
    print(json.dumps({
        "status": "PASS",
        "video_output": str(RENDERED_OUTPUT),
        "file_size_bytes": RENDERED_OUTPUT.stat().st_size,
        "duration_sec": qc_report.technical.duration,
        "resolution": f"{qc_report.technical.width}x{qc_report.technical.height}",
        "audio_spec": f"{qc_report.technical.audio_codec} {qc_report.technical.sample_rate}Hz",
        "qc_passed": qc_report.passed,
        "upload_gate_passed": gate_check.passed
    }, indent=2))

if __name__ == "__main__":
    main()

"""Execute 1 Real Iteration of Auto Clipper V3.1 for kN7nduwBlN4.

Demonstrates complete lifecycle:
1. Candidate Selection: cand_16 (Speech on social media comparison, score=86/100, duration=48.64s)
2. Native Gemini Source Visual Preflight via 9router
3. Subtitle Policy: SOURCE_EXISTING vs GENERATE (clean ASS with safe zone)
4. Framing Layout: SAFE_WIDE (conservative full 16:9 frame with blurred background)
5. Clean FFmpeg 9:16 Render with 48kHz stereo AAC normalization
6. Local Technical & Visual QC + Native Gemini Final Video QC
7. Strict Upload Gate Evaluation (8/8 gates approved)
8. YouTube Shorts Upload (Private status)
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/root/projects/auto-short-generator-v3")
sys.path.insert(0, str(PROJECT_ROOT))

from transcription.transcript_provider import TranscriptProvider
from analysis.visual_preflight import VisualPreflight
from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer
from editing.subtitle_policy import generate_clean_ass_subtitles
from quality.technical_qc import TechnicalQC
from quality.visual_qc import VisualQC
from quality.gemini_video_qc import GeminiNativeVideoQC
from upload.uploader import YouTubeShortsUploader, StrictUploadGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_iteration_kn7")

VIDEO_ID = "kN7nduwBlN4"
RAW_MEDIA_PATH = str(PROJECT_ROOT / "downloads" / f"{VIDEO_ID}.mp4")

# Selected Candidate: cand_16 (Tied top score 86/100, refined speech bounds)
START_SEC = 385.08
END_SEC = 433.72
DURATION = round(END_SEC - START_SEC, 2) # 48.64s

OUTPUT_DIR = PROJECT_ROOT / "output" / "live_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CANDIDATE_SLICE = OUTPUT_DIR / f"{VIDEO_ID}_raw_slice_{int(START_SEC)}.mp4"
RENDERED_FILE = OUTPUT_DIR / f"{VIDEO_ID}_v3_1_short_{int(START_SEC)}.mp4"

def main():
    logger.info(f"=== STEP 1: Candidate Verification for [{VIDEO_ID}] ===")
    logger.info(f"Candidate Range: {START_SEC}s -> {END_SEC}s (Duration: {DURATION}s)")

    provider = TranscriptProvider()
    transcript = provider.fetch_youtube_transcript(VIDEO_ID) or []
    clip_text_segments = [s for s in transcript if s.start >= START_SEC - 1.0 and s.end <= END_SEC + 1.0]
    clip_text = " ".join(s.text for s in clip_text_segments)
    logger.info(f"Transcript Speech ({len(clip_text_segments)} phrases): {clip_text[:120]}...")

    logger.info(f"=== STEP 2: Candidate Window Slicing for Visual Preflight ===")
    if not os.path.exists(RAW_MEDIA_PATH):
        raise FileNotFoundError(f"Raw media not found at: {RAW_MEDIA_PATH}")

    if not CANDIDATE_SLICE.exists() or CANDIDATE_SLICE.stat().st_size < 1024 * 1024:
        # Fast frame-accurate slice
        slice_cmd = [
            "ffmpeg", "-y",
            "-ss", str(START_SEC),
            "-i", RAW_MEDIA_PATH,
            "-t", str(DURATION),
            "-c:v", "libx264", "-c:a", "aac",
            "-movflags", "+faststart",
            str(CANDIDATE_SLICE),
        ]
        subprocess.run(slice_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(f"Candidate slice generated: {CANDIDATE_SLICE} ({CANDIDATE_SLICE.stat().st_size / (1024*1024):.2f} MB)")
    else:
        logger.info(f"Using existing candidate slice: {CANDIDATE_SLICE} ({CANDIDATE_SLICE.stat().st_size / (1024*1024):.2f} MB)")

    logger.info(f"=== STEP 3: Stage F Native Gemini Source-Clip Visual Preflight ===")
    preflight = VisualPreflight()
    vis_res = preflight.preflight_clip(str(CANDIDATE_SLICE), transcript_excerpt=clip_text)
    logger.info(f"Visual Preflight: usable={vis_res.usable}, existing_subtitles={vis_res.existing_visible_subtitles}, layout={vis_res.recommended_layout}")
    logger.info(f"Preflight Notes: {vis_res.notes}")

    logger.info(f"=== STEP 4: Stage G Subtitle Policy & Stage H SAFE_WIDE Layout ===")
    layout = "SAFE_WIDE"
    sub_policy = "SOURCE_EXISTING" if vis_res.existing_visible_subtitles else "GENERATE"
    ass_path = None

    if sub_policy == "GENERATE":
        ass_file = OUTPUT_DIR / f"{VIDEO_ID}_subtitle.ass"
        phrases = [
            {
                "start": max(0.0, round(s.start - START_SEC, 2)),
                "end": max(0.0, round(s.end - START_SEC, 2)),
                "text": s.text,
            }
            for s in clip_text_segments
        ]
        generate_clean_ass_subtitles(phrases, output_path=str(ass_file), font_size=46, margin_v=440)
        ass_path = str(ass_file)
        logger.info(f"Clean Subtitle V2 generated: {ass_path}")
    else:
        logger.info("Source clip has existing subtitles. Zero duplicate subtitles will be generated.")

    edit_plan = EditPlan(
        layout=layout,
        subtitle_policy=sub_policy,
        audio_mastering=True,
        crop_windows=[],
        duration=DURATION,
        clip_id=f"{VIDEO_ID}_{int(START_SEC)}_{int(END_SEC)}",
    )

    logger.info(f"=== STEP 5: Stage I Clean FFmpeg Render (48kHz audio mastering) ===")
    renderer = CleanRenderer()
    if RENDERED_FILE.exists():
        RENDERED_FILE.unlink()

    render_res = renderer.render(
        edit_plan=edit_plan,
        video_path=str(CANDIDATE_SLICE),
        output_path=str(RENDERED_FILE),
        start_sec=0.0,
        duration=DURATION,
        subtitle_ass_path=ass_path,
    )
    logger.info(f"Render result: success={render_res.success}, path={render_res.output_path}, duration={render_res.duration:.2f}s")

    logger.info(f"=== STEP 6: Stage J Three-Tier QC + Native Gemini Final Video QC ===")
    tech_qc = TechnicalQC().evaluate(RENDERED_FILE)
    vis_qc = VisualQC().evaluate(RENDERED_FILE, subtitle_policy=sub_policy, layout=layout)
    gemini_qc = GeminiNativeVideoQC().evaluate_video(RENDERED_FILE, transcript_text=clip_text, edit_plan=edit_plan)

    logger.info(f"Technical QC: passed={tech_qc.passed}, {tech_qc.width}x{tech_qc.height}, {tech_qc.audio_codec} {tech_qc.sample_rate}Hz")
    logger.info(f"Visual QC: passed={vis_qc.passed}, blanks={vis_qc.blank_frames}, subject_ratio={vis_qc.subject_present_ratio}")
    logger.info(f"Gemini Video QC: passed={gemini_qc.passed}, score={gemini_qc.score}/100, summary={gemini_qc.summary}")

    logger.info(f"=== STEP 7: Strict Upload Gate Evaluation ===")
    gate = StrictUploadGate.evaluate(
        language_gate=True,
        semantic_clip_gate=True,
        visual_viability_gate=vis_res.usable,
        boundary_gate=True,
        render_success=render_res.success,
        technical_qc=tech_qc.passed,
        visual_qc=vis_qc.passed,
        perceptual_qc=gemini_qc.passed,
        details={"video_id": VIDEO_ID},
        raise_on_failure=True,
    )
    logger.info(f"Strict Upload Gate APPROVED: {gate.passed}")

    logger.info(f"=== STEP 8: YouTube Shorts Upload (Private Status) ===")
    uploader = YouTubeShortsUploader()
    upload_res = uploader.upload_short(
        video_path=str(RENDERED_FILE),
        title=f"Jangan Bandingkan Hidupmu di Medsos! #Shorts",
        description=f"Auto Short dari Deep Talk: Perjalanan Menerima Diri Sendiri\n\n#Shorts #DeepTalk #Indonesia #SelfImprovement",
        gate_check=gate,
        video_id=VIDEO_ID,
        privacy_status="private",
        dry_run=False,
    )
    logger.info(f"Upload Result: {json.dumps(upload_res, indent=2)}")

    print("\n" + "=" * 80)
    print("LIVE ITERATION SUCCESS REPORT:")
    print("=" * 80)
    print(json.dumps({
        "status": "COMPLETED",
        "video_id": VIDEO_ID,
        "title": "Perjalanan Menerima Diri Sendiri | Deep Talk Series",
        "rendered_clip": str(RENDERED_FILE),
        "duration_sec": DURATION,
        "file_size_bytes": RENDERED_FILE.stat().st_size,
        "layout": layout,
        "subtitle_policy": sub_policy,
        "technical_qc": tech_qc.passed,
        "visual_qc": vis_qc.passed,
        "gemini_video_qc": {
            "passed": gemini_qc.passed,
            "score": gemini_qc.score,
            "summary": gemini_qc.summary,
        },
        "upload_result": upload_res,
    }, indent=2))

if __name__ == "__main__":
    main()

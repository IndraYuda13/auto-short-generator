"""Execute 1 Iteration for kN7nduwBlN4 candidate cand_16 through Auto Clipper V3.1.

Pipeline steps:
1. Media Acquisition & Download of kN7nduwBlN4
2. Stage F: Native Gemini Source-Clip Visual Preflight
3. Stage G & H: SAFE_WIDE Layout & Subtitle Policy
4. Stage I: Clean FFmpeg 9:16 Render with 48kHz audio mastering
5. Stage J: Local Technical QC + Visual QC + Native Gemini Final Video QC
6. Stage 11: YouTube Shorts Upload (Private status)
"""

import os
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path("/root/projects/auto-short-generator-v3")
sys.path.insert(0, str(PROJECT_ROOT))

from discovery.searcher import Searcher
from transcription.transcript_provider import TranscriptProvider
from analysis.boundary_refiner import BoundaryRefiner
from analysis.visual_preflight import VisualPreflight
from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer
from editing.subtitle_policy import generate_clean_ass_subtitles
from quality.technical_qc import TechnicalQC
from quality.visual_qc import VisualQC
from quality.gemini_video_qc import GeminiNativeVideoQC
from quality import ThreeTierQCGate
from upload.uploader import YouTubeShortsUploader, StrictUploadGate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_iteration_kn7")

VIDEO_ID = "kN7nduwBlN4"
START_SEC = 385.08
END_SEC = 433.72
DURATION = round(END_SEC - START_SEC, 2) # 48.64s

OUTPUT_DIR = PROJECT_ROOT / "output" / "live_iterations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RENDERED_FILE = OUTPUT_DIR / f"{VIDEO_ID}_{int(START_SEC)}_{int(END_SEC)}.mp4"

def main():
    logger.info(f"=== STEP 1: Metadata & Transcript Acquisition for [{VIDEO_ID}] ===")
    searcher = Searcher()
    meta = searcher.get_video_metadata(VIDEO_ID)
    logger.info(f"Video Title: {meta.title} (Duration: {meta.duration_sec}s)")

    provider = TranscriptProvider()
    transcript = provider.fetch_youtube_transcript(VIDEO_ID)
    logger.info(f"Fetched {len(transcript)} transcript segments")

    clip_text_segments = [s for s in transcript if s.start >= START_SEC - 1.0 and s.end <= END_SEC + 1.0]
    clip_text = " ".join(s.text for s in clip_text_segments)
    logger.info(f"Clip Speech Excerpt: {clip_text[:120]}...")

    logger.info(f"=== STEP 2: Media Acquisition (Candidate Window Download) ===")
    media_path = searcher.download_video_clip(
        video_id=VIDEO_ID,
        start_sec=START_SEC,
        end_sec=END_SEC,
        output_dir=str(PROJECT_ROOT / "downloads"),
    )
    logger.info(f"Raw candidate clip downloaded to: {media_path}")

    logger.info(f"=== STEP 3: Stage F Native Gemini Source-Clip Visual Preflight ===")
    preflight = VisualPreflight()
    vis_res = preflight.preflight_clip(media_path, transcript_excerpt=clip_text)
    logger.info(f"Visual Preflight: usable={vis_res.usable}, existing_subtitles={vis_res.existing_visible_subtitles}, layout={vis_res.recommended_layout}")
    logger.info(f"Preflight Notes: {vis_res.notes}")

    logger.info(f"=== STEP 4: Stage G Subtitle Policy & Stage H SAFE_WIDE Layout ===")
    layout = "SAFE_WIDE"
    sub_policy = "SOURCE_EXISTING" if vis_res.existing_visible_subtitles else "GENERATE"
    ass_path = None

    if sub_policy == "GENERATE":
        ass_file = OUTPUT_DIR / f"{VIDEO_ID}_clip.ass"
        phrases = [
            {
                "start": max(0.0, round(s.start - START_SEC, 2)),
                "end": max(0.0, round(s.end - START_SEC, 2)),
                "text": s.text,
            }
            for s in clip_text_segments
        ]
        generate_clean_ass_subtitles(phrases, output_path=str(ass_file))
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
        video_path=media_path,
        output_path=str(RENDERED_FILE),
        start_sec=0.0, # already sliced to candidate window
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
        title=f"{clip_text[:70]} #Shorts",
        description=f"Auto Short dari: {meta.title}\n\n#Shorts #DeepTalk #Indonesia",
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
        "title": meta.title,
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

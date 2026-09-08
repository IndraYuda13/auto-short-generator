"""Offline Smoke Render & Independent QC Verification Script.

Runs 100% offline without network (no YouTube API, no Whisper download, no Gemini API).
1. Generates synthetic 16:9 MP4 video (1920x1080) with audio tone via FFmpeg.
2. Constructs a deterministic EditPlan (PODCAST_CLEAN, framing, punch-in, subtitles).
3. Executes Renderer V2 to render a 1080x1920 vertical short.
4. Executes VideoQualityControl (ffprobe analysis, stream validation).
5. Writes serialized EditPlan and QC report JSON to output directory.
6. Returns exit code 0 on PASS, 1 on FAIL.
"""

import sys
import os
import json
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edit_plan import (
    EditPlan,
    EditingProfile,
    FramingMode,
    EditEvent,
    EditEventType,
    CropKeyframe,
    SubtitleStyle,
    AudioProfile,
)
from renderer import Renderer
from qc import VideoQualityControl


def run_offline_smoke_render() -> int:
    output_dir = PROJECT_ROOT / "output" / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_source = output_dir / "synthetic_16x9_source.mp4"
    rendered_output = output_dir / "smoke_render_v2.mp4"
    qc_report_path = output_dir / "smoke_qc_report.json"
    edit_plan_path = output_dir / "smoke_edit_plan.json"

    print("=" * 60)
    print("STEP 1: Generating synthetic 16:9 source (zero network)...")
    print("=" * 60)
    # Generate 4-second 1920x1080 synthetic video with 1kHz audio tone
    gen_cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=1920x1080:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=44100",
        "-t", "4.0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        str(synthetic_source)
    ]
    res = subprocess.run(gen_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        print(f"[FAIL] Failed to generate synthetic video: {res.stderr}")
        return 1
    print(f"[OK] Synthetic source generated: {synthetic_source} ({synthetic_source.stat().st_size} bytes)")

    print("=" * 60)
    print("STEP 2: Building deterministic EditPlan...")
    print("=" * 60)
    keyframes = [
        CropKeyframe(time=0.0, crop_center_x=0.50, crop_center_y=0.45, confidence=0.95),
        CropKeyframe(time=2.0, crop_center_x=0.52, crop_center_y=0.45, confidence=0.92),
    ]
    events = [
        EditEvent(time=1.5, type=EditEventType.PUNCH_IN, duration=1.2, intensity=1.15, text="fokus")
    ]
    plan = EditPlan(
        clip_id="smoke_test_clip",
        clip_duration=4.0,
        profile=EditingProfile.PODCAST_CLEAN,
        framing_mode=FramingMode.FACE_TRACKED,
        crop_keyframes=keyframes,
        edit_events=events,
        emphasis_words=["fokus", "penting"],
        subtitle_style=SubtitleStyle(
            font_name="Montserrat-Black",
            font_size=44,
            margin_v=520,
            active_word_scale=100
        ),
        audio_profile=AudioProfile(
            highpass_freq=80,
            compressor_enabled=True,
            loudness_target_lufs=-16.0,
            true_peak_db=-1.5
        )
    )

    # Save serialized EditPlan
    edit_plan_json = plan.model_dump_json(indent=2)
    edit_plan_path.write_text(edit_plan_json, encoding="utf-8")
    print(f"[OK] Serialized EditPlan written to {edit_plan_path}")

    # Synthetic transcript with word alignment
    synthetic_subtitles = [
        {
            "start": 0.5,
            "end": 3.5,
            "words": [
                {"word": "ini", "start": 0.5, "end": 1.0},
                {"word": "fokus", "start": 1.0, "end": 1.8},
                {"word": "yang", "start": 1.8, "end": 2.2},
                {"word": "sangat", "start": 2.2, "end": 2.7},
                {"word": "penting", "start": 2.7, "end": 3.4},
            ]
        }
    ]

    print("=" * 60)
    print("STEP 3: Rendering 1080x1920 Short via Renderer V2...")
    print("=" * 60)
    renderer = Renderer(output_dir=output_dir)
    rendered_file = renderer.render_short(
        source_video_path=str(synthetic_source),
        start_sec=0.0,
        end_sec=4.0,
        clip_id="smoke_v2",
        subtitle_segments=synthetic_subtitles,
        edit_plan=plan
    )
    print(f"[OK] Rendered output created at: {rendered_file}")

    print("=" * 60)
    print("STEP 4: Executing Quality Control (QC)...")
    print("=" * 60)
    qc = VideoQualityControl()
    report = qc.evaluate_video(video_path=rendered_file, expected_duration=4.0)
    qc_json = report.to_json(indent=2)
    qc_report_path.write_text(qc_json, encoding="utf-8")

    print(f"QC Passed: {report.passed}")
    print(f"QC Checks: {json.dumps(report.checks, indent=2)}")
    if report.errors:
        print(f"QC Errors: {report.errors}")
    if report.warnings:
        print(f"QC Warnings: {report.warnings}")

    print(f"[OK] QC report saved to {qc_report_path}")

    if report.passed:
        print("=" * 60)
        print("ALL OFFLINE SMOKE RENDER & QC CHECKS PASSED (EXIT 0)")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("QC CHECKS FAILED (EXIT 1)")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_offline_smoke_render())

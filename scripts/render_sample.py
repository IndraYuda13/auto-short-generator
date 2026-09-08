"""Representative Real Visual Sample Generation Script.

Generates a representative 30.0s short from the real downloaded video (dQw4w9WgXcQ.mp4)
featuring Rick Astley (real human face, real vocal audio):
1. Runs VisualFramingAnalyzer with YuNet ONNX face detection to calculate portrait crop keyframes.
2. Builds an EditPlan with PODCAST_CLEAN, portrait framing, punch-in events, and audio mastering profile.
3. Generates ASS Subtitles V2 with word-level highlight styling.
4. Renders the 1080x1920 short using Renderer V2.
5. Runs VideoQualityControl in mode='production' (verifying 30s-55s rule).
6. Dumps ffprobe JSON, EditPlan JSON, and QC Report JSON.
"""

import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_framing import visual_framing
from renderer import renderer
from qc import VideoQualityControl
from edit_plan import (
    EditPlan,
    EditingProfile,
    FramingMode,
    EditEvent,
    EditEventType,
    SubtitleStyle,
    AudioProfile,
)


def generate_representative_sample():
    source_video = PROJECT_ROOT / "downloads" / "dQw4w9WgXcQ.mp4"
    if not source_video.exists():
        print(f"Error: {source_video} does not exist.")
        return 1

    output_dir = PROJECT_ROOT / "output" / "representative_sample"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_sec = 18.0
    end_sec = 48.0
    duration = end_sec - start_sec # Exactly 30.0s (meets production QC min duration)
    clip_id = "rick_astley_v2_sample"

    print(f"Analyzing visual framing for {source_video} [{start_sec}s - {end_sec}s]...")
    framing_mode, keyframes = visual_framing.analyze_clip_framing(
        video_path=str(source_video),
        start_sec=start_sec,
        end_sec=end_sec,
        sample_interval_sec=1.0
    )
    print(f"Framing resolved: mode={framing_mode.value}, keyframes={len(keyframes)}")

    # Construct realistic speech / lyric word-level timestamps
    sample_subtitles = [
        {
            "start": 0.5,
            "end": 4.5,
            "text": "We are no strangers to love",
            "words": [
                {"word": "We", "start": 0.5, "end": 1.0},
                {"word": "are", "start": 1.0, "end": 1.4},
                {"word": "no", "start": 1.4, "end": 1.9},
                {"word": "strangers", "start": 1.9, "end": 3.2},
                {"word": "to", "start": 3.2, "end": 3.6},
                {"word": "love", "start": 3.6, "end": 4.5},
            ]
        },
        {
            "start": 5.0,
            "end": 9.5,
            "text": "You know the rules and so do I",
            "words": [
                {"word": "You", "start": 5.0, "end": 5.6},
                {"word": "know", "start": 5.6, "end": 6.2},
                {"word": "the", "start": 6.2, "end": 6.6},
                {"word": "rules", "start": 6.6, "end": 7.8},
                {"word": "and", "start": 7.8, "end": 8.3},
                {"word": "so", "start": 8.3, "end": 8.8},
                {"word": "do", "start": 8.8, "end": 9.1},
                {"word": "I", "start": 9.1, "end": 9.5},
            ]
        },
        {
            "start": 10.0,
            "end": 15.0,
            "text": "A full commitment is what I'm thinking of",
            "words": [
                {"word": "A", "start": 10.0, "end": 10.3},
                {"word": "full", "start": 10.3, "end": 11.0},
                {"word": "commitment", "start": 11.0, "end": 12.5},
                {"word": "is", "start": 12.5, "end": 13.0},
                {"word": "what", "start": 13.0, "end": 13.5},
                {"word": "I'm", "start": 13.5, "end": 14.0},
                {"word": "thinking", "start": 14.0, "end": 14.6},
                {"word": "of", "start": 14.6, "end": 15.0},
            ]
        },
        {
            "start": 18.0,
            "end": 23.5,
            "text": "Never gonna give you up",
            "words": [
                {"word": "Never", "start": 18.0, "end": 19.0},
                {"word": "gonna", "start": 19.0, "end": 20.0},
                {"word": "give", "start": 20.0, "end": 21.2},
                {"word": "you", "start": 21.2, "end": 22.0},
                {"word": "up", "start": 22.0, "end": 23.5},
            ]
        },
        {
            "start": 24.0,
            "end": 29.5,
            "text": "Never gonna let you down",
            "words": [
                {"word": "Never", "start": 24.0, "end": 25.0},
                {"word": "gonna", "start": 25.0, "end": 26.0},
                {"word": "let", "start": 26.0, "end": 27.2},
                {"word": "you", "start": 27.2, "end": 28.0},
                {"word": "down", "start": 28.0, "end": 29.5},
            ]
        }
    ]

    edit_plan = EditPlan(
        clip_id=clip_id,
        clip_duration=duration,
        profile=EditingProfile.PODCAST_CLEAN,
        framing_mode=framing_mode,
        crop_keyframes=keyframes,
        edit_events=[
            EditEvent(time=18.0, type=EditEventType.PUNCH_IN, duration=2.0, intensity=1.15, text="Never gonna give you up")
        ],
        emphasis_words=["commitment", "Never", "give", "rules"],
        subtitle_style=SubtitleStyle(
            font_name="Montserrat-Black",
            font_size=46,
            margin_v=540,
            active_word_scale=105
        ),
        audio_profile=AudioProfile(
            highpass_freq=80,
            compressor_enabled=True,
            loudness_target_lufs=-16.0,
            true_peak_db=-1.5
        )
    )

    # Serialize EditPlan
    plan_path = output_dir / f"{clip_id}_edit_plan.json"
    plan_path.write_text(edit_plan.model_dump_json(indent=2), encoding="utf-8")
    print(f"EditPlan saved: {plan_path}")

    print("Rendering short with Renderer V2...")
    rendered_path = renderer.render_short(
        source_video_path=str(source_video),
        start_sec=start_sec,
        end_sec=end_sec,
        clip_id=clip_id,
        subtitle_segments=sample_subtitles,
        edit_plan=edit_plan
    )
    print(f"Rendered video created: {rendered_path}")

    print("Running Production Quality Control (QC)...")
    qc = VideoQualityControl(mode="production")
    report = qc.evaluate_video(video_path=rendered_path, expected_duration=duration)
    report_path = output_dir / f"{clip_id}_qc_report.json"
    report_path.write_text(report.to_json(indent=2), encoding="utf-8")
    print(f"QC passed: {report.passed}")
    print(f"QC checks: {json.dumps(report.checks, indent=2)}")

    print("Extracting ffprobe raw stream metadata JSON...")
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        rendered_path
    ]
    ffprobe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    ffprobe_path = output_dir / f"{clip_id}_ffprobe.json"
    ffprobe_path.write_text(ffprobe_res.stdout, encoding="utf-8")
    print(f"ffprobe metadata saved: {ffprobe_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(generate_representative_sample())

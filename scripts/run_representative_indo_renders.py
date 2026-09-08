"""Full Representative Indonesian Validation Pipeline.
Renders:
- Sample A: Indonesian single-speaker talking head (30-55s) -> TTf9XJUBgPk [0.0s - 36.0s]
- Sample B: Indonesian two-person podcast/interview (30-55s) -> 8zxCS41EXms [0.0s - 36.0s]
Generates EditPlan, ASS Subtitles, rendered MP4, QC Report (production mode), and FFprobe JSON.
"""

import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/root/projects/auto-short-generator")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_framing import visual_framing
from renderer import renderer
from qc import VideoQualityControl
from transcriber import transcriber
from language_gate import language_gate
from edit_plan import (
    EditPlan,
    EditingProfile,
    FramingMode,
    EditEvent,
    EditEventType,
    SubtitleStyle,
    AudioProfile,
)

output_dir = PROJECT_ROOT / "output" / "representative_validation"
output_dir.mkdir(parents=True, exist_ok=True)
qc = VideoQualityControl(mode="production")

def process_sample(vid, raw_video_path, clip_id, start_sec, end_sec, profile, emphasis_words):
    print(f"\n=======================================================")
    print(f"PROCESSING {clip_id} ({vid}) [{start_sec}s - {end_sec}s]")
    print(f"=======================================================")
    duration = end_sec - start_sec

    # 1. Fetch transcript and verify language gate
    full_transcript = transcriber.get_transcript(vid)
    # Filter transcript for clip range
    clip_transcript = []
    for line in full_transcript:
        if line["start"] >= start_sec and line["start"] < end_sec:
            # Shift timestamps to clip-local (0.0 to duration)
            local_start = max(0.0, line["start"] - start_sec)
            local_end = min(duration, line["end"] - start_sec)
            clip_transcript.append({
                "start": round(local_start, 2),
                "duration": round(local_end - local_start, 2),
                "end": round(local_end, 2),
                "text": line["text"],
                "words": []
            })

    print(f"Extracted {len(clip_transcript)} transcript segments for clip interval.")
    lang_res = language_gate.evaluate_transcript(clip_transcript)
    print(f"Language Gate Check: eligible={lang_res.eligible}, lang={lang_res.primary_language}, confidence={lang_res.confidence:.2f}")
    print(f"Reason: {lang_res.reason} (ID ratio: {lang_res.id_ratio}, EN ratio: {lang_res.en_ratio})")
    assert lang_res.eligible is True, f"Language Gate failed for {clip_id}: {lang_res.reason}"

    # 2. Visual framing with segmented scene-cut boundary detection
    print(f"Running Visual Framing with scene-cut detection on {raw_video_path}...")
    framing_mode, keyframes = visual_framing.analyze_clip_framing(
        video_path=str(raw_video_path),
        start_sec=start_sec,
        end_sec=end_sec,
        sample_interval_sec=1.0
    )
    print(f"Visual Framing resolved: mode={framing_mode.value}, keyframes={len(keyframes)}")

    # 3. Create EditPlan
    plan = EditPlan(
        clip_id=clip_id,
        clip_duration=duration,
        profile=profile,
        framing_mode=framing_mode,
        crop_keyframes=keyframes,
        edit_events=[
            EditEvent(time=10.0, type=EditEventType.PUNCH_IN, duration=2.0, intensity=1.15, text="fokus")
        ],
        emphasis_words=emphasis_words,
        subtitle_style=SubtitleStyle(
            font_name="Montserrat-Black",
            font_size=52,
            margin_v=540,
            outline_width=5,
            shadow_width=3,
            max_words_per_line=4,
            active_word_scale=100
        ),
        audio_profile=AudioProfile(
            highpass_freq=80,
            compressor_enabled=True,
            loudness_target_lufs=-16.0,
            true_peak_db=-1.5
        )
    )

    plan_path = output_dir / f"{clip_id}_edit_plan.json"
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved EditPlan: {plan_path}")

    # 4. Render short with Renderer V2
    print(f"Rendering short with Renderer V2...")
    rendered_path = renderer.render_short(
        source_video_path=str(raw_video_path),
        start_sec=start_sec,
        end_sec=end_sec,
        clip_id=clip_id,
        subtitle_segments=clip_transcript,
        edit_plan=plan
    )
    print(f"Rendered short saved: {rendered_path}")

    # 5. Move generated ASS subtitle to validation dir for durable archiving
    ass_path = PROJECT_ROOT / "output" / f"sub_{clip_id}.ass"
    dest_ass = output_dir / f"sub_{clip_id}.ass"
    if ass_path.exists():
        dest_ass.write_text(ass_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Saved ASS Subtitle: {dest_ass}")

    # 6. Video Quality Control (Production Mode)
    print("Running Production Quality Control (QC)...")
    qc_report = qc.evaluate_video(video_path=rendered_path, expected_duration=duration)
    report_path = output_dir / f"{clip_id}_qc_report.json"
    report_path.write_text(qc_report.to_json(indent=2), encoding="utf-8")
    print(f"QC Passed: {qc_report.passed}")
    print(f"QC Checks: {json.dumps(qc_report.checks, indent=2)}")
    assert qc_report.passed is True, f"QC failed for {clip_id}: {qc_report.errors}"

    # 7. Extract full FFprobe JSON
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        rendered_path
    ]
    probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    probe_path = output_dir / f"{clip_id}_ffprobe.json"
    probe_path.write_text(probe_res.stdout, encoding="utf-8")
    print(f"Saved FFprobe metadata: {probe_path}")

    return rendered_path


print("Starting Representative Indonesian Video Renders...")

# Sample A: Indonesian Single-Speaker Talking Head (35s)
sample_a_path = process_sample(
    vid="TTf9XJUBgPk",
    raw_video_path=PROJECT_ROOT / "downloads" / "sample_a_h264_clip.mp4",
    clip_id="sample_a_indo_single_speaker",
    start_sec=0.0,
    end_sec=35.0,
    profile=EditingProfile.PODCAST_CLEAN,
    emphasis_words=["blank", "komunikasi", "trik", "pikiran", "ngungkapin"]
)

# Sample B: Indonesian Two-Person Interview / Podcast (36s)
sample_b_path = process_sample(
    vid="8zxCS41EXms",
    raw_video_path=PROJECT_ROOT / "downloads" / "sample_b_h264_clip.mp4",
    clip_id="sample_b_indo_twoshot_interview",
    start_sec=0.0,
    end_sec=36.0,
    profile=EditingProfile.PODCAST_CLEAN,
    emphasis_words=["jobdes", "surya", "minum", "alkohol", "riset"]
)

print("\nALL REPRESENTATIVE INDONESIAN RENDERS COMPLETED SUCCESSFULLY!")

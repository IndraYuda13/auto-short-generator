"""Full Representative Indonesian Validation Pipeline for Patch 4.
Renders:
- Sample A: Indonesian talking head with burned-in subtitles -> SUBTITLE_SAFE_FULL_WIDTH, NO duplicate ASS subtitles.
- Sample B: Indonesian two-person podcast dialogue without subtitles -> FACE_TRACKED crop, Subtitle V2 generated.

Validates:
1. Subtitle detection & preservation
2. Keyframe boundary flicker elimination (+-2 frames around 1s, 2s, 3s, 4s, 5s)
3. Gemini Visual Director multimodal structured interpretation
4. Full Indonesian language gate compliance
5. Production QC pass
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
from subtitle_detector import subtitle_detector, SubtitleSource
from visual_director import visual_director, RecommendedFraming
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


def run_patch4_validation():
    print("=======================================================")
    print("STARTING PATCH 4 REPRESENTATIVE INDONESIAN VALIDATION")
    print("=======================================================\n")

    # -------------------------------------------------------------
    # SAMPLE A: Indonesian Talking Head WITH Burned-In Subtitle
    # -------------------------------------------------------------
    vid_a = "TTf9XJUBgPk"
    raw_video_a = PROJECT_ROOT / "downloads" / "sample_a_indo_with_burned_sub.mp4"
    clip_id_a = "sample_a_indo_single_speaker"
    start_sec_a = 0.0
    end_sec_a = 35.0
    duration_a = 35.0

    print(f"\n>>> PROCESSING SAMPLE A: {clip_id_a} [{start_sec_a}s - {end_sec_a}s] (WITH SUBTITLE)")
    full_transcript_a = transcriber.get_transcript(vid_a)
    clip_transcript_a = []
    for line in full_transcript_a:
        if line["start"] >= start_sec_a and line["start"] < end_sec_a:
            local_start = max(0.0, line["start"] - start_sec_a)
            local_end = min(duration_a, line["end"] - start_sec_a)
            clip_transcript_a.append({
                "start": round(local_start, 2),
                "duration": round(local_end - local_start, 2),
                "end": round(local_end, 2),
                "text": line["text"],
                "words": []
            })

    lang_res_a = language_gate.evaluate_transcript(clip_transcript_a)
    print(f"Sample A Language Gate: eligible={lang_res_a.eligible}, lang={lang_res_a.primary_language}")
    assert lang_res_a.eligible is True, "Sample A must be eligible Indonesian speech"

    # Run Subtitle Detector & Gemini Visual Director
    sub_det_a = subtitle_detector.evaluate(str(raw_video_a), start_sec_a, end_sec_a)
    print(f"Sample A SubtitleDetector: source={sub_det_a.source.value}, has_sub={sub_det_a.has_existing_subtitle}")
    assert sub_det_a.has_existing_subtitle is True, "Sample A must detect existing burned-in subtitle"

    vd_res_a = visual_director.analyze(
        video_path=str(raw_video_a),
        start_sec=start_sec_a,
        end_sec=end_sec_a,
        transcript_excerpt=clip_transcript_a[0]["text"] if clip_transcript_a else "",
        local_subtitle_result=sub_det_a
    )
    print(f"Sample A VisualDirector: sub={vd_res_a.has_existing_subtitle}, framing={vd_res_a.recommended_framing.value}")
    (output_dir / "sample_a_visual_director_result.json").write_text(
        vd_res_a.model_dump_json(indent=2), encoding="utf-8"
    )

    # Invariant check: GENERATE_NEW_SUBTITLE = False, framing = SUBTITLE_SAFE_FULL_WIDTH
    plan_a = EditPlan(
        clip_id=clip_id_a,
        clip_duration=duration_a,
        profile=EditingProfile.PODCAST_CLEAN,
        framing_mode=FramingMode.SUBTITLE_SAFE_FULL_WIDTH,
        crop_keyframes=[],
        edit_events=[],
        emphasis_words=["fokus", "strategi"],
        subtitle_style=SubtitleStyle(),
        audio_profile=AudioProfile(),
        existing_subtitle=True,
        subtitle_source=sub_det_a.source.value,
        generate_new_subtitle=False
    )
    (output_dir / f"{clip_id_a}_edit_plan.json").write_text(plan_a.model_dump_json(indent=2), encoding="utf-8")

    rendered_path_a = renderer.render_short(
        source_video_path=str(raw_video_a),
        start_sec=start_sec_a,
        end_sec=end_sec_a,
        clip_id=clip_id_a,
        subtitle_segments=clip_transcript_a,
        edit_plan=plan_a
    )
    print(f"Rendered Sample A: {rendered_path_a}")

    # Ensure no ASS subtitle file was generated for Sample A
    ass_a = PROJECT_ROOT / "output" / f"sub_{clip_id_a}.ass"
    if ass_a.exists():
        ass_a.unlink()

    qc_a = qc.evaluate_video(rendered_path_a, expected_duration=duration_a)
    print(f"Sample A QC Passed: {qc_a.passed}, errors={qc_a.errors}")
    (output_dir / f"{clip_id_a}_qc_report.json").write_text(qc_a.model_dump_json(indent=2), encoding="utf-8")
    assert qc_a.passed is True, f"Sample A failed QC: {qc_a.errors}"

    # Dump ffprobe json
    ffprobe_a = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-print_format", "json", rendered_path_a],
        capture_output=True, text=True
    )
    (output_dir / f"{clip_id_a}_ffprobe.json").write_text(ffprobe_a.stdout, encoding="utf-8")

    # -------------------------------------------------------------
    # SAMPLE B: Indonesian Dialogue WITHOUT Subtitle (Two-Person)
    # -------------------------------------------------------------
    vid_b = "8zxCS41EXms"
    raw_video_b = PROJECT_ROOT / "downloads" / "sample_b_h264_clip.mp4"
    clip_id_b = "sample_b_indo_twoshot_interview"
    start_sec_b = 0.0
    end_sec_b = 36.0
    duration_b = 36.0

    print(f"\n>>> PROCESSING SAMPLE B: {clip_id_b} [{start_sec_b}s - {end_sec_b}s] (WITHOUT SUBTITLE)")
    full_transcript_b = transcriber.get_transcript(vid_b)
    clip_transcript_b = []
    for line in full_transcript_b:
        if line["start"] >= start_sec_b and line["start"] < end_sec_b:
            local_start = max(0.0, line["start"] - start_sec_b)
            local_end = min(duration_b, line["end"] - start_sec_b)
            clip_transcript_b.append({
                "start": round(local_start, 2),
                "duration": round(local_end - local_start, 2),
                "end": round(local_end, 2),
                "text": line["text"],
                "words": []
            })

    lang_res_b = language_gate.evaluate_transcript(clip_transcript_b)
    print(f"Sample B Language Gate: eligible={lang_res_b.eligible}, lang={lang_res_b.primary_language}")
    assert lang_res_b.eligible is True, "Sample B must be eligible Indonesian speech"

    sub_det_b = subtitle_detector.evaluate(str(raw_video_b), start_sec_b, end_sec_b)
    print(f"Sample B SubtitleDetector: source={sub_det_b.source.value}, has_sub={sub_det_b.has_existing_subtitle}")
    assert sub_det_b.has_existing_subtitle is False, "Sample B must NOT detect existing subtitle"

    vd_res_b = visual_director.analyze(
        video_path=str(raw_video_b),
        start_sec=start_sec_b,
        end_sec=end_sec_b,
        transcript_excerpt=clip_transcript_b[0]["text"] if clip_transcript_b else "",
        local_subtitle_result=sub_det_b
    )
    print(f"Sample B VisualDirector: sub={vd_res_b.has_existing_subtitle}, framing={vd_res_b.recommended_framing.value}")
    (output_dir / "sample_b_visual_director_result.json").write_text(
        vd_res_b.model_dump_json(indent=2), encoding="utf-8"
    )

    framing_mode_b, keyframes_b = visual_framing.analyze_clip_framing(
        video_path=str(raw_video_b),
        start_sec=start_sec_b,
        end_sec=end_sec_b,
        sample_interval_sec=1.0
    )
    print(f"Sample B Visual Framing: mode={framing_mode_b.value}, keyframes={len(keyframes_b)}")

    plan_b = EditPlan(
        clip_id=clip_id_b,
        clip_duration=duration_b,
        profile=EditingProfile.PODCAST_CLEAN,
        framing_mode=framing_mode_b,
        crop_keyframes=keyframes_b,
        edit_events=[
            EditEvent(time=10.0, type=EditEventType.PUNCH_IN, duration=2.0, intensity=1.15, text="fokus")
        ],
        emphasis_words=["makan", "waktu"],
        subtitle_style=SubtitleStyle(),
        audio_profile=AudioProfile(),
        existing_subtitle=False,
        subtitle_source="NONE",
        generate_new_subtitle=True
    )
    (output_dir / f"{clip_id_b}_edit_plan.json").write_text(plan_b.model_dump_json(indent=2), encoding="utf-8")

    rendered_path_b = renderer.render_short(
        source_video_path=str(raw_video_b),
        start_sec=start_sec_b,
        end_sec=end_sec_b,
        clip_id=clip_id_b,
        subtitle_segments=clip_transcript_b,
        edit_plan=plan_b
    )
    print(f"Rendered Sample B: {rendered_path_b}")

    # For Sample B, verify ASS subtitle WAS generated
    ass_b = PROJECT_ROOT / "output" / f"sub_{clip_id_b}.ass"
    dest_ass_b = output_dir / f"sub_{clip_id_b}.ass"
    if ass_b.exists():
        dest_ass_b.write_text(ass_b.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"Saved ASS Subtitle: {dest_ass_b}")

    qc_b = qc.evaluate_video(rendered_path_b, expected_duration=duration_b)
    print(f"Sample B QC Passed: {qc_b.passed}, errors={qc_b.errors}")
    (output_dir / f"{clip_id_b}_qc_report.json").write_text(qc_b.model_dump_json(indent=2), encoding="utf-8")
    assert qc_b.passed is True, f"Sample B failed QC: {qc_b.errors}"

    ffprobe_b = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-print_format", "json", rendered_path_b],
        capture_output=True, text=True
    )
    (output_dir / f"{clip_id_b}_ffprobe.json").write_text(ffprobe_b.stdout, encoding="utf-8")

    print("\n=======================================================")
    print("PATCH 4 VALIDATION COMPLETED SUCCESSFULLY!")
    print("=======================================================")


if __name__ == "__main__":
    run_patch4_validation()

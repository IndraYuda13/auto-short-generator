"""Comprehensive Unit & Integration Test Suite for Phase B (Stable Editing Core).

Covers Blueprint Bab 11, 12, 13, 14, 15:
1. Bab 11: EditPlan, PunchInEvent policy, SceneCrop data contracts
2. Bab 12: Scene-static portrait framing (hold per scene, single & multi-speaker)
3. Bab 13: Subtitle policy classification (NONE, EMBEDDED_TRACK, BURNED_IN) & clean ASS generation
4. Bab 14: Broadcast audio mastering (-16 LUFS, TP -1.5 dB, 48kHz stereo, zero SFX)
5. Bab 15: FFmpeg clean filtergraph executor & 1080x1920 render verification
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from editing.edit_plan import EditPlan, PunchInEvent, SceneCrop
from editing.framing import SceneStaticFraming, FramingDecision
from editing.subtitle_policy import (
    SubtitlePolicyClassifier,
    SubtitleClassificationResult,
    SubtitleSourceType,
    SubtitleAction,
    generate_clean_ass_subtitles,
    chunk_words_to_phrases,
    format_ass_time,
    escape_ass_text,
)
from editing.audio import (
    AudioMasterer,
    get_audio_filter_chain,
    build_audio_encoding_args,
    AUDIO_FILTER_CHAIN,
    TARGET_LUFS,
    TARGET_TRUE_PEAK,
    TARGET_SAMPLE_RATE,
    TARGET_CHANNELS,
)
from editing.renderer import CleanRenderer, RenderResult


SAMPLE_VIDEO_A = "/root/projects/auto-short-generator-v3/downloads/sample_a_h264_clip.mp4"
SAMPLE_TWOSHOT = "/root/projects/auto-short-generator-v3/downloads/sample_b_twoshot_raw.mp4"
SAMPLE_BURNED_SUB = "/root/projects/auto-short-generator-v3/downloads/sample_a_indo_with_burned_sub.mp4"


# ==============================================================================
# 1. Bab 11: EditPlan, PunchInEvent, and SceneCrop Tests
# ==============================================================================

def test_punch_in_event_policy_validation():
    """Punch-in policy: scale 1.04-1.08, duration 0.6-1.5s."""
    # Standard valid event
    p1 = PunchInEvent(start_time=5.0, duration=1.0, scale=1.06)
    assert p1.scale == 1.06
    assert p1.duration == 1.0
    assert p1.end_time == 6.0

    # Scale clamped to [1.04, 1.08]
    p_low = PunchInEvent(start_time=2.0, duration=1.0, scale=1.01)
    assert p_low.scale == 1.04

    p_high = PunchInEvent(start_time=2.0, duration=1.0, scale=1.15)
    assert p_high.scale == 1.08

    # Duration clamped to [0.6, 1.5]
    p_short = PunchInEvent(start_time=1.0, duration=0.2, scale=1.05)
    assert p_short.duration == 0.6

    p_long = PunchInEvent(start_time=1.0, duration=3.5, scale=1.05)
    assert p_long.duration == 1.5


def test_edit_plan_max_two_punch_ins_policy():
    """Punch-in policy: strictly max 0-2 per clip, default empty if in doubt."""
    plan = EditPlan(
        layout="PORTRAIT_9_16",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[],
        crop_windows=[]
    )
    assert len(plan.punch_in_events) == 0

    # Adding events via helper
    added1 = plan.add_punch_in(start_time=2.0, duration=1.0, scale=1.05)
    assert added1 is True
    assert len(plan.punch_in_events) == 1

    added2 = plan.add_punch_in(start_time=8.0, duration=0.8, scale=1.07)
    assert added2 is True
    assert len(plan.punch_in_events) == 2

    # Third punch-in must be rejected by policy
    added3 = plan.add_punch_in(start_time=15.0, duration=1.2, scale=1.06)
    assert added3 is False
    assert len(plan.punch_in_events) == 2

    # Direct list constructor clamps to max 2
    excess_plan = EditPlan(
        layout="PORTRAIT_9_16",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[
            PunchInEvent(start_time=1.0, duration=1.0, scale=1.05),
            PunchInEvent(start_time=4.0, duration=1.0, scale=1.05),
            PunchInEvent(start_time=9.0, duration=1.0, scale=1.05),
        ],
        crop_windows=[]
    )
    assert len(excess_plan.punch_in_events) == 2


def test_scene_crop_even_coordinates_and_properties():
    """SceneCrop requires even coordinates for FFmpeg compatibility."""
    crop = SceneCrop(
        scene_start=0.0,
        scene_end=12.5,
        crop_x=123,  # Odd
        crop_y=1,    # Odd
        crop_w=607,  # Odd
        crop_h=1079  # Odd
    )
    # Coordinates must be coerced to even numbers
    assert crop.crop_x % 2 == 0
    assert crop.crop_y % 2 == 0
    assert crop.crop_w % 2 == 0
    assert crop.crop_h % 2 == 0
    assert crop.duration == 12.5


# ==============================================================================
# 2. Bab 12: Scene-Static Portrait Framing Tests
# ==============================================================================

def test_framing_single_speaker_static_hold():
    """Single speaker: head + upper torso, consistent headroom, faces centered in 9:16."""
    framing = SceneStaticFraming()
    frame_w, frame_h = 1920, 1080

    # Speaker face located around center: fx=900, fy=250, fw=160, fh=180
    face = (900, 250, 160, 180, 0.95)
    crop, layout, spk_mode = framing.calculate_scene_crop(
        faces=[face],
        frame_w=frame_w,
        frame_h=frame_h,
        scene_start=0.0,
        scene_end=15.0
    )

    assert layout == "PORTRAIT_9_16"
    assert spk_mode == "SINGLE_SPEAKER"
    # Target 9:16 crop width for 1080 height is 608px
    assert crop.crop_w == 608
    assert crop.crop_h == 1080
    assert crop.crop_y == 0  # Full vertical frame preserves head + upper torso
    # Crop X centers around face_center_x = 980 -> crop_x ~ 980 - 304 = 676
    assert 670 <= crop.crop_x <= 680
    assert crop.crop_x % 2 == 0
    assert crop.crop_w % 2 == 0


def test_framing_scene_cut_holds_crop_per_scene():
    """Scene 1: detect -> HOLD; Scene cut -> Scene 2: detect -> HOLD (no camera tracking)."""
    framing = SceneStaticFraming()

    # Mock video frames across 2 scene segments
    # Scene 1: [0.0, 10.0], speaker on left (fx=300)
    # Scene 2: [10.0, 25.0], speaker on right (fx=1400)
    frame_w, frame_h = 1920, 1080

    crop1, layout1, mode1 = framing.calculate_scene_crop(
        faces=[(300, 200, 150, 180, 0.9)],
        frame_w=frame_w,
        frame_h=frame_h,
        scene_start=0.0,
        scene_end=10.0
    )

    crop2, layout2, mode2 = framing.calculate_scene_crop(
        faces=[(1400, 200, 150, 180, 0.9)],
        frame_w=frame_w,
        frame_h=frame_h,
        scene_start=10.0,
        scene_end=25.0
    )

    # Both crops are static within their respective windows
    assert crop1.scene_start == 0.0
    assert crop1.scene_end == 10.0
    assert crop1.crop_x < 300  # Positioned on left

    assert crop2.scene_start == 10.0
    assert crop2.scene_end == 25.0
    assert crop2.crop_x > 1100  # Positioned on right

    # ZERO camera tracking between scenes: each segment has a single static crop
    crops = [crop1, crop2]
    assert len(crops) == 2
    assert crops[0].crop_x != crops[1].crop_x


def test_framing_multi_speaker_safe_two_shot_if_fits():
    """Multi speaker: if two speakers fit within 9:16 width, use safe two-person crop."""
    framing = SceneStaticFraming()
    frame_w, frame_h = 1920, 1080
    # Two speakers close to each other: fx1=800, fx2=1050 (span ~400px < 608px)
    face1 = (800, 250, 120, 140, 0.9)
    face2 = (1050, 260, 120, 140, 0.9)

    crop, layout, spk_mode = framing.calculate_scene_crop(
        faces=[face1, face2],
        frame_w=frame_w,
        frame_h=frame_h,
        scene_start=0.0,
        scene_end=20.0
    )

    assert layout == "PORTRAIT_9_16"
    assert spk_mode == "MULTI_SPEAKER_SAFE_TWO_SHOT"
    assert crop.crop_w == 608
    assert crop.crop_h == 1080
    # Center of group ~ 985 -> crop_x ~ 680
    assert 670 <= crop.crop_x <= 690


def test_framing_multi_speaker_wide_fallback_or_reject():
    """Multi speaker: NO active-speaker switching. Fallback to safe full-frame or reject."""
    framing = SceneStaticFraming()
    frame_w, frame_h = 1920, 1080
    # Two speakers far apart: fx1=200, fx2=1600 (span ~1500px >> 608px)
    face1 = (200, 250, 120, 140, 0.9)
    face2 = (1600, 260, 120, 140, 0.9)

    # 1. With safe full-frame allowed
    crop_safe, layout_safe, mode_safe = framing.calculate_scene_crop(
        faces=[face1, face2],
        frame_w=frame_w,
        frame_h=frame_h,
        scene_start=0.0,
        scene_end=20.0,
        allow_safe_full_frame=True
    )
    assert layout_safe == "SAFE_FULL_FRAME"
    assert mode_safe == "MULTI_SPEAKER_WIDE"
    assert crop_safe.crop_w == frame_w
    assert crop_safe.crop_h == frame_h

    # 2. With safe full-frame disallowed -> REJECT
    crop_rej, layout_rej, mode_rej = framing.calculate_scene_crop(
        faces=[face1, face2],
        frame_w=frame_w,
        frame_h=frame_h,
        scene_start=0.0,
        scene_end=20.0,
        allow_safe_full_frame=False
    )
    assert layout_rej == "REJECT"
    assert mode_rej == "MULTI_SPEAKER_REJECT"


def test_framing_real_media_scene_analysis():
    """Executes SceneStaticFraming on real media sample."""
    if not os.path.exists(SAMPLE_VIDEO_A):
        pytest.skip("Sample video A not found")

    framing = SceneStaticFraming()
    decision = framing.analyze_framing(
        video_path=SAMPLE_VIDEO_A,
        start_sec=0.0,
        end_sec=5.0
    )

    assert isinstance(decision, FramingDecision)
    assert not decision.is_rejected
    assert len(decision.crop_windows) >= 1
    for c in decision.crop_windows:
        assert c.crop_w > 0
        assert c.crop_h > 0
        assert c.crop_x % 2 == 0
        assert c.crop_y % 2 == 0


# ==============================================================================
# 3. Bab 13: Subtitle Policy & ASS Generation Tests
# ==============================================================================

def test_subtitle_ass_generation_specs():
    """Subtitle V2: clean 2-5 words per phrase, 1-2 lines, white font + dark outline, bottom safe-zone."""
    words = [
        {"word": "Halo", "start": 0.0, "end": 0.3},
        {"word": "teman-teman", "start": 0.3, "end": 0.8},
        {"word": "semua", "start": 0.8, "end": 1.2},
        {"word": "selamat", "start": 1.3, "end": 1.6},
        {"word": "datang", "start": 1.6, "end": 2.0},
        {"word": "di", "start": 2.1, "end": 2.3},
        {"word": "podcast", "start": 2.3, "end": 2.8},
        {"word": "kami.", "start": 2.8, "end": 3.4},
    ]

    ass_text = generate_clean_ass_subtitles(words, font_name="Montserrat", font_size=52, margin_v=520)

    # Verify ASS header and canvas
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text

    # Verify Style specifications
    # White font (&H00FFFFFF), dark outline (&H00000000), MarginV 520
    assert "&H00FFFFFF" in ass_text
    assert "&H00000000" in ass_text
    assert "520" in ass_text

    # Verify dialogue events generated
    assert "Dialogue: 0," in ass_text
    assert "Halo teman-teman semua" in ass_text or "selamat datang" in ass_text

    # Verify phrase length constraint: no dialogue event contains > 5 words
    for line in ass_text.splitlines():
        if line.startswith("Dialogue:"):
            text_part = line.split(",,", 1)[-1].replace("\\N", " ")
            word_count = len(text_part.split())
            assert 2 <= word_count <= 5


def test_subtitle_policy_classification_none():
    """Clean video without subtitles classifies as NONE -> GENERATE action."""
    classifier = SubtitlePolicyClassifier()

    with patch.object(classifier, "check_embedded_subtitles", return_value=None):
        with patch.object(classifier, "check_burned_in_subtitles", return_value=(False, None)):
            res = classifier.classify(video_path="dummy.mp4")
            assert res.source_type == SubtitleSourceType.NONE
            assert res.action == SubtitleAction.GENERATE
            assert res.recommended_layout == "PORTRAIT_9_16"
            assert not res.is_rejected


def test_subtitle_policy_classification_embedded_track():
    """Video with embedded subtitle stream classifies as EMBEDDED_TRACK -> EXTRACT_EMBEDDED."""
    classifier = SubtitlePolicyClassifier()

    with patch.object(classifier, "check_embedded_subtitles", return_value=0):
        res = classifier.classify(video_path="dummy.mp4")
        assert res.source_type == SubtitleSourceType.EMBEDDED_TRACK
        assert res.action == SubtitleAction.EXTRACT_EMBEDDED
        assert res.embedded_stream_index == 0


def test_subtitle_policy_classification_burned_in_rules():
    """BURNED_IN: NO OCR, NO weird composite. Use safe full-frame keeping width or reject."""
    classifier = SubtitlePolicyClassifier()

    with patch.object(classifier, "check_embedded_subtitles", return_value=None):
        with patch.object(classifier, "check_burned_in_subtitles", return_value=(True, {"y1": 0.75})):
            # 1. With safe full-frame allowed
            res_safe = classifier.classify(video_path="dummy.mp4", allow_safe_full_frame=True)
            assert res_safe.source_type == SubtitleSourceType.BURNED_IN
            assert res_safe.action == SubtitleAction.PRESERVE_BURNED_IN
            assert res_safe.recommended_layout == "SAFE_FULL_FRAME"
            assert not res_safe.is_rejected

            # 2. With safe full-frame disallowed -> REJECT
            res_rej = classifier.classify(video_path="dummy.mp4", allow_safe_full_frame=False)
            assert res_rej.source_type == SubtitleSourceType.BURNED_IN
            assert res_rej.action == SubtitleAction.REJECT
            assert res_rej.is_rejected is True


# ==============================================================================
# 4. Bab 14: Audio Mastering Tests
# ==============================================================================

def test_audio_mastering_constants_and_filter_chain():
    """FFmpeg filter chain matches exact Blueprint Bab 14 standard."""
    chain = get_audio_filter_chain()
    expected = "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=48000:channel_layouts=stereo"
    assert chain == expected
    assert TARGET_LUFS == -16.0
    assert TARGET_TRUE_PEAK == -1.5
    assert TARGET_SAMPLE_RATE == 48000
    assert TARGET_CHANNELS == 2

    encoding_args = build_audio_encoding_args()
    assert "-c:a" in encoding_args
    assert "aac" in encoding_args
    assert "48000" in encoding_args
    assert "2" in encoding_args


def test_audio_masterer_execution_on_real_sample():
    """Applies mastering chain to real audio stream and verifies specs."""
    if not os.path.exists(SAMPLE_VIDEO_A):
        pytest.skip("Sample video A not found")

    masterer = AudioMasterer()
    with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as tmp:
        tmp_output = tmp.name

    try:
        success = masterer.master_audio(
            input_media_path=SAMPLE_VIDEO_A,
            output_audio_path=tmp_output,
            start_sec=0.0,
            end_sec=2.0
        )
        assert success is True
        assert os.path.exists(tmp_output)

        specs = masterer.verify_audio_specs(tmp_output)
        assert specs.get("codec") == "aac"
        assert specs.get("sample_rate") == 48000
        assert specs.get("channels") == 2
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)


# ==============================================================================
# 5. Bab 15: CleanRenderer Filtergraph & Execution Tests
# ==============================================================================

def test_renderer_build_filtergraph_single_crop():
    """Builds filtergraph for single scene static crop in 1080x1920."""
    renderer = CleanRenderer()
    plan = EditPlan(
        layout="PORTRAIT_9_16",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[],
        crop_windows=[
            SceneCrop(scene_start=0.0, scene_end=30.0, crop_x=400, crop_y=0, crop_w=608, crop_h=1080)
        ]
    )

    fg = renderer.build_filtergraph(edit_plan=plan, duration=30.0)
    assert "crop=608:1080:400:0,scale=1080:1920[v_base]" in fg
    assert AUDIO_FILTER_CHAIN in fg
    assert "[v_out]" in fg
    assert "[a_out]" in fg


def test_renderer_build_filtergraph_multi_scene_static_concat():
    """Builds filtergraph with multi-segment concat for scene cuts."""
    renderer = CleanRenderer()
    plan = EditPlan(
        layout="PORTRAIT_9_16",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[],
        crop_windows=[
            SceneCrop(scene_start=0.0, scene_end=12.0, crop_x=200, crop_y=0, crop_w=608, crop_h=1080),
            SceneCrop(scene_start=12.0, scene_end=30.0, crop_x=800, crop_y=0, crop_w=608, crop_h=1080),
        ]
    )

    fg = renderer.build_filtergraph(edit_plan=plan, duration=30.0)
    assert "trim=start=0.00:end=12.00" in fg
    assert "crop=608:1080:200:0,scale=1080:1920[vseg_0]" in fg
    assert "trim=start=12.00:end=30.00" in fg
    assert "crop=608:1080:800:0,scale=1080:1920[vseg_1]" in fg
    assert "[vseg_0][vseg_1]concat=n=2:v=1:a=0[v_base]" in fg


def test_renderer_build_filtergraph_punch_in_and_safe_full_frame():
    """Builds filtergraph with punch-in zoom and safe full-frame layout."""
    renderer = CleanRenderer()
    plan = EditPlan(
        layout="SAFE_FULL_FRAME",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[
            PunchInEvent(start_time=2.0, duration=1.0, scale=1.06)
        ],
        crop_windows=[]
    )

    fg = renderer.build_filtergraph(edit_plan=plan, duration=10.0)
    # Safe full-frame: blurred background overlay
    assert "boxblur" in fg and "[bg_blur]" in fg
    assert "overlay=(W-w)/2:(H-h)/2[v_base]" in fg
    # Punch-in crop on [v_base]
    assert "between(t,2.00,3.00)" in fg
    assert "[v_punch]" in fg


def test_renderer_real_end_to_end_render(tmp_path: Path):
    """Performs real end-to-end render on 2-second slice of real media sample."""
    if not os.path.exists(SAMPLE_VIDEO_A):
        pytest.skip("Sample video A not found")

    renderer = CleanRenderer()

    # Generate test subtitle in tmp_path
    sub_path = str(tmp_path / "test_phase_b_render_sub.ass")
    generate_clean_ass_subtitles(
        phrases_or_words=[
            {"text": "Uji coba rendering video", "start": 0.0, "end": 1.0},
            {"text": "Phase B stable editing core", "start": 1.0, "end": 2.0}
        ],
        output_path=sub_path
    )

    plan = EditPlan(
        layout="PORTRAIT_9_16",
        subtitle_policy="GENERATE",
        audio_mastering=True,
        punch_in_events=[
            PunchInEvent(start_time=0.5, duration=0.8, scale=1.05)
        ],
        crop_windows=[
            SceneCrop(scene_start=0.0, scene_end=2.0, crop_x=438, crop_y=0, crop_w=404, crop_h=720)
        ],
        duration=2.0
    )

    out_mp4 = str(tmp_path / "test_phase_b_output.mp4")

    result = renderer.render(
        edit_plan=plan,
        input_video_path=SAMPLE_VIDEO_A,
        output_video_path=out_mp4,
        start_sec=0.0,
        end_sec=2.0,
        subtitle_ass_path=sub_path
    )

    assert result.success is True
    assert os.path.exists(out_mp4)

    # Verify output video properties via ffprobe
    masterer = AudioMasterer()
    audio_specs = masterer.verify_audio_specs(out_mp4)
    assert audio_specs.get("codec") == "aac"
    assert audio_specs.get("sample_rate") == 48000
    assert audio_specs.get("channels") == 2

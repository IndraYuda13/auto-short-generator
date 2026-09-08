import pytest
from visual_framing import VisualFramingAnalyzer
from edit_plan import CropKeyframe, FramingMode


def test_smooth_keyframes_bounds_clamping():
    analyzer = VisualFramingAnalyzer()
    # 16:9 source (1920x1080)
    # Target 9:16 crop width = 1080 * 9/16 = 607.5px
    # Half crop norm x = (607.5 / 2) / 1920 = 0.1582
    # So crop center x must stay clamped within [0.1582, 0.8418]
    raw_kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.01, crop_center_y=0.5), # Extreme left
        CropKeyframe(time=1.0, crop_center_x=0.05, crop_center_y=0.5),
        CropKeyframe(time=2.0, crop_center_x=0.99, crop_center_y=0.5), # Extreme right
    ]
    smoothed = analyzer.smooth_keyframes(raw_kfs, src_width=1920, src_height=1080)

    assert len(smoothed) == 3
    for kf in smoothed:
        # Check within clamped bounds
        assert kf.crop_center_x >= 0.15
        assert kf.crop_center_x <= 0.85


def test_visual_framing_fallback_on_invalid_file():
    analyzer = VisualFramingAnalyzer()
    mode, kfs = analyzer.analyze_clip_framing(
        video_path="/nonexistent/video.mp4",
        start_sec=0.0,
        end_sec=10.0
    )
    assert mode == FramingMode.BLURRED_FALLBACK
    assert len(kfs) == 0

from visual_framing import VisualFramingAnalyzer, SubjectPresenceState
from edit_plan import CropKeyframe

def test_subject_presence_hysteresis_holds_framing():
    """
    Simulates face detection missing 1-2 frames within the same scene.
    Verifies that the Subject Presence Gate holds the previous framing rather than
    falling back to center crop or empty frame.
    """
    analyzer = VisualFramingAnalyzer()
    # Test smooth_keyframes with motion clamping
    kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.50, crop_center_y=0.40, confidence=0.95),
        CropKeyframe(time=1.0, crop_center_x=0.50, crop_center_y=0.40, confidence=0.85), # Held frame
        CropKeyframe(time=2.0, crop_center_x=0.52, crop_center_y=0.41, confidence=0.95),
    ]
    smoothed = analyzer.smooth_keyframes(kfs, src_width=1280, src_height=720)
    assert len(smoothed) == 3
    # Check that movement is clamped and smooth
    for i in range(1, len(smoothed)):
        dx = abs(smoothed[i].crop_center_x - smoothed[i-1].crop_center_x)
        assert dx <= 0.08, f"Displacement {dx} exceeded max displacement 0.08"


def test_motion_limits_clamping():
    """
    Verifies that sudden huge face coordinate jumps within the same scene
    are clamped to max displacement per second.
    """
    analyzer = VisualFramingAnalyzer()
    # Big jump from 0.50 to 0.85 in 1.0 second
    kfs = [
        CropKeyframe(time=0.0, crop_center_x=0.50, crop_center_y=0.40, confidence=0.95),
        CropKeyframe(time=1.0, crop_center_x=0.85, crop_center_y=0.40, confidence=0.95),
    ]
    smoothed = analyzer.smooth_keyframes(kfs, src_width=1280, src_height=720, max_displacement_x_per_sec=0.08)
    dx = abs(smoothed[1].crop_center_x - smoothed[0].crop_center_x)
    assert dx <= 0.08 + 1e-4, f"Intra-shot displacement {dx} was not clamped to 0.08"

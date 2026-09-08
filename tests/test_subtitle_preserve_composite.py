from renderer import Renderer
from edit_plan import EditPlan, FramingMode, CropKeyframe

def test_subtitle_preserve_composite_filtergraph():
    """
    Verifies that SUBTITLE_PRESERVE_COMPOSITE generates a dual-layer composite filtergraph:
    1. Video layer cropped to 0..ih*0.72 and scaled to 1080:1920
    2. Subtitle band cropped to ih*0.72..ih and overlaid onto reading zone (y=1480)
    3. Null passthrough for subtitles (no duplicate ASS or drawtext)
    """
    ren = Renderer()
    plan = EditPlan(
        clip_id="test_comp",
        clip_duration=10.0,
        framing_mode=FramingMode.SUBTITLE_PRESERVE_COMPOSITE,
        crop_keyframes=[CropKeyframe(time=0.0, crop_center_x=0.5, crop_center_y=0.4)],
        existing_subtitle=True,
        generate_new_subtitle=False
    )

    fg, extra, aout = ren._build_v2_pipeline(plan, ass_path=None, duration=10.0)

    assert "split=2[v_in][sub_in]" in fg
    assert "crop='ih*0.72*9/16':'ih*0.72'" in fg
    assert "crop=iw:ih*0.28:0:ih*0.72,scale=1080:-2:flags=bicubic[sub_band]" in fg
    assert "overlay=0:1480[base_v]" in fg
    assert "[base_v]null[outv]" in fg
    assert "subtitles=" not in fg
    assert "drawtext=" not in fg

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from visual_framing import visual_framing

comp_dir = Path("output/representative_validation/comparisons")
timestamps = [5, 10, 20, 30]

det_src = visual_framing._get_detector((1280, 720))

print("=== MAGNIFICATION AUDIT ON SAMPLE A ===")
for t in timestamps:
    src_file = comp_dir / f"sample_a_source_excerpt_{t}s.jpg"
    out_file = comp_dir / f"sample_a_output_frame_{t}s.jpg"

    src_img = cv2.imread(str(src_file))
    out_img = cv2.imread(str(out_file))

    det_src.setInputSize((1280, 720))
    _, faces_src = det_src.detect(src_img)

    det_src.setInputSize((1080, 1920))
    _, faces_out = det_src.detect(out_img)

    src_face_h = faces_src[0][3] if faces_src is not None and len(faces_src) > 0 else 0
    out_face_h = faces_out[0][3] if faces_out is not None and len(faces_out) > 0 else 0

    ratio = out_face_h / (src_face_h * 1080 / 1280) if src_face_h > 0 else 0
    # Also compare with old letterbox (which had face_h ~ 150px):
    letterbox_old_h = 153.0
    mag_vs_letterbox = out_face_h / letterbox_old_h if letterbox_old_h > 0 else 0

    print(f"t={t}s: Source Face H={src_face_h:.1f}px | Composite Output Face H={out_face_h:.1f}px")
    print(f"       Magnification vs old SUBTITLE_SAFE_FULL_WIDTH letterbox (153px): {mag_vs_letterbox:.2f}x (Threshold >= 1.5x)")
    assert mag_vs_letterbox >= 1.5, f"Magnification at {t}s failed threshold"

print("\nALL MAGNIFICATION THRESHOLDS PASSED!")

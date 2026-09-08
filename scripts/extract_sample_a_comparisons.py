import cv2
import numpy as np
import subprocess
from pathlib import Path

# Paths
source_video_a = "downloads/sample_a_h264_clip.mp4"
output_video_a = "output/short_sample_a_indo_single_speaker.mp4"
comp_dir = Path("output/representative_validation/comparisons")
comp_dir.mkdir(parents=True, exist_ok=True)

timestamps = [5.0, 10.0, 20.0, 30.0]

cap_src = cv2.VideoCapture(source_video_a)
cap_out = cv2.VideoCapture(output_video_a)

for t in timestamps:
    # Source frame
    cap_src.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
    ret_src, frame_src = cap_src.read()
    if ret_src:
        src_path = comp_dir / f"sample_a_source_excerpt_{int(t)}s.jpg"
        cv2.imwrite(str(src_path), frame_src)
        print(f"Extracted source @ {t}s: {src_path} (shape: {frame_src.shape})")

    # Output frame
    cap_out.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
    ret_out, frame_out = cap_out.read()
    if ret_out:
        out_path = comp_dir / f"sample_a_output_frame_{int(t)}s.jpg"
        cv2.imwrite(str(out_path), frame_out)
        print(f"Extracted output @ {t}s: {out_path} (shape: {frame_out.shape})")

cap_src.release()
cap_out.release()
print("Sample A comparison frames extraction complete!")

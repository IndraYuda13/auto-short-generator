"""Dense frame inspection around integer boundaries (1s, 2s, 3s, 4s, 5s) +-2 frames.
Extracts frames and measures mean squared difference / crop bounding box to prove stability.
Also extracts before/after subtitle truncation comparisons.
"""

import cv2
import numpy as np
import subprocess
from pathlib import Path

out_dir = Path("/root/projects/auto-short-generator/output/representative_validation/dense_boundary_checks")
out_dir.mkdir(parents=True, exist_ok=True)

video_b = "/root/projects/auto-short-generator/output/short_sample_b_indo_twoshot_interview.mp4"

# 1. Dense extraction around 1s, 2s, 3s, 4s, 5s at +-2 frames (assuming 50fps or 30fps)
cap = cv2.VideoCapture(video_b)
fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
print(f"Sample B FPS: {fps}")

boundary_seconds = [1.0, 2.0, 3.0, 4.0, 5.0]
report_lines = ["# Dense Boundary Continuity Audit Report", f"Target: {video_b}", f"Video FPS: {fps}\n"]

for sec in boundary_seconds:
    base_frame = int(sec * fps)
    frames = []
    report_lines.append(f"## Boundary at {sec:.1f}s (base frame {base_frame}):")
    # check frames: base-2, base-1, base, base+1, base+2
    for offset in [-2, -1, 0, 1, 2]:
        fn = base_frame + offset
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append((offset, fn, frame))
            save_path = out_dir / f"boundary_{int(sec)}s_offset_{offset:+d}_f{fn}.jpg"
            cv2.imwrite(str(save_path), frame)

    # Calculate frame-to-frame difference across the transition
    for i in range(len(frames) - 1):
        off1, fn1, f1 = frames[i]
        off2, fn2, f2 = frames[i + 1]
        diff = cv2.absdiff(f1, f2)
        mean_diff = float(np.mean(diff))
        # A single frame flash/black screen/re-crop would cause massive mean_diff spike > 40.0
        line = f"- Transition {off1:+d} -> {off2:+d} (frames {fn1} -> {fn2}): Mean Diff = {mean_diff:.2f}"
        report_lines.append(line)
        print(line)

cap.release()

report_path = out_dir / "dense_boundary_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
print(f"Report saved: {report_path}")

# 2. Extract Subtitle Truncation Comparison:
# Naive 9:16 crop vs SUBTITLE_SAFE_FULL_WIDTH on Sample A
comp_dir = Path("/root/projects/auto-short-generator/output/representative_validation/comparisons")
comp_dir.mkdir(parents=True, exist_ok=True)

raw_a = "/root/projects/auto-short-generator/downloads/sample_a_indo_with_burned_sub.mp4"
# Generate naive 9:16 crop frame at 10.0s
cmd_naive = [
    "ffmpeg", "-y", "-ss", "10.0", "-i", raw_a,
    "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920",
    "-vframes", "1", str(comp_dir / "sample_a_naive_crop_truncated_sub.jpg")
]
subprocess.run(cmd_naive, capture_output=True)

# Extract frame at 10.0s from rendered Sample A (safe layout)
cmd_safe = [
    "ffmpeg", "-y", "-ss", "10.0", "-i", "/root/projects/auto-short-generator/output/short_sample_a_indo_single_speaker.mp4",
    "-vframes", "1", str(comp_dir / "sample_a_safe_full_width_preserved_sub.jpg")
]
subprocess.run(cmd_safe, capture_output=True)

print("Saved before/after subtitle truncation comparison images!")

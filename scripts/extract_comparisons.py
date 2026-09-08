"""Extracts before and after comparison frames at identical timestamps for visual audit."""

import subprocess
from pathlib import Path

out_dir = Path("/root/projects/auto-short-generator/output/representative_validation/comparisons")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Subtitle Readability Comparison:
# Before: Old sub style (font_size=44, outline=4, shadow=2, margin_v=520) on sample_a_larry_king
# After: New sub style (font_size=52, outline=5, shadow=3, margin_v=540) on sample_a_indo_single_speaker
# Timestamp: 10.0s
subprocess.run([
    "ffmpeg", "-y", "-ss", "10.0", "-i",
    "/root/projects/auto-short-generator/output/representative_validation/sample_a_larry_king_single_speaker.mp4",
    "-vframes", "1", str(out_dir / "subtitle_before_font44_t10.jpg")
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

subprocess.run([
    "ffmpeg", "-y", "-ss", "10.0", "-i",
    "/root/projects/auto-short-generator/output/representative_validation/sample_a_indo_single_speaker.mp4",
    "-vframes", "1", str(out_dir / "subtitle_after_font52_readability_t10.jpg")
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 2. Scene-cut / Face transition comparison:
# At cut transition in Sample B (t=16.0s and t=26.0s where morphing previously occurred)
# Extract before (unsegmented cross-cut smoothing) vs after (segmented scene-cut tracking)
subprocess.run([
    "ffmpeg", "-y", "-ss", "16.0", "-i",
    "/root/projects/auto-short-generator/output/representative_validation/sample_b_twoshot_interview.mp4",
    "-vframes", "1", str(out_dir / "transition_before_morphing_t16.jpg")
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

subprocess.run([
    "ffmpeg", "-y", "-ss", "16.0", "-i",
    "/root/projects/auto-short-generator/output/representative_validation/sample_b_indo_twoshot_interview.mp4",
    "-vframes", "1", str(out_dir / "transition_after_segmented_tracking_t16.jpg")
], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Comparison frames extracted successfully to", out_dir)
for f in out_dir.glob("*.jpg"):
    print("-", f.name, f"({f.stat().st_size} bytes)")

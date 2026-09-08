import subprocess
from pathlib import Path
import hashlib

output_dir = Path("/root/projects/auto-short-generator/output/representative_validation/comparisons")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Extract Sample A comparisons around 10s: source excerpt vs output frame
raw_a = "/root/projects/auto-short-generator/downloads/sample_a_h264_clip.mp4"
out_a = "/root/projects/auto-short-generator/output/short_sample_a_indo_single_speaker.mp4"

subprocess.run(["ffmpeg", "-y", "-ss", "10.0", "-i", raw_a, "-frames:v", "1", str(output_dir / "sample_a_source_excerpt_10s.jpg")], check=True)
subprocess.run(["ffmpeg", "-y", "-ss", "10.0", "-i", out_a, "-frames:v", "1", str(output_dir / "sample_a_output_frame_10s.jpg")], check=True)

# 2. Extract Sample B comparisons around 1.8s, 2.0s, 2.2s: source vs output
raw_b = "/root/projects/auto-short-generator/downloads/sample_b_h264_clip.mp4"
out_b = "/root/projects/auto-short-generator/output/short_sample_b_indo_twoshot_interview.mp4"

for t in [1.8, 2.0, 2.2]:
    t_str = f"{t:.1f}"
    subprocess.run(["ffmpeg", "-y", "-ss", t_str, "-i", raw_b, "-frames:v", "1", str(output_dir / f"sample_b_source_excerpt_{t_str}s.jpg")], check=True)
    subprocess.run(["ffmpeg", "-y", "-ss", t_str, "-i", out_b, "-frames:v", "1", str(output_dir / f"sample_b_output_frame_{t_str}s.jpg")], check=True)

# 3. Calculate SHA-256 for artifacts
for p in [raw_a, out_a, raw_b, out_b]:
    h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    print(f"SHA256({p}) = {h}")

print("Frame extraction and hash calculation complete!")

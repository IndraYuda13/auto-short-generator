import cv2
import numpy as np
import subprocess
from pathlib import Path

comp_dir = Path("output/representative_validation/comparisons")
timestamps = [5, 10, 20, 30]

print("=== SAMPLE A SUBTITLE & GHOST AUDIT ===")
for t in timestamps:
    out_file = comp_dir / f"sample_a_output_frame_{t}s.jpg"
    img = cv2.imread(str(out_file))

    # Upper video layer: y=0..1480
    upper = img[0:1480, :]
    upper_gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
    # Check bottom 100px of video layer (y=1380..1480) for any subtitle text bleed
    bleed_zone = upper_gray[1380:1480, :]
    _, bleed_thresh = cv2.threshold(bleed_zone, 220, 255, cv2.THRESH_BINARY)
    bleed_bright_count = np.count_nonzero(bleed_thresh)

    # Subtitle band: y=1480..1650
    band = img[1480:1650, :]
    band_gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, band_thresh = cv2.threshold(band_gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite(f"/tmp/audit_band_{t}s.png", band_thresh)

    res = subprocess.run(["tesseract", f"/tmp/audit_band_{t}s.png", "stdout", "--oem", "1", "-l", "eng+ind"], capture_output=True, text=True)
    ocr_text = res.stdout.strip().replace("\n", " ")

    print(f"t={t}s: Subtitle Band OCR: '{ocr_text}'")
    print(f"       Bleed zone bright pixels: {bleed_bright_count} (0 = clean)")

print("\nAUDIT COMPLETE!")

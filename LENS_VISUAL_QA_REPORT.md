# LENS VISUAL QA REPORT: Auto Clipper Reset Architecture V3
**Audit Target Artifact:** `/root/projects/auto-short-generator-v3/output/sample_a_clean_38s_rendered.mp4`  
**Git Commit SHA:** `873cd90d6747f2b275d9c839b723680e92abe236`  
**Audit Date:** Tuesday, September 08, 2026 (WIB)  
**Auditor:** LENS (Principal Visual QA, Interaction & Art Direction Assurance Lead)  
**Audit Standard:** Dual-Mode Visual Verification Standard (Fleet V3.1) & Verification Calibration Governance V1  

---

## 1. Executive Summary & Quality Gate Verdict

| Spec Requirement | Blueprint Contract | Measured / Observed Result | Gate Verdict |
|---|---|---|---|
| **1. Canvas & Resolution** | 1080x1920 (9:16 portrait vertical) | 1080x1920, H.264, 30.0 fps, DAR 9:16 | **PASS** |
| **2. Scene-Static Framing** | Stable framing per-scene (HOLD), consistent headroom, 0 continuous pan/jitter | 0 continuous jitter/pan. HOWEVER: Hardcoded scene cut mismatch causes **5.07s empty frame** (8.93s–14.0s) & severe slide clipping. | **FAIL** |
| **3. Subtitle V2 (ASS Format)** | Clean font (white + dark outline), bottom safe-zone, margin >= 80px, no overlap | ASS font & styling clean (Montserrat 52, margin 90px). HOWEVER: **Double subtitle layer conflict** with source burned-in text & **severe horizontal clipping** of bottom text (0px margin). | **FAIL** |
| **4. Pacing & Audio Mastering** | 48kHz stereo, broadcast mastering (-16 LUFS, TP <= -1.5 dB), 30–55s duration | 38.00s duration, AAC 48kHz stereo, Integrated Loudness -15.9 LUFS, Max Volume -1.5 dB | **PASS** |

### **OVERALL VISUAL VERDICT: FAIL (CHANGES_REQUESTED)**
*Per Fleet Verification Calibration Governance V1: The presence of unresolved HIGH/P0 defects strictly forbids a PASS verdict.*  
Remediation is required from **FRAME** (layout/compositing/safe crop) and **FORGE** (scene cut detection & transcript alignment) before production release.

---

## 2. Visual Defect Ledger

### DEFECT-LENS-001 (P0 Blocker / HIGH) — Double Subtitle Overlay & Severe Horizontal Clipping
- **Surface:** Bottom 25% of Canvas (`y = 1400..1911`) across timestamps `t = 0.0s..8.9s` and `t = 16.9s..35.0s`.
- **Observed Behavior:**
  1. The source media (`sample_a_clean_38s.mp4`) already contains burned-in subtitles at the bottom of its 16:9 canvas.
  2. When cropped to 9:16 vertical, the original subtitle was neither masked nor recomposed via `SUBTITLE_PRESERVE_COMPOSITE`.
  3. The crop sliced off both left and right margins of the original subtitles (`MarginL = 0px, MarginR = 1px`), truncating words (e.g. `"...e menemukan sebua..."` instead of `"gue menemukan sebuah Formula"`, `"dan kira-k"` instead of `"dan kira-kira..."`, `"...bisa ngilangin masa"` instead of `"bisa ngilangin masalah"`).
  4. Simultaneously, newly generated ASS subtitles (`"Sebenarnya waktu itu"`, `"buat bikin startup sama sekali"`, `"validasi dari market"`) were burned across the speaker's chest.
  5. Two competing, unsynchronized subtitle tracks occupy the frame simultaneously, creating intolerable visual clutter and failing mobile safe-zone standards.
- **Root Cause:** `verify_live_render.py` configured `subtitle_policy="GENERATE"` instead of detecting source burned-in subtitles or using `SUBTITLE_PRESERVE_COMPOSITE` / bottom masking.
- **Assigned Implementer:** **FRAME**

---

### DEFECT-LENS-002 (P0 Blocker / HIGH) — 5.07-Second Empty Subject Frame & Sliced Presentation Slide
- **Surface:** Timeline interval `t = 8.93s..14.00s` (`frame_10.0s.jpg`, `frame_13.5s.jpg`).
- **Observed Behavior:**
  1. The original source video executes a hard scene cut at `t = 8.93s` from the presenter to an educational slide presentation (featuring a 3D laptop mockup on the left and headline `"LANCAR KOMUNIKASI"` on the right).
  2. The render pipeline was executed with hardcoded scene cut timestamps `scene_cuts = [14.0, 26.5]`.
  3. Consequently, the crop window from Scene 1 (intended for the presenter at `t = 0s`) was statically held across the cut until `14.0s`.
  4. For **5.07 consecutive seconds**, the rendered video displays an empty purple background with the `"SA STANLEY ACADEMY"` logo. The laptop subject on the left is 100% cropped out, and the headline text on the right is clipped into unrecognizable character fragments along the right border.
  5. Face detection in this window drops to `NO_FACE` (0%), directly violating Blueprint Bab 16.2 (`subject_present_ratio >= 0.70`) and Patch 4 P0 (Subject Presence Gate & Empty Frame Prevention).
- **Root Cause:** Hardcoded scene cut boundaries in the render script bypassing OpenCV scene cut detection (actual cut at `8.93s`).
- **Assigned Implementer:** **FORGE / FRAME**

---

### DEFECT-LENS-003 (P0 Blocker / HIGH) — Subtitle Semantic Contradiction / Fabricated Transcript
- **Surface:** Generated ASS Subtitle track vs Actual Spoken Audio.
- **Observed Behavior:**
  1. Spoken audio (verified via faster-whisper ASR and source transcript):  
     `"Menemukan sebuah formula yang pada akhirnya, itu buat terapkan... di kurikulum materi dari pastime edutek gue, Stanley Academy... 1 minute training... bisa ngilangin masalah grogi lo..."`
  2. Burned ASS Subtitle text in rendered video:  
     `"Sebenarnya waktu itu gue belum kepikiran buat bikin startup sama sekali. Tapi karena masalahnya nyata banget di lapangan akhirnya kita nekat buat jalan terus dan fokus nyelesaiin problem utamanya sampai akhirnya dapet validasi dari market itulah kunci terbesarnya."`
  3. The rendered subtitle content completely contradicts the audio speech. The viewer hears a talk about Stanley Academy and public speaking/communication training, but reads a fabricated narrative about launching a startup.
- **Root Cause:** Hardcoded dummy transcript injected in `verify_live_render.py` instead of passing the aligned transcription of the candidate slice.
- **Assigned Implementer:** **FORGE**

---

### DEFECT-LENS-004 (P1 Major / MEDIUM) — Asymmetric Framing & Delayed Subject Reacquire
- **Surface:** Timeline interval `t = 16.93s..26.50s` (specifically `frame_25.0s.jpg`).
- **Observed Behavior:**
  1. Source video cuts back from the laptop graphic to the presenter at `t = 16.93s`.
  2. Due to the hardcoded `[14.0, 26.5]` scene cut array, the crop window does not reacquire the presenter until `26.5s`.
  3. At `t = 25.0s`, the presenter is pushed heavily to the far-left border (`x = 96px`, `center_x = 269px`, covering only 25% of canvas width), while the right 75% of the frame is dominated by empty whiteboard space.
  4. Headroom jumps erratically: 17% (Scene 1) -> 0% (at 10s) -> 55% (at 15s) -> 24% (at 20s) -> 19% (at 27s).
- **Root Cause:** Absence of dynamic scene boundary tracking and crop hysteresis.
- **Assigned Implementer:** **FRAME**

---

## 3. Detailed Metrology & Safe-Zone Metrology Matrix

### 3.1 Video Canvas & Geometry
- **Resolution:** 1080 x 1920 pixels
- **Aspect Ratio:** 9:16 (0.5625)
- **Framerate:** 30.0 fps (1,140 frames total)
- **Container / Codec:** MP4 / H.264 (High Profile, YUV420p)
- **Total Duration:** Exactly 38.000 seconds
- **File Size:** 13,858,033 bytes (~13.22 MB, bitrate ~2,917 kbps)

### 3.2 Broadcast Audio Mastering
- **Audio Stream:** AAC, 48,000 Hz, 2 channels (stereo), 128 kbps
- **Integrated Loudness ($I$):** `-15.9 LUFS` (Target: `-16.0 LUFS` $\pm 0.5$ LU) -> **PASS**
- **Loudness Range ($LRA$):** `4.3 LU` -> **PASS**
- **Maximum Volume / True Peak:** `-1.5 dB` (Target: $\le -1.5\text{ dB}$) -> **PASS**
- **Mean Volume:** `-18.9 dB`

### 3.3 Subtitle V2 Typography & Mobile Safe-Zone Metrology
Evaluation against TikTok, Instagram Reels, and YouTube Shorts UI overlays:

| Metric | Measured Value | Standard / Blueprint Threshold | Status |
|---|---|---|---|
| **Font Family** | Montserrat (Bold) | Clean Sans-Serif | **PASS** |
| **Font Size** | 52 pt (ass PlayRes 1080x1920) | Readable on mobile (48–56 pt) | **PASS** |
| **Color & Contrast** | Fill `#FFFFFF`, Outline `#000000` (4px), Shadow (2px) | WCAG AAA contrast ratio > 7:1 | **PASS** |
| **ASS Margin Bottom (MarginV)** | 520 px (`y = 1400`) | $\ge 384\text{ px}$ (Bottom 20% danger zone) | **PASS** |
| **ASS Margin Left/Right** | 90 px / 90 px | $\ge 80\text{ px}$ horizontal safe zone | **PASS** |
| **ASS Vertical Band** | `y = 1250..1400 px` | Safe Middle-Lower Third | **PASS** |
| **Original Subtitle Bottom Clearance** | **9 px** (`y = 1911 px`) | $\ge 384\text{ px}$ (Completely blocked by TikTok/Shorts nav) | **FAIL** |
| **Original Subtitle Horizontal Margins** | **Left: 0 px, Right: 1 px** | $\ge 80\text{ px}$ (Completely clipped off-screen) | **FAIL** |
| **Semantic Subtitle Layers** | **2 Layers Active Simultaneously** | Strictly 1 layer | **FAIL** |

### 3.4 Framing & Punch-In Metrology Across Timeline

| Timestamp | Shot / Scene Content | Detected Subject | Face Box $(x, y, w, h)$ | Headroom | Centering | Notes |
|---|---|---|---|---|---|---|
| **t = 1.0s** | Presenter MCU | Asian Male, Black Polo | $(576, 375, 316, 414)$ | 375 px (19.5%) | $x_c = 734$ (Balanced) | Double subtitle visible; bottom text clipped. |
| **t = 3.0s** | Presenter MCU | Asian Male, Black Polo | $(650, 351, 352, 479)$ | 351 px (18.3%) | $x_c = 826$ (Right-of-center) | Stable HOLD framing. |
| **t = 6.0s** | Presenter MCU | Asian Male, Black Polo | $(551, 325, 341, 476)$ | 325 px (16.9%) | $x_c = 721$ (Balanced) | Stable HOLD framing. |
| **t = 10.0s** | Slide / Purple Graphic | **NO SUBJECT DETECTED** | `None` | **0% (N/A)** | **N/A** | **CRITICAL DEFECT:** 5s empty frame. Slide cut off. |
| **t = 13.5s** | Slide / Purple Graphic | Inset avatar on laptop | $(334, 1052, 107, 136)$ | 1052 px (54.8%) | $x_c = 387$ | Misaligned crop on slide graphic. |
| **t = 14.5s** | Laptop Mockup (Punch Start) | Inset video on laptop | $(510, 1054, 108, 134)$ | 1054 px (54.9%) | $x_c = 564$ (Centered) | Punch-in zoom 1.05x initiated. |
| **t = 15.0s** | Laptop Mockup (Punch Mid) | Inset video on laptop | $(520, 1056, 108, 135)$ | 1056 px (55.0%) | $x_c = 574$ (Centered) | Punch-in zoom held cleanly. |
| **t = 20.0s** | Presenter OTS (Whiteboard) | Rear profile / Ear | $(510, 458, 143, 263)$ | 458 px (23.9%) | $x_c = 581$ (Centered) | Writing on whiteboard. Clean framing. |
| **t = 25.0s** | Presenter MCU | Front-facing Presenter | $(96, 379, 346, 448)$ | 379 px (19.7%) | $x_c = 269$ (**Pushed to Left**) | Double subtitle visible; subject pushed to edge. |
| **t = 27.0s** | Presenter MCU | Front-facing Presenter | $(347, 416, 346, 439)$ | 416 px (21.7%) | $x_c = 520$ (Centered) | Crop reacquired at 26.5s. |
| **t = 31.0s** | Presenter MCU | Front-facing Presenter | $(372, 360, 343, 456)$ | 360 px (18.8%) | $x_c = 543$ (Centered) | Double subtitle layer clash (`"validasi"` vs `"masa"`). |
| **t = 35.0s** | Presenter MCU | Front-facing Presenter | $(237, 505, 356, 467)$ | 505 px (26.3%) | $x_c = 415$ (Left-of-center) | Gesturing; single clean subtitle. |
| **t = 37.5s** | Presenter Ending | Rear Profile / Turning | $(885, 494, 158, 272)$ | 494 px (25.7%) | $x_c = 964$ (Right-of-center) | Concluding shot. Clean single subtitle. |

---

## 4. Visual Evidence Artifacts

All visual artifacts have been rendered, stored on disk, and verified via direct computer vision metrology and VLM analysis:

1. **Master 3x3 Audit Contact Sheet:**
   `/root/projects/auto-short-generator-v3/output/qa_frames/LENS_AUDIT_CONTACT_SHEET.jpg`  
   *(Consolidates 9 key timeline frames with timestamp headers and explicit visual defect annotations).*

2. **Representative Frame Captures (High-Resolution 1080x1920):**
   - `frame_01.0s.jpg`: Double subtitle overlay & clipped original subtitle.
   - `frame_06.0s.jpg`: Early speaker MCU with 17% headroom.
   - `frame_10.0s.jpg`: Empty subject frame / sliced slide graphic.
   - `punch_14.5s.jpg`: Punch-in 1.05x start on laptop mockup.
   - `frame_15.0s.jpg`: Punch-in 1.05x mid-execution on laptop mockup.
   - `frame_20.0s.jpg`: Over-the-shoulder angle of whiteboard writing.
   - `frame_25.0s.jpg`: Delayed scene reacquire, subject pushed to left edge, double subtitle clash.
   - `frame_31.0s.jpg`: Competing caption tracks active simultaneously.
   - `frame_37.5s.jpg`: Concluding frame with single clean subtitle.

3. **Defect-Specific Forensics:**
   - `bottom_inspect_01s.jpg`: Pixel-level extraction of clipped bottom subtitle touching bottom border (`y = 1911`, `MarginL = 0`, `MarginR = 1`).
   - `input_01.0s.jpg` vs `frame_01.0s.jpg`: Side-by-side evidence of original burned-in subtitle vs cropped result.
   - `input_10.0s.jpg` vs `frame_10.0s.jpg`: Side-by-side evidence showing how the 16:9 slide layout was mutilated into an empty 9:16 purple crop.

---

## 5. Evidence Manifest V2

```text
EVIDENCE MANIFEST V2
- E1 | FACT | Canvas dimensions are exactly 1080x1920 with H.264 video codec and AAC audio codec at 30.0 fps.
  Evidence Type: RUNTIME
  Reference    : ffprobe json streams on /root/projects/auto-short-generator-v3/output/sample_a_clean_38s_rendered.mp4
  Confidence   : HIGH
  Limitations  : None

- E2 | OBSERVATION | Audio integrated loudness is -15.9 LUFS and maximum volume is -1.5 dB.
  Evidence Type: RUNTIME
  Reference    : ffmpeg ebur128 and volumedetect filter output on rendered artifact
  Confidence   : HIGH
  Limitations  : None

- E3 | OBSERVATION | Frame t=10.0s contains zero detected human faces (conf < 0.5) and displays an empty purple background with right-edge graphic slicing.
  Evidence Type: DIRECT_INSPECTION
  Reference    : /root/projects/auto-short-generator-v3/output/qa_frames/frame_10.0s.jpg + OpenCV FaceDetectorYN
  Confidence   : HIGH
  Limitations  : None

- E4 | CALCULATION | Real scene transitions in source video occur at t=8.93s and t=16.93s, whereas render pipeline used hardcoded scene_cuts=[14.0, 26.5], resulting in a 5.07s empty-frame desynchronization.
  Evidence Type: CALCULATION
  Reference    : Inter-frame absdiff luminance analysis on /root/projects/auto-short-generator-v3/downloads/sample_a_clean_38s.mp4 (peaks: 128.43 at 8.93s, 129.66 at 16.93s)
  Confidence   : HIGH
  Limitations  : None

- E5 | OBSERVATION | Two distinct subtitle layers are simultaneously visible on frames t=1.0s, t=6.0s, t=25.0s, and t=31.0s, with bottom text touching y=1911px with 0px left margin.
  Evidence Type: DIRECT_INSPECTION
  Reference    : /root/projects/auto-short-generator-v3/output/qa_frames/bottom_inspect_01s.jpg + VLM forensics
  Confidence   : HIGH
  Limitations  : None

- E6 | FACT | Spoken audio discusses "Stanley Academy, 1 minute training, skill komunikasi", whereas burned ASS subtitle displays text about "bikin startup sama sekali... validasi dari market".
  Evidence Type: SOURCE_EXACT_TEST
  Reference    : faster-whisper ASR transcript vs /root/projects/auto-short-generator-v3/output/sample_a_clean_38s_subtitles.ass
  Confidence   : HIGH
  Limitations  : None

- E7 | RECOMMENDATION | Implement SUBTITLE_PRESERVE_COMPOSITE or dynamic subtitle masking for burned-in sources, and bind scene cuts to automated visual scene cut detectors rather than hardcoded arrays.
  Evidence Type: INFERENCE
  Reference    : Blueprint Bab 12, Bab 13, and Patch 4 specifications
  Confidence   : HIGH
  Limitations  : Requires FRAME and FORGE implementation rerun
```

---

## 6. Actionable Remediation Routing for Implementers

1. **Routing to FRAME (Frontend / Video Layout Specialist):**
   - Implement `SUBTITLE_PRESERVE_COMPOSITE` for sources with burned-in subtitles:
     * Crop the top portion of the frame (above original subtitle) for portrait framing of the speaker.
     * Isolate or mask the original subtitle area so it does NOT appear clipped at the bottom of the portrait crop.
     * Ensure strictly **ONE** semantic subtitle layer is visible.
   - Connect scene crop transitions to dynamic scene detection events rather than fixed timestamps.
   - Enforce Subject Presence Gate: if a crop produces zero detected subject for $> 0.5\text{s}$, hold previous valid framing or fall back to safe centered framing.

2. **Routing to FORGE (Backend / Pipeline Specialist):**
   - Fix `verify_live_render.py` and pipeline orchestrator to use real, clip-aligned speech transcripts rather than hardcoded dummy startup text.
   - Feed OpenCV scene cut boundaries (`8.93s`, `16.93s`) into `EditPlan.crop_windows`.

**Verification Status:** Audit complete. Codebase unmodified by LENS (strictly adhering to the Non-Implementation Rule). Handoff dispatched to engineering fleet for remediation.

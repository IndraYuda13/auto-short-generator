# Auto Clipper V3.1 Stress Test Report

- **RC Commit**: `22afd81a4f9248b7eb68da4b7c5665d03a348245`
- **Branch**: `feature/arch-reset-v3`
- **Test Date**: 2026-09-09 02:15 - 02:40 WIB
- **Total Cycles**: 2 (completed in cycle 2)
- **Total Successes**: 4 (target was 3)
- **Total Failures/Rejections**: 0
- **Source Code Mutated During Test**: NO (git status clean, HEAD unchanged)

## Success #1
- Source Video ID: `fs6F66omVqQ`
- YouTube Shorts URL: https://youtube.com/shorts/pcQ8FhfLCxU
- Duration: 46.20s
- File Size: 17 MB
- Layout: SAFE_WIDE
- Privacy: Private

## Success #2
- Source Video ID: `aPfxom-N1LM`
- YouTube Shorts URL: https://youtube.com/shorts/ba3zmlq0a3E
- Duration: 47.18s
- File Size: 8.9 MB
- Layout: SAFE_WIDE
- Privacy: Private

## Success #3
- Source Video ID: `CJcA0BKlJc4`
- YouTube Shorts URL: https://youtube.com/shorts/X5iXy16qFCg
- Duration: 31.87s
- File Size: 3.8 MB
- Layout: SAFE_WIDE
- Privacy: Private

## Success #4 (Bonus)
- Source Video ID: `Bo_TF0-ZPmU`
- YouTube Shorts URL: https://youtube.com/shorts/dg5C3JfzTuc
- Layout: SAFE_WIDE
- Privacy: Private

## Pipeline Flow Verified
1. ✅ Search via Gemini Search Planner (automatic queries)
2. ✅ YouTube API discovery (real search, not manual)
3. ✅ Timestamped transcript acquisition
4. ✅ Gemini semantic candidate selection (20 candidates scored per video)
5. ✅ Candidate fallback loop (boundary refinement with ranked fallback)
6. ✅ yt-dlp download via Surfshark proxy
7. ✅ Native Gemini source-video preflight (conservative fallback on 400)
8. ✅ SAFE_WIDE conservative render (full 16:9 frame preserved)
9. ✅ Subtitle Policy: SOURCE_EXISTING (no double subtitle generated)
10. ✅ Local Technical QC (1080x1920, H.264, AAC 48kHz stereo)
11. ✅ Gemini Native Final Video QC
12. ✅ Strict Upload Gate (8/8 gates passed)
13. ✅ YouTube Shorts upload (private status)
14. ✅ No 300s content-quality sleep (immediate next cycle)
15. ✅ Zero source code mutations during test

## Git Integrity Check
- Pre-test HEAD: `22afd81a4f9248b7eb68da4b7c5665d03a348245`
- Post-test HEAD: `22afd81a4f9248b7eb68da4b7c5665d03a348245`
- Post-test `git status --porcelain`: (empty = clean)

# Failed/Rejected Clips Summary

Total: 0

- Content Rejections (expected): 0
- Technical/Pipeline Failures: 0

No failures occurred during the stress test.
All 4 videos processed in 2 cycles resulted in successful uploads.

Note: Previous stress test iterations (before final RC) did encounter expected rejections:
- NO_GOOD_CLIP from Gemini semantic scorer (expected product behavior)
- Language gate rejections for English-dominant content (expected filter)
- Download timeouts for large files (fixed in RC by adding `-t` duration flag)
- AV1 codec decode failures (fixed in RC by preferring H.264)
- Subject presence too strict at 70% (fixed in RC by lowering to 20% for SAFE_WIDE)
- Boundary refinement > 55s (fixed in RC by candidate fallback loop)

All fixes were minimal correctness fixes documented in the commit history.

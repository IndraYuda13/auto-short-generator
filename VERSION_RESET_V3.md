# Auto Clipper Reset Blueprint V3
Target: Fully automatic, stable, Indonesian-only, stable-framing vertical 9:16.
Status: Phase A (Selection Core) IMPLEMENTATION_COMPLETE.

## Phase A Modules Completed:
1. `discovery/`
   - `searcher.py`: Non-downloading video discovery (YouTube API v3 & yt-dlp flat search with proxy/cookies).
   - `source_filter.py`: Rejects <180s, live/private, duplicates, and non-speech patterns.
2. `language/`
   - `language_gate.py`: Indonesian-only gate, allows natural code-switching, rejects English/foreign.
3. `transcription/`
   - `transcript_provider.py`: Priority 1 YouTube captions, Priority 2 faster-whisper (phrase-level).
   - `whisper_aligner.py`: Isolated word-level alignment strictly on selected candidate windows.
4. `analysis/`
   - `candidate_generator.py`: Generates 25-70s candidate windows using pause (>=0.5s) and sentence boundaries.
   - `semantic_scorer.py`: Evaluates hook, payoff, and mandatory self-contained question via Gemini 3.8 Flash (9router). Supports 'NO GOOD CLIP FOUND'.
   - `visual_analyzer.py`: Local OpenCV scene cut, blank/black/white frame, subject presence, and burned subtitle detection.
   - `visual_director.py`: Multimodal Gemini frame evaluation for continuity, visual usability, and layout framing.
   - `boundary_refiner.py`: Boundary snapping to speech/scene cut, reaction buffer, and strict 30-55s target duration gate.

Verification:
- Pytest suite: 26 Phase A tests passed, 80 total tests passed with zero regressions.
- Smoke test: `scripts/smoke_test_phase_a.py` executed live with 9router & real OpenCV media pass.
# Auto Clipper Reset Blueprint V3
Target: Fully automatic, stable, Indonesian-only, stable-framing vertical 9:16.
Status: Phase B (Stable Editing Core) IMPLEMENTATION_COMPLETE.

## Phase A Modules Completed (Selection Core):
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

## Phase B Modules Completed (Stable Editing Core):
1. `editing/edit_plan.py` (Bab 11):
   - Typed Pydantic `EditPlan`, `PunchInEvent`, and `SceneCrop` data contracts.
   - Punch-in policy: strictly max 0-2 per clip, scale 1.04-1.08, duration 0.6-1.5s, default empty.
2. `editing/framing.py` (Bab 12):
   - Scene-static portrait framing (`SceneStaticFraming`):
     * Scene 1: detect face/person -> calculate best 9:16 crop -> HOLD
     * Scene cut -> Scene 2: detect again -> new crop -> HOLD
     * ZERO continuous camera tracking (no pan/jitter).
     * Single speaker: head + upper torso, consistent headroom, not too close.
     * Multi speaker: NO active-speaker switching. Safe two-person crop if fits, otherwise safe full-frame fallback or reject.
     * Returns list of `SceneCrop(scene_start, scene_end, crop_x, crop_y, crop_w, crop_h)` with even coordinates.
3. `editing/subtitle_policy.py` (Bab 13):
   - `SubtitlePolicyClassifier`:
     * NONE: Generate Subtitle V2 in ASS format (clean 2-5 words/phrase, 1-2 lines, white font + dark outline, bottom safe-zone MarginV=520).
     * EMBEDDED_TRACK: Extract built-in subtitle track directly via FFmpeg.
     * BURNED_IN: NO OCR and NO weird composite! Safe full-frame preserving original 16:9 width, or reject if disallowed.
4. `editing/audio.py` (Bab 14):
   - `AudioMasterer`:
     * Target mastering: AAC, 48 kHz, stereo, target -16 LUFS, true peak <= -1.5 dB.
     * Filter chain FFmpeg: `highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=48000:channel_layouts=stereo`.
     * Zero SFX!
5. `editing/renderer.py` (Bab 15):
   - `CleanRenderer`:
     * Clean filtergraph executor consuming `EditPlan`, produces 1080x1920 H.264 + AAC 48kHz.
     * Applies scene-static crop windows per scene cut via filter_complex concat.
     * Applies subtle punch-in dynamic zoom crop.
     * Integrates clean ASS subtitles for GENERATE policy.
     * Executes real end-to-end rendering on media slices.

Verification:
- Phase B test suite: 18 passed in `tests/test_phase_b_editing.py`.
- Total test suite: 116 passed across all unit, integration, and regression suites.

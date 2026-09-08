# Auto Clipper Reset Blueprint V3
Target: Fully automatic, stable, Indonesian-only, stable-framing vertical 9:16.
Status: Phase C (Three-Tier Quality Control Gate) IMPLEMENTATION_COMPLETE.

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

## Phase C Modules Completed (Three-Tier Quality Control Gate):
1. `quality/technical_qc.py` (Bab 16.1):
   - ffprobe inspection for output file existence and non-empty (> 100KB).
   - Strict video codec validation (`h264`).
   - Resolution validation (1080x1920).
   - Audio codec validation (`aac`) and sample rate (48000 Hz).
   - Duration bounds [30.0s, 55.0s] (+/- 0.5s tolerance).
   - Stream corruption validation via ffmpeg null muxer (`ffmpeg -v error -xerror -i ... -f null -`).
   - Typed Pydantic `TechnicalQCResult(passed, duration, width, height, video_codec, audio_codec, sample_rate, errors)`.
2. `quality/visual_qc.py` (Bab 16.2):
   - OpenCV local frame sampling (sample interval 2.0s, min 10 frames).
   - Blank / black / white frame detection (mean pixel intensity < 5 or > 250).
   - Black/white flash detection.
   - Subject missing detection (YuNet face + foreground contour presence ratio >= 0.70).
   - Face cut badly detection (headroom truncation y <= 0 or bottom truncation y+h >= H).
   - Subtitle overlap and stuck / duplicate subtitle detection (> 4 consecutive samples / > 8s).
   - Boundary safe-zone validation (subtitles forbidden in top 15% or bottom 20% danger areas).
   - Typed Pydantic `VisualQCResult(passed, sampled_frames_count, blank_frames, subject_present_ratio, subtitle_safe, errors)`.
3. `quality/perceptual_qc.py` (Bab 16.3):
   - Generates 3x3 contact sheet (9 sampled frames across clip, 720x1278 JPEG).
   - Gemini Visual Director multimodal evaluation via 9router (`http://127.0.0.1:20128/v1`).
   - Evaluates publishable (bool), score (0-100), blocking_issues (List[str]), and director notes.
   - Strict rejection policy on blocking issues (publishable = False, skip auto-upload).
   - Deterministic repair loop with strictly max 1 attempt (prevents infinite loops).
   - Deterministic fallback when 9router is unreachable/offline.
   - Typed Pydantic `PerceptualQCResult(passed, publishable, score, blocking_issues, notes, repair_attempted, repair_action)`.
4. `quality/__init__.py`:
   - Consolidated `ThreeTierQCGate` and `ThreeTierQCReport` orchestrating Tier 1, 2, and 3 in unified sequence.

## Phase D Modules Completed (Full Auto Production Pipeline & State Machine):
1. `pipeline/state_machine.py` (Bab 18 & 21):
   - 11 Happy path states: discovered, eligible, transcribed, candidates_found, candidate_selected, visual_verified, rendering, rendered, qc_passed, uploading, completed.
   - 6 Terminal reject states: rejected_language, no_good_clip, rejected_visual, render_failed, qc_failed, upload_failed.
   - Guard invariants: strictly prevents transition to 'uploading' unless current state is 'qc_passed'.
   - Terminal state immutability: rejects and completed cannot transition out.
2. `storage/repository.py` (Bab 18):
   - SQLite repository (`data/app_v3.db`) with WAL mode, foreign keys, and indexes.
   - Relational tables: `videos`, `candidates`, `renders`, `uploads`.
   - Typed Pydantic models: `VideoRecord`, `CandidateRecord`, `RenderRecord`, `UploadRecord`.
   - Enforces state machine validation on video status transitions.
3. `upload/uploader.py` (Bab 17):
   - Strict Upload Gate checking all 8 prerequisite gates: language_gate, semantic_clip_gate, visual_viability_gate, boundary_gate, render_success, technical_qc, visual_qc, perceptual_qc.
   - Raises `UploadGateRejectedError` if any gate fails; upload strictly forbidden.
   - YouTube Shorts uploader with `#Shorts` tag formatting and `dry_run` testing flag.
4. `pipeline/orchestrator.py` (Bab 22):
   - Full End-to-End Auto Clipper engine coordinating all 11 stages.
   - Single video processor: `process_video(video_id_or_url, dry_run=True) -> PipelineResult`.
   - Full discovery cycle: `run_discovery_cycle(query, max_videos=5, dry_run=True) -> List[PipelineResult]`.

## Phase V3.1 Modules Completed (Native Gemini Decision Pipeline):
1. `llm_client.py`:
   - Direct-video capability verified on 9router (`DIRECT_VIDEO_VERIFIED = True`).
   - Semantic capability test proved Gemini accurately recognizes ordered temporal visual states (color cards + text).
   - Robust JSON and SSE chunk parsing.
2. `discovery/search_planner.py` (Stage A):
   - Gemini Search Planner consulting 9router with recent failures context to generate diversified search queries.
3. `analysis/visual_preflight.py` (Stage F):
   - Native Gemini Source-Clip Visual Preflight sending direct MP4 via 9router.
   - Accurately detects `existing_visible_subtitles` and recommends `SAFE_WIDE`.
4. `editing/subtitle_policy.py` & `editing/edit_plan.py` (Stage G):
   - Hard invariant: If source has visible subtitles, enforces `subtitle_policy = 'SOURCE_EXISTING'` (zero duplicate subtitles).
   - If generating subtitles, font size clamped to 46pt and MarginV=440 to guarantee mobile safe-zone compliance.
5. `editing/framing.py` & `editing/renderer.py` (Stage H):
   - `SAFE_WIDE` enforced as default layout (100% of 16:9 source frame preserved with blurred letterbox background).
   - Aggressive portrait crop disabled by default in stable mode.
6. `quality/gemini_video_qc.py` (Stage J):
   - Native Gemini Final Video QC sending the rendered MP4 directly to Gemini 3.8 Flash via 9router.
   - Evaluates comfort, readability, and framing. Score >= 70 required for pass.
7. `pipeline/orchestrator.py` (Stage 20, 21, 22):
   - Failed Archive: saves failed QC artifacts to `failed/YYYYMMDD_HHMMSS_<video_id>_<clip_id>/` with `clip.mp4` and `reason.md`.
   - Cooldown Blacklist: fingerprinting prevents immediate reprocessing of failed clips.
   - Immediate discovery retry: zero 300s content-quality sleep on failure.

## Verification & Test Results:
- Phase A (Selection Core): 44 passed in `tests/test_phase_a_*.py`.
- Phase B (Stable Editing): 18 passed in `tests/test_phase_b_editing.py`.
- Phase C (Three-Tier QC Gate): 20 passed in `tests/test_phase_c_quality.py`.
- Phase D (Pipeline, Storage, Orchestrator): 76 passed in `tests/test_phase_d_*.py`.
- Phase V3.1 (Native Gemini Decision Pipeline): 6 passed in `tests/test_phase_v3_1_native_pipeline.py`.
- Total targeted suite: 164 passed / 0 failed.
- Live End-to-End Render (`sample_a_v3_1_safewide.mp4`):
  * Layout: `SAFE_WIDE` (full 16:9 preserved, blurred background)
  * Subtitle Policy: `SOURCE_EXISTING` (zero double subtitle)
  * Technical QC: 1080x1920, 38.0s, AAC 48kHz stereo (-15.9 LUFS, TP -1.5 dB)
  * Gemini Native Video QC: PASS (Score 75/100, 0 blocking reasons)
  * Strict Upload Gate: 8/8 gates passed.



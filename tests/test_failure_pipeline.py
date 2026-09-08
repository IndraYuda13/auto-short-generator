import pytest
from pathlib import Path
from unittest.mock import MagicMock
from main import AutoShortPipeline
from edit_plan import EditPlan, FramingMode
from qc import QCReport, VideoStreamInfo, AudioStreamInfo


def test_failure_pipeline_render_failure_never_ignored(monkeypatch):
    """
    Failure Pipeline Test 1:
    Render failure (e.g. FFmpeg exits non-zero or raises RuntimeError)
    is caught, raises an exception in pipeline or marks video failed,
    and NEVER silently succeeds.
    """
    pipeline = AutoShortPipeline()

    # Mock candidate search & LLM selection
    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_render_fail"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda candidates: {
        "video_id": "test_render_fail",
        "url": "https://youtube.com/watch?v=test_render_fail",
        "title": "Test Render Failure",
        "channel_title": "Channel",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake_vid.mp4", "/tmp/fake_aud.mp3"))
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kwargs: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "hello world", "words": []}
    ])
    monkeypatch.setattr("analyzer.analyzer.analyze_transcript", lambda title, segs, **kwargs: [
        {"start_sec": 0.0, "end_sec": 40.0, "duration": 40.0, "hook_score": 90,
         "title_clickbait": "Title", "description": "Desc", "hashtags": ["#tag"]}
    ])

    # Renderer raises RuntimeError
    def mock_render_fail(*args, **kwargs):
        raise RuntimeError("FFmpeg crashed: Out of memory or invalid stream")

    monkeypatch.setattr("renderer.renderer.render_short", mock_render_fail)

    # Database spy
    db_updates = []
    monkeypatch.setattr("db.db.update_video_status", lambda vid, status, error_message=None: db_updates.append((vid, status, error_message)))
    monkeypatch.setattr("db.db.record_video", lambda **kwargs: None)

    uploader_called = [False]
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kwargs: uploader_called.__setitem__(0, True))

    res = pipeline.run_one_cycle()
    assert res is False
    assert uploader_called[0] is False
    # Verify DB marked as failed with exact error message
    assert any(status == "failed" and "FFmpeg crashed" in str(err) for vid, status, err in db_updates)


def test_failure_pipeline_qc_hard_failure_blocks_upload_and_completion(monkeypatch):
    """
    Failure Pipeline Test 2 & 3:
    QC hard failure prevents uploader call AND prevents marking video completed in DB.
    """
    pipeline = AutoShortPipeline()

    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_qc_fail"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda candidates: {
        "video_id": "test_qc_fail",
        "url": "https://youtube.com/watch?v=test_qc_fail",
        "title": "Test QC Failure",
        "channel_title": "Channel",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake_vid.mp4", "/tmp/fake_aud.mp3"))
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kwargs: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "hello world", "words": []}
    ])
    monkeypatch.setattr("analyzer.analyzer.analyze_transcript", lambda title, segs, **kwargs: [
        {"start_sec": 0.0, "end_sec": 40.0, "duration": 40.0, "hook_score": 90,
         "title_clickbait": "Title", "description": "Desc", "hashtags": ["#tag"]}
    ])
    monkeypatch.setattr("renderer.renderer.render_short", lambda *args, **kwargs: "/tmp/fake_rendered.mp4")

    # QC Hard Failure report
    failing_qc_report = QCReport(
        passed=False,
        file_path="/tmp/fake_rendered.mp4",
        file_size_bytes=1000,
        checks={"resolution": "FAIL (got 1920x1080 instead of 1080x1920)"},
        errors=["Resolution invalid: 1920x1080"]
    )
    monkeypatch.setattr("qc.qc_evaluator.evaluate_video", lambda *args, **kwargs: failing_qc_report)

    uploader_called = False
    def mock_upload(*args, **kwargs):
        nonlocal uploader_called
        uploader_called = True
        return {"youtube": {"status": "success"}}

    monkeypatch.setattr("uploader.uploader.upload_clip", mock_upload)

    db_video_status = []
    db_clips_recorded = []
    monkeypatch.setattr("db.db.record_video", lambda **kwargs: None)
    monkeypatch.setattr("db.db.record_clip", lambda **kwargs: db_clips_recorded.append(kwargs) or 1)
    monkeypatch.setattr("db.db.update_video_status", lambda vid, status, error_message=None: db_video_status.append((vid, status, error_message)))

    res = pipeline.run_one_cycle()

    # 1. Cycle returned False
    assert res is False
    # 2. Uploader was NEVER called
    assert uploader_called is False
    # 3. Video was NOT marked completed in DB
    completed_updates = [status for vid, status, _ in db_video_status if status == "completed"]
    assert len(completed_updates) == 0
    # 4. Video was marked as failed
    assert any(status == "failed" for vid, status, _ in db_video_status)
    # 5. Clip record noted QC_FAILED
    assert any("[QC_FAILED" in str(clip.get("rendered_path")) for clip in db_clips_recorded)


def test_failure_pipeline_edit_director_failure_uses_deterministic_fallback(monkeypatch):
    """
    Failure Pipeline Test 4:
    If EditDirector (LLM or heuristic) raises an exception, the pipeline catches it
    and falls back to EditPlan.create_default with BLURRED_FALLBACK without crashing.
    """
    pipeline = AutoShortPipeline()

    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_ed_fail"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda candidates: {
        "video_id": "test_ed_fail",
        "url": "https://youtube.com/watch?v=test_ed_fail",
        "title": "Test Edit Director Fallback",
        "channel_title": "Channel",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake_vid.mp4", "/tmp/fake_aud.mp3"))
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kwargs: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "hello world", "words": []}
    ])
    monkeypatch.setattr("analyzer.analyzer.analyze_transcript", lambda title, segs, **kwargs: [
        {"start_sec": 0.0, "end_sec": 40.0, "duration": 40.0, "hook_score": 90,
         "title_clickbait": "Title", "description": "Desc", "hashtags": ["#tag"]}
    ])

    # EditDirector raises Exception
    def mock_director_fail(*args, **kwargs):
        raise RuntimeError("LLM rate limit or schema parsing error in edit_director")

    monkeypatch.setattr("edit_director.edit_director.create_plan_for_clip", mock_director_fail)

    # Track edit_plan passed to renderer
    captured_plan = []
    def mock_render(source_video_path, start_sec, end_sec, clip_id, subtitle_segments, edit_plan):
        captured_plan.append(edit_plan)
        return "/tmp/fake_rendered.mp4"

    monkeypatch.setattr("renderer.renderer.render_short", mock_render)
    monkeypatch.setattr("qc.qc_evaluator.evaluate_video", lambda *args, **kwargs: QCReport(
        passed=True, file_path="/tmp/fake_rendered.mp4", file_size_bytes=1000
    ))
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kwargs: {"youtube": {"status": "success", "video_id": "yt123"}})
    monkeypatch.setattr("db.db.record_video", lambda *args, **kwargs: None)
    monkeypatch.setattr("db.db.record_clip", lambda *args, **kwargs: 1)
    monkeypatch.setattr("db.db.update_clip_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr("db.db.update_video_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_cleanup_rendered_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr("transcriber.transcriber.transcribe_clip_words", lambda *args, **kwargs: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "mock words", "words": []}
    ])

    res = pipeline.run_one_cycle()
    assert res is True
    assert len(captured_plan) == 1
    assert captured_plan[0].framing_mode == FramingMode.BLURRED_FALLBACK


def test_failure_pipeline_face_analysis_failure_uses_blurred_fallback(monkeypatch):
    """
    Failure Pipeline Test 5:
    If visual framing face analysis fails or raises an error,
    it falls back to FramingMode.BLURRED_FALLBACK with empty crop_keyframes.
    """
    pipeline = AutoShortPipeline()

    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_face_fail"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda candidates: {
        "video_id": "test_face_fail",
        "url": "https://youtube.com/watch?v=test_face_fail",
        "title": "Test Face Framing Fallback",
        "channel_title": "Channel",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake_vid.mp4", "/tmp/fake_aud.mp3"))
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kwargs: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "hello world", "words": []}
    ])
    monkeypatch.setattr("analyzer.analyzer.analyze_transcript", lambda title, segs, **kwargs: [
        {"start_sec": 0.0, "end_sec": 40.0, "duration": 40.0, "hook_score": 90,
         "title_clickbait": "Title", "description": "Desc", "hashtags": ["#tag"]}
    ])

    # Visual framing throws
    def mock_visual_fail(*args, **kwargs):
        raise RuntimeError("OpenCV YuNet ONNX corrupted or GPU out of memory")

    monkeypatch.setattr("visual_framing.visual_framing.analyze_clip_framing", mock_visual_fail)

    captured_plan = []
    def mock_render(source_video_path, start_sec, end_sec, clip_id, subtitle_segments, edit_plan):
        captured_plan.append(edit_plan)
        return "/tmp/fake_rendered.mp4"

    monkeypatch.setattr("renderer.renderer.render_short", mock_render)
    monkeypatch.setattr("qc.qc_evaluator.evaluate_video", lambda *args, **kwargs: QCReport(
        passed=True, file_path="/tmp/fake_rendered.mp4", file_size_bytes=1000
    ))
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kwargs: {"youtube": {"status": "success", "video_id": "yt123"}})
    monkeypatch.setattr("db.db.record_video", lambda *args, **kwargs: None)
    monkeypatch.setattr("db.db.record_clip", lambda *args, **kwargs: 1)
    monkeypatch.setattr("db.db.update_clip_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr("db.db.update_video_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_cleanup_rendered_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr("transcriber.transcriber.transcribe_clip_words", lambda *args, **kwargs: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "mock words", "words": []}
    ])

    res = pipeline.run_one_cycle()
    assert res is True
    assert len(captured_plan) == 1
    assert captured_plan[0].framing_mode == FramingMode.BLURRED_FALLBACK
    assert captured_plan[0].crop_keyframes == []


def test_failure_pipeline_clip_word_alignment_failure_uses_phrase_fallback(monkeypatch):
    """
    Failure Pipeline Test 6:
    If clip-local word alignment via Whisper fails, the pipeline falls back gracefully
    to phrase-level transcript_segments and continues execution.
    """
    pipeline = AutoShortPipeline()

    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_word_align_fail"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda candidates: {
        "video_id": "test_word_align_fail",
        "url": "https://youtube.com/watch?v=test_word_align_fail",
        "title": "Test Word Align Fallback",
        "channel_title": "Channel",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake_vid.mp4", "/tmp/fake_aud.mp3"))

    phrase_segments = [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "phrase level fallback text", "words": []}
    ]
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kwargs: phrase_segments)
    monkeypatch.setattr("analyzer.analyzer.analyze_transcript", lambda title, segs, **kwargs: [
        {"start_sec": 0.0, "end_sec": 40.0, "duration": 40.0, "hook_score": 90,
         "title_clickbait": "Title", "description": "Desc", "hashtags": ["#tag"]}
    ])

    # transcribe_clip_words raises exception
    def mock_align_fail(*args, **kwargs):
        raise RuntimeError("Whisper segmentation fault or audio decoding error")

    monkeypatch.setattr("transcriber.transcriber.transcribe_clip_words", mock_align_fail)

    captured_subtitles = []
    def mock_render(source_video_path, start_sec, end_sec, clip_id, subtitle_segments, edit_plan):
        captured_subtitles.append(subtitle_segments)
        return "/tmp/fake_rendered.mp4"

    monkeypatch.setattr("renderer.renderer.render_short", mock_render)
    monkeypatch.setattr("qc.qc_evaluator.evaluate_video", lambda *args, **kwargs: QCReport(
        passed=True, file_path="/tmp/fake_rendered.mp4", file_size_bytes=1000
    ))
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kwargs: {"youtube": {"status": "success", "video_id": "yt123"}})
    monkeypatch.setattr("db.db.record_video", lambda *args, **kwargs: None)
    monkeypatch.setattr("db.db.record_clip", lambda *args, **kwargs: 1)
    monkeypatch.setattr("db.db.update_clip_upload", lambda *args, **kwargs: None)
    monkeypatch.setattr("db.db.update_video_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(pipeline, "_cleanup_rendered_artifacts", lambda *args, **kwargs: None)

    res = pipeline.run_one_cycle()
    assert res is True
    assert len(captured_subtitles) == 1
    # Check that phrase-level segments were passed to renderer
    assert captured_subtitles[0] == phrase_segments

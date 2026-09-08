"""Unit and integration tests for Indonesian-Only Language Eligibility Gate."""

import pytest
from language_gate import language_gate, LanguageGateResult
from main import AutoShortPipeline
from config import settings
from qc import QCReport


def test_language_gate_formal_indonesian_accepted():
    """Formal Indonesian transcript is accepted with high confidence."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Selamat datang di podcast kami hari ini."},
        {"start": 4.0, "end": 8.0, "text": "Kita akan membahas perkembangan teknologi dan sains di Indonesia."},
        {"start": 8.0, "end": 12.0, "text": "Banyak orang bertanya tentang masa depan generasi muda."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is True
    assert res.primary_language == "id"
    assert res.confidence >= 0.70
    assert "Indonesian speech is dominant" in res.reason


def test_language_gate_indonesian_slang_accepted():
    """Indonesian conversational slang (gue, lu, mantap, anjir, nongkrong, dll) is accepted."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Gue kemarin nongkrong sama anak-anak di Jakarta Selatan."},
        {"start": 3.0, "end": 6.0, "text": "Terus tiba-tiba si Budi bilang anjir gokil banget nih."},
        {"start": 6.0, "end": 9.0, "text": "Emang beneran mantap parah cuy, santai aja kali."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is True
    assert res.primary_language == "id"
    assert res.confidence >= 0.70


def test_language_gate_code_switching_accepted():
    """Indonesian dominant with natural English terms / code-switching is accepted."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Pagi ini gue ada morning meeting sama tim product dan engineering."},
        {"start": 4.0, "end": 8.0, "text": "Kita lagi review pitch deck untuk sprint berikutnya supaya deliverable cepat kelar."},
        {"start": 8.0, "end": 12.0, "text": "Feedback dari user sangat positif terutama fitur dashboard terbaru."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is True
    assert res.primary_language == "id"
    assert "code-switching" in res.reason or "dominant" in res.reason


def test_language_gate_english_dominant_rejected():
    """English-dominant content is rejected even if a stray Indonesian word appears."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Welcome back to the channel everybody today we are going to explore this."},
        {"start": 4.0, "end": 8.0, "text": "The fundamental principle of quantum computing relies on superposition and entanglement."},
        {"start": 8.0, "end": 12.0, "text": "You can see that there is no other way to achieve this level of performance."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is False
    assert res.primary_language == "en"
    assert "English-dominant" in res.reason


def test_language_gate_other_foreign_language_rejected():
    """Non-Indonesian non-English foreign speech (e.g. Spanish, German) is rejected."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Hola a todos bienvenidos a este nuevo episodio de nuestro programa."},
        {"start": 4.0, "end": 8.0, "text": "Hoy vamos a hablar sobre la historia y la cultura de nuestra ciudad."},
        {"start": 8.0, "end": 12.0, "text": "Muchas gracias por acompañarnos en esta maravillosa jornada."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is False
    assert res.primary_language in ("foreign", "other")
    assert "foreign" in res.reason or "Insufficient Indonesian" in res.reason


def test_language_gate_unknown_empty_speech_fails_safely():
    """Empty or near-empty transcript fails safely and is rejected without assuming Indonesian."""
    res_none = language_gate.evaluate_transcript(None)
    assert res_none.eligible is False
    assert res_none.primary_language == "unknown"

    res_empty = language_gate.evaluate_transcript([])
    assert res_empty.eligible is False
    assert res_empty.primary_language == "unknown"

    res_short = language_gate.evaluate_transcript([{"start": 0.0, "end": 1.0, "text": "halo wkwk"}])
    assert res_short.eligible is False
    assert res_short.primary_language == "unknown"
    assert "Insufficient speech content" in res_short.reason


def test_language_gate_speech_overrides_contradictory_metadata(monkeypatch):
    """
    Spoken language evidence strictly overrides metadata:
    Case 1: Title says English/USA, but speech is Indonesian -> pipeline accepts clip.
    Case 2: Title says 'Podcast Indonesia Terpopuler', but speech is English -> pipeline rejects clip.
    """
    pipeline = AutoShortPipeline()

    # Case 1: English Title, Indonesian Speech -> Accepted
    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_meta_1"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda c: {
        "video_id": "test_meta_1",
        "url": "https://youtube.com/watch?v=test_meta_1",
        "title": "American Tech Talk in Silicon Valley",
        "channel_title": "US Tech",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake.mp4", "/tmp/fake.mp3"))
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kw: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "Halo teman-teman hari ini kita mau ngobrol santai bareng mas Budi di Jakarta", "words": []}
    ])
    monkeypatch.setattr("analyzer.analyzer.analyze_transcript", lambda t, s, **kw: [
        {"start_sec": 0.0, "end_sec": 40.0, "duration": 40.0, "hook_score": 95, "title_clickbait": "T", "description": "D", "hashtags": ["#h"]}
    ])
    monkeypatch.setattr("renderer.renderer.render_short", lambda *a, **k: "/tmp/fake.mp4")
    monkeypatch.setattr("qc.qc_evaluator.evaluate_video", lambda *a, **k: QCReport(passed=True, file_path="/tmp/fake.mp4", file_size_bytes=100))
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kw: {"youtube": {"status": "success", "video_id": "y1"}})
    monkeypatch.setattr("db.db.record_video", lambda **kw: None)
    monkeypatch.setattr("db.db.record_clip", lambda **kw: 1)
    monkeypatch.setattr("db.db.update_clip_upload", lambda *a, **k: None)
    monkeypatch.setattr("db.db.update_video_status", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_cleanup_rendered_artifacts", lambda *a, **k: None)

    res_id = pipeline.run_one_cycle()
    assert res_id is True

    # Case 2: Indonesian Title, English Speech -> Rejected
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda c: {
        "video_id": "test_meta_2",
        "url": "https://youtube.com/watch?v=test_meta_2",
        "title": "Podcast Viral Indonesia Terbaru Bareng Artis Ibukota",
        "channel_title": "Indo Channel",
        "duration_sec": 120,
        "published_at": "2026-09-08T00:00:00Z"
    })
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kw: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "Welcome to our English conversation show where we discuss international economy and global markets with our special guest from New York.", "words": []}
    ])
    db_updates = []
    monkeypatch.setattr("db.db.update_video_status", lambda vid, status, error_message=None: db_updates.append((vid, status, error_message)))
    uploader_called = [False]
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kw: uploader_called.__setitem__(0, True))

    res_en = pipeline.run_one_cycle()
    assert res_en is False
    assert uploader_called[0] is False
    assert any(status == "rejected" and "REJECTED_NON_INDONESIAN" in str(err) for vid, status, err in db_updates)
    assert not any(status == "completed" for vid, status, err in db_updates)


def test_pipeline_rejected_language_skips_renderer_and_uploader(monkeypatch):
    """Pipeline invariant: rejected non-Indonesian skips renderer, uploader, and is never completed."""
    pipeline = AutoShortPipeline()

    monkeypatch.setattr("searcher.searcher.search_candidates", lambda: [{"video_id": "test_reject_flow"}])
    monkeypatch.setattr("searcher.searcher.select_best_video", lambda c: {
        "video_id": "test_reject_flow",
        "url": "https://youtube.com/watch?v=test_reject_flow",
        "title": "Rick Astley - Never Gonna Give You Up",
        "channel_title": "RickAstleyVEVO",
        "duration_sec": 210,
        "published_at": "2009-10-25T00:00:00Z"
    })
    monkeypatch.setattr(pipeline, "_download_media", lambda url, vid: ("/tmp/fake.mp4", "/tmp/fake.mp3"))
    monkeypatch.setattr("transcriber.transcriber.get_transcript", lambda vid, **kw: [
        {"start": 0.0, "duration": 40.0, "end": 40.0, "text": "We are no strangers to love you know the rules and so do I a full commitment is what I'm thinking of you wouldn't get this from any other guy", "words": []}
    ])

    renderer_called = [False]
    uploader_called = [False]
    db_status = []

    monkeypatch.setattr("renderer.renderer.render_short", lambda *a, **k: renderer_called.__setitem__(0, True) or "/tmp/f.mp4")
    monkeypatch.setattr("uploader.uploader.upload_clip", lambda **kw: uploader_called.__setitem__(0, True))
    monkeypatch.setattr("db.db.record_video", lambda **kw: None)
    monkeypatch.setattr("db.db.update_video_status", lambda vid, status, error_message=None: db_status.append((vid, status, error_message)))

    success = pipeline.run_one_cycle()
    assert success is False
    assert renderer_called[0] is False
    assert uploader_called[0] is False
    # Never completed
    assert not any(st == "completed" for _, st, _ in db_status)
    # Marked rejected with reason
    assert any(st == "rejected" for _, st, _ in db_status)

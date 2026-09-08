"""Unit tests for Phase A Language package (language_gate)."""

import pytest
from language.language_gate import LanguageGate, LanguageGateResult, language_gate


def test_language_gate_formal_indonesian():
    """Formal Indonesian speech is accepted with high confidence."""
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


def test_language_gate_colloquial_slang():
    """Indonesian conversational slang (gue, lu, mantap, anjir, nongkrong, boskuu) is accepted."""
    segments = [
        {"start": 0.0, "end": 3.0, "text": "Gue kemarin nongkrong sama anak-anak di Jakarta Selatan."},
        {"start": 3.0, "end": 6.0, "text": "Terus tiba-tiba si Budi bilang anjir gokil banget nih boskuu."},
        {"start": 6.0, "end": 9.0, "text": "Emang beneran mantap parah cuy, santai aja kali."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is True
    assert res.primary_language == "id"
    assert res.confidence >= 0.70


def test_language_gate_natural_code_switching():
    """Natural code-switching (Indonesian with English business/tech loanwords) is accepted."""
    text = "Gue waktu itu basically belum ngerti PMF dan lagi review pitch deck sprint berikutnya."
    res = language_gate.evaluate_text_sample(text)
    assert res.eligible is True
    assert res.primary_language == "id"
    assert "code-switching" in res.reason or "dominant" in res.reason


def test_language_gate_english_dominant_rejected():
    """English-dominant content is rejected."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Welcome back to the channel everybody today we are going to explore this."},
        {"start": 4.0, "end": 8.0, "text": "The fundamental principle of quantum computing relies on superposition."},
        {"start": 8.0, "end": 12.0, "text": "You can see that there is no other way to achieve this level of performance."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is False
    assert res.primary_language == "en"
    assert "English-dominant" in res.reason


def test_language_gate_foreign_language_rejected():
    """Non-Indonesian non-English foreign speech is rejected."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "Hola a todos bienvenidos a este nuevo episodio de nuestro programa."},
        {"start": 4.0, "end": 8.0, "text": "Hoy vamos a hablar sobre la historia y la cultura de nuestra ciudad."},
        {"start": 8.0, "end": 12.0, "text": "Muchas gracias por acompañarnos en esta maravillosa jornada."}
    ]
    res = language_gate.evaluate_transcript(segments)
    assert res.eligible is False
    assert res.primary_language in ("foreign", "other")


def test_language_gate_empty_or_too_short():
    """Empty or insufficient speech fails safely."""
    res_none = language_gate.evaluate_transcript(None)
    assert res_none.eligible is False
    assert res_none.primary_language == "unknown"

    res_empty = language_gate.evaluate_transcript([])
    assert res_empty.eligible is False

    res_short = language_gate.evaluate_text_sample("halo bro")
    assert res_short.eligible is False
    assert "Insufficient speech content" in res_short.reason

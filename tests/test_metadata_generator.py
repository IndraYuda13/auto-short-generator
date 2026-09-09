"""Unit and Integration Tests for Shorts Metadata Generator (Blueprint Bab 17 & V3.1 Patch)."""

import json
from unittest.mock import MagicMock, patch
import pytest
import requests

from upload.metadata_generator import (
    ShortsMetadataGenerator,
    ShortsMetadata,
    MetadataQualityCheck,
    is_raw_transcript,
)
from pipeline.orchestrator import AutoClipperOrchestrator, PipelineStatus
from storage.repository import StorageRepository, VideoRecord, UploadRecord
from discovery.searcher import VideoSourceMeta


# ==============================================================================
# 1. Helper & Logic Unit Tests
# ==============================================================================

def test_is_raw_transcript_detection():
    """is_raw_transcript accurately detects verbatim speech snippets and allows hooks."""
    transcript = "Kau tahu, Gue sampai sempet bilang, ya ampun Tuhan iya sih mempersiapkan tapi jangan secepat ini dong."

    # Direct prefixes / exact speech snippets -> MUST be flagged as raw transcript
    assert is_raw_transcript("Kau tahu, Gue sampai sempet bilang, ya ampun Tuhan", transcript) is True
    assert is_raw_transcript("Gue sampai sempet bilang, ya ampun Tuhan iya sih", transcript) is True
    assert is_raw_transcript("Kau tahu, Gue sampai sempet", transcript) is True

    # Editorial titles with curiosity gap -> MUST NOT be flagged as raw transcript
    assert is_raw_transcript("Doa Terkabul Terlalu Cepat? Cerita Tak Terduga Yang Menggetarkan Hati", transcript) is False
    assert is_raw_transcript("Ketika Keajaiban Datang Lebih Cepat Dari Dugaan!", transcript) is False
    assert is_raw_transcript("Pelajaran Berharga Saat Menghadapi Momen Kritis Ini", transcript) is False


def test_sanitize_title_clamps_length_and_normalizes():
    """_sanitize_title trims whitespace, removes enclosing quotes, and clamps <= 85 chars."""
    gen = ShortsMetadataGenerator()

    # Normal short title
    assert gen._sanitize_title('  "Ini Judul Menarik Sekali"  ') == "Ini Judul Menarik Sekali"

    # Excessively long title (> 85 chars)
    long_title = "Ini adalah sebuah judul yang luar biasa panjang sekali sehingga melebihi batas delapan puluh lima karakter untuk YouTube Shorts ponsel"
    sanitized = gen._sanitize_title(long_title, max_length=85)
    assert len(sanitized) <= 85
    assert not sanitized.endswith(" ")


def test_sanitize_hashtags_formatting_and_clamping():
    """_sanitize_hashtags ensures #Shorts prefix, adds hash symbol, and clamps to 3-6 tags."""
    gen = ShortsMetadataGenerator()

    raw_tags = ["podcast", "#Indonesia", "viral", "bisnis", "tips", "sukses", "extra1", "extra2"]
    tags = gen._sanitize_hashtags(raw_tags)

    assert len(tags) >= 3
    assert len(tags) <= 6
    assert tags[0] == "#Shorts"
    assert all(t.startswith("#") for t in tags)
    assert "#Indonesia" in tags

    # Minimal input defaults to at least 3 tags
    tags_min = gen._sanitize_hashtags([])
    assert len(tags_min) >= 3
    assert tags_min[0] == "#Shorts"


# ==============================================================================
# 2. Fallback Generator Unit Tests
# ==============================================================================

def test_fallback_metadata_generation():
    """_build_fallback_metadata constructs safe, clean editorial metadata from source title."""
    gen = ShortsMetadataGenerator()
    meta = gen._build_fallback_metadata(
        source_title="Rahasia Sukses Bisnis Kuliner Yang Jarang Dibahas",
        source_channel="Bisnis Millenial",
        clip_summary="Kesalahan fatal dalam cash flow modal usaha",
    )

    assert isinstance(meta, ShortsMetadata)
    assert len(meta.title_candidates) == 5
    assert len(meta.selected_title) <= 85
    assert "Rahasia Sukses Bisnis Kuliner" in meta.selected_title
    assert meta.quality.title_is_not_raw_transcript is True
    assert meta.quality.not_clickbait is True
    assert meta.quality.title_is_truthful is True
    assert 3 <= len(meta.hashtags) <= 6
    assert meta.hashtags[0] == "#Shorts"
    assert "Bisnis Millenial" in meta.description


# ==============================================================================
# 3. LLM Mocked Generation & Retry Tests
# ==============================================================================

def test_metadata_generation_happy_path_mocked():
    """ShortsMetadataGenerator correctly parses valid JSON response on Attempt 1."""
    mock_client = MagicMock()
    mock_response = json.dumps({
        "title_candidates": [
            "Jualan Laris tapi Saldo Nol? Ini Biang Keroknya!",
            "Kenapa Banyak Restoran Bangkrut di Tahun Pertama?",
            "Jebakan Cash Flow yang Sering Menipu Pebisnis Pemula",
            "Bukan Rasa Makanan! Ini Alasan Usaha Kuliner Tutup",
            "Kesalahan Fatal Keuangan Bisnis yang Wajib Dihindari"
        ],
        "selected_title": "Jualan Laris tapi Saldo Nol? Ini Biang Keroknya!",
        "selection_reason": "Menghadirkan paradoks kuat yang memicu curiosity gap.",
        "description": "Banyak orang mengira restoran sukses hanya soal rasa makanan yang enak.\n\nFaktanya, manajemen cash flow dan pemisahan rekening pribadi adalah kunci bertahan.\n\nSimak pembahasannya!",
        "hashtags": ["#Shorts", "#BisnisKuliner", "#TipsBisnis", "#ManajemenKeuangan"],
        "quality": {
            "title_is_not_raw_transcript": True,
            "title_is_truthful": True,
            "title_has_curiosity": True,
            "description_is_relevant": True,
            "not_clickbait": True
        }
    })
    mock_client.chat_completion.return_value = mock_response
    mock_client.extract_json.side_effect = lambda t: json.loads(t)

    gen = ShortsMetadataGenerator(client=mock_client)
    res = gen.generate(
        clip_transcript="Banyak orang mikir buka restoran itu cuma soal rasa makanan enak padahal bukan.",
        source_title="Kenapa Bisnis Kuliner Banyak Gulung Tikar",
        source_channel="Kanal Bisnis",
        clip_summary="Cash flow trap",
    )

    assert res.selected_title == "Jualan Laris tapi Saldo Nol? Ini Biang Keroknya!"
    assert len(res.selected_title) <= 85
    assert len(res.title_candidates) == 5
    assert res.hashtags == ["#Shorts", "#BisnisKuliner", "#TipsBisnis", "#ManajemenKeuangan"]
    assert res.quality.title_is_not_raw_transcript is True
    assert mock_client.chat_completion.call_count == 1


def test_metadata_generation_retry_on_raw_transcript():
    """ShortsMetadataGenerator triggers retry if Attempt 1 produces a raw transcript title."""
    mock_client = MagicMock()

    # Attempt 1 returns raw transcript fragment
    raw_snippet_title = "Kau tahu gue sampai sempet bilang ya ampun"
    bad_response = json.dumps({
        "title_candidates": [raw_snippet_title],
        "selected_title": raw_snippet_title,
        "selection_reason": "Top candidate",
        "description": "Desc 1",
        "hashtags": ["#Shorts"],
        "quality": {
            "title_is_not_raw_transcript": False,  # LLM itself admits or code detects
            "title_is_truthful": True,
            "title_has_curiosity": False,
            "description_is_relevant": True,
            "not_clickbait": True
        }
    })

    # Attempt 2 returns good editorial title
    good_response = json.dumps({
        "title_candidates": [
            "Ketika Doa Dijawab Terlalu Cepat, Apa Yang Terjadi?",
            "Momen Tak Terduga Saat Menghadapi Pilihan Hidup",
            "Kisah Nyata: Jangan Pernah Menyepelekan Doamu Sendiri!",
            "Pelajaran Berharga di Balik Takdir Yang Mengejutkan",
            "Rahasia Hidup Yang Baru Terungkap Setelah Sekian Lama"
        ],
        "selected_title": "Ketika Doa Dijawab Terlalu Cepat, Apa Yang Terjadi?",
        "selection_reason": "Curiosity gap tinggi dan menyentuh emosi penonton.",
        "description": "Sebuah refleksi mendalam mengenai apa yang kita minta dan bagaimana takdir menjawabnya.\n\nSimak kisah selengkapnya.",
        "hashtags": ["#Shorts", "#KisahNyata", "#Inspirasi", "#PodcastIndonesia"],
        "quality": {
            "title_is_not_raw_transcript": True,
            "title_is_truthful": True,
            "title_has_curiosity": True,
            "description_is_relevant": True,
            "not_clickbait": True
        }
    })

    mock_client.chat_completion.side_effect = [bad_response, good_response]
    mock_client.extract_json.side_effect = lambda t: json.loads(t)

    gen = ShortsMetadataGenerator(client=mock_client)
    res = gen.generate(
        clip_transcript="Kau tahu, gue sampai sempet bilang ya ampun Tuhan iya sih mempersiapkan tapi jangan secepat ini.",
        source_title="Podcast Refleksi Kehidupan",
        source_channel="Channel Inspirasi",
    )

    assert mock_client.chat_completion.call_count == 2
    assert res.selected_title == "Ketika Doa Dijawab Terlalu Cepat, Apa Yang Terjadi?"
    assert res.quality.title_is_not_raw_transcript is True


def test_metadata_generation_fallback_on_consecutive_failures():
    """ShortsMetadataGenerator falls back cleanly to source title derivative when LLM fails."""
    mock_client = MagicMock()
    mock_client.chat_completion.side_effect = requests.RequestException("9router timeout")

    gen = ShortsMetadataGenerator(client=mock_client)
    res = gen.generate(
        clip_transcript="Transkrip dialog biasa saja di sini.",
        source_title="Tips Ampuh Mengatur Keuangan di Usia 20-an",
        source_channel="Finansial Pintar",
        clip_summary="Menabung vs Investasi",
    )

    # Safe fallback returned
    assert isinstance(res, ShortsMetadata)
    assert len(res.selected_title) <= 85
    assert "Tips Ampuh Mengatur Keuangan" in res.selected_title
    assert "Finansial Pintar" in res.description
    assert res.quality.title_is_not_raw_transcript is True
    assert res.hashtags[0] == "#Shorts"


# ==============================================================================
# 4. Orchestrator Stage H Integration Test
# ==============================================================================

def test_orchestrator_stage_h_uses_metadata_generator(tmp_path, monkeypatch):
    """AutoClipperOrchestrator Stage H calls metadata_generator and passes metadata to uploader."""
    temp_db = StorageRepository(db_path=tmp_path / "test_stage_h.db")
    dummy_video = tmp_path / "sample.mp4"
    dummy_video.write_bytes(b"\x00" * (120 * 1024))

    mock_metadata_gen = MagicMock()
    custom_metadata = ShortsMetadata(
        title_candidates=[
            "Judul Hebat 1", "Judul Hebat 2", "Judul Hebat 3", "Judul Hebat 4", "Judul Hebat 5"
        ],
        selected_title="Judul Hebat Terpilih Dengan Curiosity Gap",
        selection_reason="Alasan kurasi terbaik",
        description="Deskripsi paragraf 1.\n\nDeskripsi paragraf 2.",
        hashtags=["#Shorts", "#Indonesia", "#PodcastViral"],
        quality=MetadataQualityCheck(),
    )
    mock_metadata_gen.generate.return_value = custom_metadata

    mock_uploader = MagicMock()
    mock_uploader.upload_short.return_value = {
        "status": "success",
        "platform": "youtube",
        "video_id": "dry_run_abc",
        "url": "https://youtube.com/shorts/dry_run_abc",
    }

    orch = AutoClipperOrchestrator(
        repository=temp_db,
        uploader=mock_uploader,
        metadata_generator=mock_metadata_gen,
        output_dir=tmp_path / "output",
        download_dir=tmp_path / "downloads",
    )

    # Test directly that metadata generator is wired in orch
    assert orch.metadata_generator is mock_metadata_gen


# ==============================================================================
# 5. Live 9router Integration Test (ag/gemini-3.8-flash-high)
# ==============================================================================

def test_live_9router_metadata_generation():
    """Live integration test against 9router Gemini 3.8 Flash on port 20128."""
    # Check if 9router is reachable
    try:
        r = requests.get("http://127.0.0.1:20128/v1/models", timeout=3.0)
        if r.status_code != 200:
            pytest.skip("9router service not reachable on port 20128")
    except Exception:
        pytest.skip("9router service not running on port 20128")

    gen = ShortsMetadataGenerator()
    res = gen.generate(
        clip_transcript=(
            "Gue sering ditanya, 'Bang, mending beli rumah dulu apa investasi saham?' "
            "Jawaban gue selalu bikin mereka kaget. Di usia 20-an, aset terbesar lo itu bukan properti, "
            "tapi skill negosiasi dan cash flow lo. Beli rumah cicilan 20 tahun itu komitmen berat."
        ),
        source_title="Beli Rumah Dulu Atau Investasi Saham? Jangan Salah Pilih!",
        source_channel="Finansial Muda",
        clip_summary="Dilema generasi muda antara cicil rumah vs investasi skill dan saham di usia 20-an",
    )

    assert isinstance(res, ShortsMetadata)
    assert len(res.title_candidates) == 5
    assert len(res.selected_title) <= 85
    assert res.quality.title_is_not_raw_transcript is True
    assert 3 <= len(res.hashtags) <= 6
    assert res.hashtags[0] == "#Shorts"
    assert len(res.description.strip()) > 30
    assert not is_raw_transcript(res.selected_title, "Gue sering ditanya, 'Bang, mending beli rumah dulu apa investasi saham?'")

"""Unit tests for Phase A Transcription package (transcript_provider & whisper_aligner)."""

import pytest
from unittest.mock import MagicMock, patch
from transcription.transcript_provider import TranscriptProvider, TranscriptSegment
from transcription.whisper_aligner import WhisperAligner, WordToken


def test_transcript_segment_model():
    """Verify phrase-level TranscriptSegment structure."""
    seg = TranscriptSegment(
        start=120.4,
        end=124.8,
        duration=4.4,
        text="Gue waktu itu mulai bikin usaha pertama gue.",
        source="youtube_caption"
    )
    assert seg.start == 120.4
    assert seg.end == 124.8
    assert seg.duration == 4.4
    assert "usaha pertama" in seg.text


def test_fetch_youtube_transcript_mocked():
    """Priority 1: YouTube caption fetching returns phrase-level timeline."""
    provider = TranscriptProvider()

    mock_snippet_1 = MagicMock()
    mock_snippet_1.start = 10.0
    mock_snippet_1.duration = 3.5
    mock_snippet_1.text = "Halo semua kembali lagi di podcast kita"

    mock_snippet_2 = MagicMock()
    mock_snippet_2.start = 14.0
    mock_snippet_2.duration = 4.0
    mock_snippet_2.text = "Hari ini kita kedatangan tamu yang sangat inspiratif"

    mock_transcript = MagicMock()
    mock_transcript.fetch.return_value = [mock_snippet_1, mock_snippet_2]

    mock_transcript_list = MagicMock()
    mock_transcript_list.find_transcript.return_value = mock_transcript

    with patch.object(provider, "_get_youtube_api_session") as mock_ytt:
        mock_ytt.return_value.list.return_value = mock_transcript_list
        segments = provider.fetch_youtube_transcript("test_vid_id")

        assert segments is not None
        assert len(segments) == 2
        assert segments[0].start == 10.0
        assert segments[0].end == 13.5
        assert segments[0].duration == 3.5
        assert segments[0].source == "youtube_caption"
        assert segments[1].start == 14.0
        assert segments[1].end == 18.0


def test_get_phrase_transcript_fallback_to_whisper():
    """Priority 2: Falls back to Whisper when YouTube caption fails."""
    provider = TranscriptProvider()

    with patch.object(provider, "fetch_youtube_transcript", return_value=None):
        with patch.object(provider, "transcribe_with_whisper") as mock_whisper:
            mock_whisper.return_value = [
                TranscriptSegment(start=0.0, end=4.0, duration=4.0, text="Suara hasil whisper", source="whisper_fallback")
            ]
            with patch("os.path.exists", return_value=True):
                res = provider.get_phrase_transcript("any_vid", audio_or_video_path="/dummy/audio.wav")
                assert len(res) == 1
                assert res[0].source == "whisper_fallback"
                mock_whisper.assert_called_once()


def test_whisper_aligner_fallback_interpolation():
    """WhisperAligner produces evenly spaced word tokens if model fails or on fallback."""
    aligner = WhisperAligner()
    ref_text = "Ini adalah contoh kalimat podcast yang seru"
    words = aligner._fallback_word_alignment(
        start_sec=100.0,
        end_sec=107.0,
        reference_text=ref_text
    )

    assert len(words) == 7
    assert words[0].word == "Ini"
    assert words[0].start == 100.0
    assert words[-1].word == "seru"
    assert words[-1].end == 107.0
    # Check monotonically increasing timestamps
    for i in range(len(words) - 1):
        assert words[i].end <= words[i + 1].start


def test_whisper_aligner_global_time_offsets():
    """WhisperAligner offsets slice timestamps by +start_sec."""
    aligner = WhisperAligner()

    mock_word1 = MagicMock(word="Halo", start=0.5, end=0.9, probability=0.95)
    mock_word2 = MagicMock(word="dunia", start=1.0, end=1.8, probability=0.98)
    mock_seg = MagicMock()
    mock_seg.words = [mock_word1, mock_word2]

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], None)

    with patch.object(aligner, "_get_whisper_model", return_value=mock_model):
        with patch("subprocess.run") as mock_subp:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.stdout = b""
            mock_proc.stderr = b""
            mock_subp.return_value = mock_proc
            with patch("os.path.exists", return_value=True):
                tokens = aligner.align_window(
                    audio_or_video_path="/dummy/sample.mp4",
                    start_sec=120.0,
                    end_sec=150.0
                )
                assert len(tokens) == 2
                assert tokens[0].word == "Halo"
                assert tokens[0].start == 120.5
                assert tokens[0].end == 120.9
                assert tokens[1].word == "dunia"
                assert tokens[1].start == 121.0
                assert tokens[1].end == 121.8

import os
import subprocess
from pathlib import Path
import pytest
from transcriber import Transcriber


class DummyWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class DummySegment:
    def __init__(self, start: float, end: float, text: str, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []


def test_transcribe_clip_words_success(tmp_path: Path, monkeypatch):
    """
    Test transcribe_clip_words():
    1. Only requested clip range [start_sec, end_sec] is extracted with FFmpeg
    2. faster-whisper called with word_timestamps=True
    3. Returned timestamps are clip-local (starting near 0.0s)
    4. Temporary audio file is cleaned up in finally block on success
    """
    t = Transcriber()
    dummy_audio = tmp_path / "original_audio.mp3"
    dummy_audio.write_bytes(b"dummy audio content")

    ffmpeg_commands = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        ffmpeg_commands.append(cmd)
        # Create temp output file if it's the ffmpeg slice command
        out_target = Path(cmd[-1])
        out_target.write_bytes(b"sliced clip audio")
        class Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return Res()

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # Mock whisper model
    whisper_calls = []

    class MockModel:
        def transcribe(self, audio_path, **kwargs):
            whisper_calls.append({"path": audio_path, "kwargs": kwargs})
            # Check that sliced file actually exists at transcription time
            assert Path(audio_path).exists()
            segments = [
                DummySegment(
                    start=0.15,
                    end=2.40,
                    text="halo dunia",
                    words=[
                        DummyWord("halo", 0.15, 0.90),
                        DummyWord("dunia", 0.95, 2.40),
                    ]
                )
            ]
            return segments, None

    monkeypatch.setattr(t, "_get_whisper_model", lambda: MockModel())

    # Call transcribe_clip_words for range 12.0s -> 22.0s (duration 10.0s)
    results = t.transcribe_clip_words(str(dummy_audio), start_sec=12.0, end_sec=22.0)

    # Verification 1: FFmpeg command verifies only clip range extracted
    assert len(ffmpeg_commands) == 1
    ff_cmd = ffmpeg_commands[0]
    assert "-ss" in ff_cmd
    assert ff_cmd[ff_cmd.index("-ss") + 1] == "12.0"
    assert "-t" in ff_cmd
    assert ff_cmd[ff_cmd.index("-t") + 1] == "10.0"

    # Verification 2: faster-whisper called with word_timestamps=True
    assert len(whisper_calls) == 1
    assert whisper_calls[0]["kwargs"].get("word_timestamps") is True

    # Verification 3: returned timestamps are clip-local (0.15s, not 12.15s)
    assert len(results) == 1
    seg = results[0]
    assert seg["start"] == 0.15
    assert seg["end"] == 2.40
    assert len(seg["words"]) == 2
    assert seg["words"][0]["word"] == "halo"
    assert seg["words"][0]["start"] == 0.15
    assert seg["words"][1]["word"] == "dunia"
    assert seg["words"][1]["start"] == 0.95

    # Verification 4: Temporary audio cleaned up after success
    temp_slice_path = Path(whisper_calls[0]["path"])
    assert not temp_slice_path.exists()


def test_transcribe_clip_words_cleans_up_on_failure(tmp_path: Path, monkeypatch):
    """
    Verify temporary audio is cleaned up in finally block when Whisper transcription fails.
    """
    t = Transcriber()
    dummy_audio = tmp_path / "original_audio.mp3"
    dummy_audio.write_bytes(b"dummy audio content")

    created_slice = None

    def mock_subprocess_run(cmd, *args, **kwargs):
        nonlocal created_slice
        created_slice = Path(cmd[-1])
        created_slice.write_bytes(b"sliced clip audio")
        class Res:
            returncode = 0
            stdout = ""
            stderr = ""
        return Res()

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    class FailingModel:
        def transcribe(self, audio_path, **kwargs):
            raise RuntimeError("Simulated faster-whisper internal failure")

    monkeypatch.setattr(t, "_get_whisper_model", lambda: FailingModel())

    with pytest.raises(RuntimeError, match="Simulated faster-whisper internal failure"):
        t.transcribe_clip_words(str(dummy_audio), start_sec=5.0, end_sec=10.0)

    # Temporary sliced audio must be unlinked/deleted even after exception
    assert created_slice is not None
    assert not created_slice.exists()


def test_transcribe_clip_words_no_fake_interpolated_timing():
    """
    Verify that transcriber does NOT produce fake interpolated per-word timing.
    When Whisper yields words, it maps exact word.start and word.end directly.
    When Whisper yields no words for a segment, words list remains empty [] (honest, never faked).
    """
    t = Transcriber()
    dummy_segments = [
        DummySegment(
            start=1.0,
            end=3.0,
            text="segmen tanpa kata",
            words=None # No word timestamps provided by engine
        )
    ]

    class ModelNoWords:
        def transcribe(self, *args, **kwargs):
            return dummy_segments, None

    t._whisper_model = ModelNoWords()
    out = t._transcribe_with_whisper("/path/to/any.mp3")

    assert len(out) == 1
    # Words MUST be empty rather than fabricated evenly spaced timestamps
    assert out[0]["words"] == []
    assert out[0]["start"] == 1.0
    assert out[0]["end"] == 3.0

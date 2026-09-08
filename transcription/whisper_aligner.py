"""Whisper Aligner module for Auto Short Generator Phase A.

Performs precise word-level alignment strictly on selected candidate windows (e.g. 30-55s).
Avoids expensive whole-podcast word alignment by slicing audio for the candidate window only.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel

from config import settings

logger = logging.getLogger(__name__)


class WordToken(BaseModel):
    """Word-level timestamped token."""
    word: str
    start: float
    end: float
    probability: float = 1.0


class WhisperAligner:
    """Performs word-level alignment on a specific candidate time window."""

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        whisper_device: Optional[str] = None,
        whisper_compute_type: Optional[str] = None,
    ):
        self.whisper_model_name = whisper_model or getattr(settings, "WHISPER_MODEL", "base")
        self.whisper_device = whisper_device or getattr(settings, "WHISPER_DEVICE", "cpu")
        self.whisper_compute_type = whisper_compute_type or getattr(settings, "WHISPER_COMPUTE_TYPE", "int8")
        self._whisper_instance: Optional[WhisperModel] = None

    def _get_whisper_model(self) -> WhisperModel:
        if self._whisper_instance is None:
            logger.info(f"Loading faster-whisper for aligner: {self.whisper_model_name}")
            self._whisper_instance = WhisperModel(
                self.whisper_model_name,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        return self._whisper_instance

    def align_window(
        self,
        audio_or_video_path: str,
        start_sec: float,
        end_sec: float,
        reference_text: Optional[str] = None
    ) -> List[WordToken]:
        """
        Extracts candidate audio slice [start_sec, end_sec] and performs word-level alignment.
        Word timestamps are re-offset to global media coordinates [start_sec + offset].
        """
        if not os.path.exists(audio_or_video_path):
            raise FileNotFoundError(f"Media file not found for alignment: {audio_or_video_path}")

        duration = max(0.1, end_sec - start_sec)
        tmp_slice_wav = f"/tmp/align_slice_{Path(audio_or_video_path).stem}_{int(start_sec)}_{int(end_sec)}.wav"

        # Extract precise 16kHz mono audio slice
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", str(audio_or_video_path),
            "-t", f"{duration:.3f}",
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_slice_wav
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            if res.returncode != 0 or not os.path.exists(tmp_slice_wav):
                logger.warning(f"ffmpeg audio slice failed: {res.stderr.decode('utf-8', errors='ignore')}")
                return self._fallback_word_alignment(start_sec, end_sec, reference_text)

            model = self._get_whisper_model()
            segments, info = model.transcribe(
                tmp_slice_wav,
                language="id",
                beam_size=1,
                word_timestamps=True,
                vad_filter=True,
            )

            word_tokens: List[WordToken] = []
            for seg in segments:
                if hasattr(seg, "words") and seg.words:
                    for w in seg.words:
                        clean_word = w.word.strip()
                        if clean_word:
                            word_tokens.append(WordToken(
                                word=clean_word,
                                start=round(start_sec + float(w.start), 3),
                                end=round(start_sec + float(w.end), 3),
                                probability=round(float(getattr(w, "probability", 1.0) or 1.0), 3)
                            ))

            if not word_tokens and reference_text:
                logger.info("Whisper slice yielded no word tokens; falling back to reference text interpolation")
                return self._fallback_word_alignment(start_sec, end_sec, reference_text)

            return word_tokens
        except Exception as e:
            logger.error(f"Whisper alignment error: {e}")
            return self._fallback_word_alignment(start_sec, end_sec, reference_text)
        finally:
            if os.path.exists(tmp_slice_wav):
                try:
                    os.remove(tmp_slice_wav)
                except Exception:
                    pass

    @staticmethod
    def _fallback_word_alignment(
        start_sec: float,
        end_sec: float,
        reference_text: Optional[str]
    ) -> List[WordToken]:
        """Evenly distributes words across the window if model transcription fails."""
        if not reference_text:
            return []

        words = [w.strip() for w in reference_text.split() if w.strip()]
        if not words:
            return []

        duration = max(0.1, end_sec - start_sec)
        word_dur = duration / len(words)

        tokens: List[WordToken] = []
        for i, w in enumerate(words):
            w_start = start_sec + i * word_dur
            w_end = w_start + word_dur
            tokens.append(WordToken(
                word=w,
                start=round(w_start, 3),
                end=round(w_end, 3),
                probability=0.5
            ))
        return tokens

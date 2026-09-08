"""Transcript Provider module for Auto Short Generator Phase A.

Hierarchical transcript acquisition:
Priority 1: Official YouTube captions / transcripts (fast, exact).
Priority 2: faster-whisper fallback (model base/small, Indonesian) if no transcript is available.

Stores phrase-level timeline entries:
[{"start": 120.4, "end": 124.8, "duration": 4.4, "text": "..."}]
Word-level alignment is deferred to whisper_aligner on selected candidates only.
"""

import os
import logging
import subprocess
import http.cookiejar
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel

from config import settings

logger = logging.getLogger(__name__)


class TranscriptSegment(BaseModel):
    """Phrase-level timeline segment."""
    start: float
    end: float
    duration: float
    text: str
    source: str = "youtube_caption"


class TranscriptProvider:
    """Provides phrase-level transcripts prioritizing YouTube captions over Whisper."""

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

    def _get_youtube_api_session(self) -> YouTubeTranscriptApi:
        """Instantiates YouTubeTranscriptApi with local proxy and optional cookies."""
        session = requests.Session()
        proxy = settings.get_random_proxy() if hasattr(settings, "get_random_proxy") else None
        if proxy:
            session.proxies.update({"http": proxy, "https": proxy})

        cookies_path = getattr(settings, "COOKIES_FILE", None)
        if cookies_path and Path(cookies_path).exists():
            try:
                cj = http.cookiejar.MozillaCookieJar(str(cookies_path))
                cj.load(ignore_discard=True, ignore_expires=True)
                for cookie in cj:
                    session.cookies.set_cookie(cookie)
            except Exception as e:
                logger.warning(f"Failed to load cookies for transcript session: {e}")

        return YouTubeTranscriptApi(http_client=session)

    def fetch_youtube_transcript(self, video_id: str) -> Optional[List[TranscriptSegment]]:
        """Attempts to retrieve official YouTube captions/transcript for a video."""
        logger.info(f"Attempting Priority 1: Fetch YouTube caption for video_id={video_id}")
        try:
            ytt = self._get_youtube_api_session()
            transcript_list = ytt.list(video_id)
            transcript = None

            # Try Indonesian first
            try:
                transcript = transcript_list.find_transcript(["id", "in"])
            except Exception:
                # Fallback to any available transcript
                for t in transcript_list:
                    transcript = t
                    break

            if transcript:
                raw_entries = transcript.fetch()
                segments: List[TranscriptSegment] = []
                for entry in raw_entries:
                    start_val = getattr(entry, "start", None)
                    if start_val is None and isinstance(entry, dict):
                        start_val = entry.get("start", 0.0)
                    duration_val = getattr(entry, "duration", None)
                    if duration_val is None and isinstance(entry, dict):
                        duration_val = entry.get("duration", 0.0)
                    text_val = getattr(entry, "text", None)
                    if text_val is None and isinstance(entry, dict):
                        text_val = entry.get("text", "")

                    start = float(start_val or 0.0)
                    duration = float(duration_val or 0.0)
                    text = str(text_val or "").replace("\n", " ").strip()
                    if text:
                        segments.append(TranscriptSegment(
                            start=round(start, 2),
                            end=round(start + duration, 2),
                            duration=round(duration, 2),
                            text=text,
                            source="youtube_caption"
                        ))

                if segments:
                    logger.info(f"Successfully fetched {len(segments)} caption segments from YouTube API")
                    return segments
        except Exception as e:
            logger.warning(f"YouTube transcript fetch failed for {video_id}: {e}")

        return None

    def _get_whisper_model(self) -> WhisperModel:
        """Lazy loads the faster-whisper model."""
        if self._whisper_instance is None:
            logger.info(
                f"Loading faster-whisper model '{self.whisper_model_name}' on "
                f"{self.whisper_device} ({self.whisper_compute_type})..."
            )
            self._whisper_instance = WhisperModel(
                self.whisper_model_name,
                device=self.whisper_device,
                compute_type=self.whisper_compute_type,
            )
        return self._whisper_instance

    def transcribe_with_whisper(
        self,
        audio_or_video_path: str,
        max_duration_sec: Optional[int] = None
    ) -> List[TranscriptSegment]:
        """
        Priority 2: Transcribes audio using faster-whisper at phrase level.
        Does not perform word-level alignment here.
        """
        logger.info(f"Priority 2: Transcribing '{audio_or_video_path}' with faster-whisper")
        if not os.path.exists(audio_or_video_path):
            raise FileNotFoundError(f"Media file not found: {audio_or_video_path}")

        effective_audio_path = audio_or_video_path
        tmp_slice_path = None

        slice_sec = max_duration_sec or getattr(settings, "MAX_AUDIO_TRANSCRIBE_SEC", 600)
        if slice_sec and slice_sec > 0:
            tmp_slice_path = f"/tmp/whisper_slice_{Path(audio_or_video_path).stem}.wav"
            cmd = [
                "ffmpeg", "-y", "-i", str(audio_or_video_path),
                "-t", str(slice_sec),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                tmp_slice_path
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
                effective_audio_path = tmp_slice_path
            except Exception as e:
                logger.warning(f"Failed to slice audio for fast whisper transcription ({e}), using full media")
                effective_audio_path = audio_or_video_path

        try:
            model = self._get_whisper_model()
            whisper_segments, info = model.transcribe(
                effective_audio_path,
                language="id",
                beam_size=1,
                word_timestamps=False,  # Phase-level only in provider
                vad_filter=True,
            )

            segments: List[TranscriptSegment] = []
            for s in whisper_segments:
                text = s.text.strip()
                if text:
                    dur = s.end - s.start
                    segments.append(TranscriptSegment(
                        start=round(s.start, 2),
                        end=round(s.end, 2),
                        duration=round(dur, 2),
                        text=text,
                        source="whisper_fallback"
                    ))

            logger.info(f"Whisper produced {len(segments)} phrase segments")
            return segments
        finally:
            if tmp_slice_path and os.path.exists(tmp_slice_path):
                try:
                    os.remove(tmp_slice_path)
                except Exception:
                    pass

    def get_phrase_transcript(
        self,
        video_id: str,
        audio_or_video_path: Optional[str] = None
    ) -> List[TranscriptSegment]:
        """
        Coordinates Priority 1 (YouTube caption) and Priority 2 (faster-whisper).
        """
        # Priority 1: Official captions
        captions = self.fetch_youtube_transcript(video_id)
        if captions:
            return captions

        # Priority 2: Whisper fallback
        if audio_or_video_path and os.path.exists(audio_or_video_path):
            return self.transcribe_with_whisper(audio_or_video_path)

        raise RuntimeError(
            f"Unable to obtain transcript for video '{video_id}': "
            "YouTube captions unavailable and no local media path provided for Whisper."
        )

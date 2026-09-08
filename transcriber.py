"""Transcriber module to fetch transcript or transcribe locally via faster-whisper."""

import os
import logging
import subprocess
import http.cookiejar
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from faster_whisper import WhisperModel

from config import settings

logger = logging.getLogger(__name__)


class Transcriber:
    def __init__(self):
        self._whisper_model = None

    def _get_youtube_transcript_api(self) -> YouTubeTranscriptApi:
        """Instantiate YouTubeTranscriptApi with local proxy and optional cookies session."""
        cookies_path = settings.COOKIES_FILE
        session = requests.Session()

        proxy = settings.get_random_proxy()
        if proxy:
            session.proxies.update({"http": proxy, "https": proxy})
            logger.info(f"Transcript session using local proxy: {proxy}")

        if cookies_path and cookies_path.exists():
            try:
                cj = http.cookiejar.MozillaCookieJar(str(cookies_path))
                cj.load(ignore_discard=True, ignore_expires=True)
                session.cookies = cj
                logger.info(f"Loaded YouTube cookies into transcript session from {cookies_path}")
            except Exception as e:
                logger.warning(f"Failed to load cookies into transcript session from {cookies_path}: {e}")
        else:
            logger.info(
                f"YouTube cookies file not found at {cookies_path}. "
                "Official transcript fetch will run using local proxy."
            )

        return YouTubeTranscriptApi(http_client=session)

    def get_transcript(self, video_id: str, audio_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Attempts to get transcript from YouTube first (fast, exact).
        If not available, transcribes audio_path using faster-whisper with word timestamps.
        
        Returns format:
        [
            {
                "start": float,
                "duration": float,
                "end": float,
                "text": str,
                "words": [{"word": str, "start": float, "end": float}] # optional
            },
            ...
        ]
        """
        logger.info(f"Attempting to fetch official transcript for video {video_id}...")
        try:
            ytt = self._get_youtube_transcript_api()
            transcript_list = ytt.list(video_id)
            transcript = None
            try:
                # Prefer Indonesian, then auto-translated, then English
                transcript = transcript_list.find_transcript(['id', 'en'])
            except Exception:
                # Get first available generated transcript
                for t in transcript_list:
                    transcript = t
                    break

            if transcript:
                raw_data = transcript.fetch()
                processed = []
                for entry in raw_data:
                    # FetchedTranscriptSnippet object or dict compatibility
                    start = float(getattr(entry, 'start', None) if hasattr(entry, 'start') else entry.get('start', 0.0))
                    duration = float(getattr(entry, 'duration', None) if hasattr(entry, 'duration') else entry.get('duration', 0.0))
                    raw_text = getattr(entry, 'text', '') if hasattr(entry, 'text') else entry.get('text', '')
                    text = str(raw_text).replace('\n', ' ').strip()
                    if text:
                        processed.append({
                            "start": start,
                            "duration": duration,
                            "end": start + duration,
                            "text": text,
                            "words": []
                        })
                logger.info(f"Successfully retrieved {len(processed)} transcript lines from YouTube API")
                return processed
        except Exception as e:
            logger.warning(f"YouTube transcript API not available for {video_id}: {e}")

        # Fallback to local faster-whisper
        if not audio_path or not os.path.exists(audio_path):
            raise RuntimeError(f"No audio file provided and YouTube transcript failed for {video_id}")

        logger.info(f"Transcribing audio file {audio_path} using faster-whisper ({settings.WHISPER_MODEL})...")
        effective_audio_path = audio_path
        slice_audio_path = None
        time_offset = 0.0

        if settings.MAX_AUDIO_TRANSCRIBE_SEC > 0:
            slice_audio_path = str(Path(audio_path).parent / f"slice_{Path(audio_path).name}")
            logger.info(f"Slicing first {settings.MAX_AUDIO_TRANSCRIBE_SEC}s of audio for fast transcription...")
            slice_cmd = [
                "ffmpeg", "-y", "-i", audio_path,
                "-t", str(settings.MAX_AUDIO_TRANSCRIBE_SEC),
                "-acodec", "copy",
                slice_audio_path
            ]
            try:
                subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                effective_audio_path = slice_audio_path
            except Exception as e:
                logger.warning(f"Failed to slice audio: {e}. Transcribing full audio.")
                effective_audio_path = audio_path

        try:
            return self._transcribe_with_whisper(effective_audio_path)
        finally:
            if slice_audio_path and os.path.exists(slice_audio_path):
                try:
                    os.remove(slice_audio_path)
                except Exception:
                    pass

    def transcribe_clip_words(
        self,
        audio_path: str,
        start_sec: float,
        end_sec: float
    ) -> List[Dict[str, Any]]:
        """
        Clip-local word alignment using faster-whisper.
        Slices [start_sec, end_sec] audio into a temporary file, runs Whisper with word_timestamps=True,
        and returns clip-local timestamps (starting near 0.0s).
        Cleans up temporary sliced audio in `finally`.
        """
        duration = end_sec - start_sec
        if duration <= 0:
            return []

        temp_slice = Path(audio_path).parent / f"clip_slice_{start_sec:.1f}_{end_sec:.1f}_{os.getpid()}.mp3"
        logger.info(
            f"Extracting clip-local audio ({start_sec}s - {end_sec}s, duration {duration:.1f}s) "
            f"to {temp_slice}..."
        )

        slice_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(audio_path),
            "-vn", "-acodec", "libmp3lame", "-q:a", "4",
            str(temp_slice)
        ]

        try:
            subprocess.run(slice_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            # Transcribe the isolated clip audio
            raw_segments = self._transcribe_with_whisper(str(temp_slice))
            # raw_segments are already clip-local (0.0 to duration)
            return raw_segments
        except Exception as e:
            logger.warning(f"Clip-local word transcription failed for [{start_sec}, {end_sec}]: {e}")
            raise
        finally:
            if temp_slice.exists():
                try:
                    temp_slice.unlink(missing_ok=True)
                except Exception:
                    pass

    def _get_whisper_model(self) -> WhisperModel:
        if self._whisper_model is None:
            logger.info(f"Loading faster-whisper model '{settings.WHISPER_MODEL}' on {settings.WHISPER_DEVICE}...")
            self._whisper_model = WhisperModel(
                settings.WHISPER_MODEL,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE
            )
        return self._whisper_model

    def _transcribe_with_whisper(self, audio_path: str) -> List[Dict[str, Any]]:
        model = self._get_whisper_model()
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            language="id" # Default Indonesian, can detect auto
        )

        results = []
        for segment in segments:
            words = []
            if segment.words:
                for w in segment.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end
                    })
            results.append({
                "start": segment.start,
                "duration": segment.end - segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": words
            })
        logger.info(f"Whisper transcription completed with {len(results)} segments")
        return results


transcriber = Transcriber()

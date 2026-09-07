"""Main Orchestration and Nonstop Loop Daemon for Auto Short Generator."""

import os
import sys
import time
import signal
import logging
import argparse
from pathlib import Path
from typing import Optional
import yt_dlp

from config import settings
from db import db
from searcher import searcher
from transcriber import transcriber
from analyzer import analyzer
from renderer import renderer
from uploader import uploader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.PROJECT_ROOT / "generator.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("AutoShortDaemon")

RUNNING = True


def signal_handler(signum, frame):
    global RUNNING
    logger.info(f"Received shutdown signal ({signum}). Gracefully stopping daemon...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class AutoShortPipeline:
    def __init__(self):
        self.download_dir = settings.DOWNLOAD_DIR

    def run_one_cycle(self) -> bool:
        """
        Executes one end-to-end cycle:
        1. Find unprocessed video candidates
        2. LLM selects best video
        3. Download audio & video source
        4. Transcribe (YouTube API -> Whisper fallback)
        5. Analyze viral hook with Gemini 3.8 Flash via 9router
        6. Render 9:16 blurred background + dynamic karaoke subtitle short
        7. Upload to YouTube / TikTok
        8. Record in DB
        """
        logger.info("=" * 60)
        logger.info("Starting new discovery & generation cycle...")
        logger.info("=" * 60)

        # Step 1 & 2: Search and Select
        candidates = searcher.search_candidates()
        if not candidates:
            logger.info("No fresh candidate videos found in this cycle. Sleeping.")
            return False

        selected_video = searcher.select_best_video(candidates)
        if not selected_video:
            logger.info("No video selected by LLM. Sleeping.")
            return False

        vid = selected_video["video_id"]
        url = selected_video["url"]
        title = selected_video["title"]

        logger.info(f"Target selected: [{vid}] '{title}' ({url})")

        # Record video as processing in DB immediately
        db.record_video(
            video_id=vid,
            url=url,
            title=title,
            channel_title=selected_video.get("channel_title", ""),
            duration_sec=selected_video.get("duration_sec", 0),
            published_at=selected_video.get("published_at", ""),
            status="processing"
        )

        source_video_path = None
        source_audio_path = None

        try:
            # Step 3: Fetch transcript first (if available from YT API, we don't need Whisper)
            transcript_segments = None
            try:
                transcript_segments = transcriber.get_transcript(vid)
            except Exception as e:
                logger.info(f"Direct transcript failed: {e}. Will download audio for Whisper.")

            # Download Video & Audio
            source_video_path, source_audio_path = self._download_media(url, vid)

            # If no transcript from YT, run Whisper on downloaded audio
            if not transcript_segments:
                transcript_segments = transcriber.get_transcript(vid, audio_path=source_audio_path)

            if not transcript_segments:
                raise RuntimeError(f"Could not obtain transcript for video {vid}")

            # Step 4: Analyze with Gemini 3.8 Flash
            clips = analyzer.analyze_transcript(title, transcript_segments, num_clips=1)
            if not clips:
                raise RuntimeError("LLM did not identify any viable viral clips in transcript")

            # Step 5: Render and Upload each clip
            for idx, clip in enumerate(clips):
                clip_label = f"{vid}_{int(clip['start_sec'])}_{int(clip['end_sec'])}"
                logger.info(f"Processing clip candidate #{idx+1}: {clip['title_clickbait']} ({clip['duration']:.1f}s)")

                # Render video with FFmpeg
                rendered_path = renderer.render_short(
                    source_video_path=source_video_path,
                    start_sec=clip["start_sec"],
                    end_sec=clip["end_sec"],
                    clip_id=clip_label,
                    subtitle_segments=transcript_segments
                )

                # Save clip in DB
                clip_db_id = db.record_clip(
                    video_id=vid,
                    start_sec=clip["start_sec"],
                    end_sec=clip["end_sec"],
                    hook_score=clip["hook_score"],
                    title=clip["title_clickbait"],
                    description=clip["description"],
                    hashtags=clip["hashtags"],
                    rendered_path=rendered_path
                )

                # Step 6: Upload
                upload_res = uploader.upload_clip(
                    video_path=rendered_path,
                    title=clip["title_clickbait"],
                    description=clip["description"],
                    hashtags=clip["hashtags"]
                )

                # Update upload status in DB
                youtube_success = False
                if "youtube" in upload_res:
                    yt_data = upload_res["youtube"]
                    yt_status = yt_data.get("status", "unknown")
                    yt_vid = yt_data.get("video_id")
                    db.update_clip_upload(
                        clip_id=clip_db_id,
                        platform="youtube",
                        status=yt_status,
                        remote_id=yt_vid
                    )
                    if yt_status == "success":
                        youtube_success = True

                if "tiktok" in upload_res:
                    tt_data = upload_res["tiktok"]
                    db.update_clip_upload(
                        clip_id=clip_db_id,
                        platform="tiktok",
                        status=tt_data.get("status", "unknown"),
                        remote_id=tt_data.get("publish_id")
                    )

                # Disk Cleanup Invariant: If successfully uploaded to YouTube Shorts,
                # automatically remove local short_*.mp4 and sub_*.ass to prevent disk overflow
                if youtube_success:
                    self._cleanup_rendered_artifacts(
                        rendered_path=rendered_path,
                        clip_label=clip_label,
                        clip_db_id=clip_db_id
                    )

            # Mark video as completed
            db.update_video_status(vid, status="completed")
            logger.info(f"Successfully finished processing video {vid}!")
            return True

        except Exception as e:
            logger.error(f"Error processing video {vid}: {e}", exc_info=True)
            db.update_video_status(vid, status="failed", error_message=str(e))
            return False

        finally:
            # Clean up source heavy video download to conserve disk space
            self._cleanup_temp_files(source_video_path, source_audio_path)

    def _download_media(self, url: str, video_id: str) -> tuple[str, str]:
        """Downloads 720p/best video and separate mp3 audio."""
        logger.info(f"Downloading source video and audio from {url}...")
        video_out_tmpl = str(self.download_dir / f"{video_id}.%(ext)s")

        proxy = "http://127.0.0.1:31001"
        ydl_opts = {
            "format": "18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/b/best",
            "outtmpl": video_out_tmpl,
            "quiet": False,
            "no_warnings": True,
            "overwrites": True,
            "proxy": proxy,
            "js_runtimes": {"deno": {"path": "/usr/local/bin/deno"}},
            "remote_components": ["ejs:github"],
        }
        if proxy:
            logger.info(f"yt-dlp download using local proxy: {proxy}")

        # Check and apply cookies if available
        cookies_path = settings.COOKIES_FILE
        if cookies_path and cookies_path.exists():
            logger.info(f"Using YouTube cookies file for video download: {cookies_path}")
            ydl_opts["cookiefile"] = str(cookies_path)
        else:
            logger.info(
                f"YouTube cookies file not found at {cookies_path}. "
                "Download will attempt using local proxy."
            )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Find downloaded video file
        video_files = list(self.download_dir.glob(f"{video_id}.*"))
        if not video_files:
            raise FileNotFoundError(f"Failed to find downloaded video file for {video_id}")

        video_path = str(video_files[0])
        audio_path = str(self.download_dir / f"{video_id}.mp3")

        # Extract audio mp3 with ffmpeg for fast whisper processing
        if not os.path.exists(audio_path):
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "libmp3lame", "-q:a", "4",
                audio_path
            ]
            import subprocess
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        return video_path, audio_path

    def _cleanup_temp_files(self, video_path: Optional[str], audio_path: Optional[str]):
        """Clean up raw downloaded long videos to avoid filling up the disk."""
        for p in [video_path, audio_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    logger.debug(f"Removed temporary source file: {p}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {p}: {e}")

    def _cleanup_rendered_artifacts(
        self,
        rendered_path: str,
        clip_label: str,
        clip_db_id: int
    ) -> None:
        """
        Post-Upload Memory/Disk Hygiene Invariant:
        When a clip is successfully uploaded to YouTube Shorts (status == 'success'),
        automatically remove short_*.mp4 and sub_*.ass from output directory,
        then update rendered_path in database to '[UPLOADED_AND_CLEANED]'.
        Pending/failed clips are preserved for review/audit.
        """
        logger.info(f"Executing post-upload cleanup for clip ID {clip_db_id} ({clip_label})...")
        files_to_remove = []

        if rendered_path and os.path.exists(rendered_path):
            files_to_remove.append(Path(rendered_path))

        # Check corresponding .ass subtitle file in output directory
        ass_candidates = [
            settings.OUTPUT_DIR / f"sub_{clip_label}.ass",
            Path(rendered_path).with_suffix(".ass") if rendered_path else None
        ]
        for ass_file in ass_candidates:
            if ass_file and ass_file.exists() and ass_file not in files_to_remove:
                files_to_remove.append(ass_file)

        # Defense-in-depth: Ensure target files are strictly within OUTPUT_DIR boundary
        resolved_output = settings.OUTPUT_DIR.resolve()
        deleted_count = 0
        for f in files_to_remove:
            try:
                resolved_f = f.resolve()
                if not resolved_f.is_relative_to(resolved_output):
                    logger.error(f"[SECURITY] Refusing to delete out-of-bounds file: {resolved_f}")
                    continue
                resolved_f.unlink(missing_ok=True)
                logger.info(f"Cleaned up local post-upload artifact: {resolved_f}")
                deleted_count += 1
            except Exception as err:
                logger.warning(f"Could not remove local file {f}: {err}")

        # Update DB rendered_path marker
        db.update_clip_rendered_path(clip_db_id, "[UPLOADED_AND_CLEANED]")
        logger.info(f"Updated DB clip #{clip_db_id} rendered_path -> [UPLOADED_AND_CLEANED] ({deleted_count} files removed)")


def run_daemon():
    logger.info("Starting Auto Short Generator Daemon...")
    pipeline = AutoShortPipeline()

    while RUNNING:
        try:
            processed = pipeline.run_one_cycle()
            stats = db.get_stats()
            logger.info(f"Current DB Stats: {stats}")
        except Exception as e:
            logger.error(f"Unexpected error in daemon loop: {e}", exc_info=True)

        logger.info(f"Cycle completed. Sleeping for {settings.LOOP_INTERVAL_SECONDS} seconds before next check...")
        # Responsive sleep loop checking RUNNING flag
        slept = 0
        while RUNNING and slept < settings.LOOP_INTERVAL_SECONDS:
            time.sleep(2)
            slept += 2

    logger.info("Auto Short Generator Daemon stopped cleanly.")


def auth_youtube():
    """Interactive OAuth flow for YouTube."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    from uploader import YOUTUBE_SCOPES

    client_secrets = settings.YOUTUBE_CLIENT_SECRETS_FILE
    token_file = settings.YOUTUBE_CREDENTIALS_FILE

    if not client_secrets.exists():
        print(f"Error: {client_secrets} not found! Download it from Google Cloud Console first.")
        sys.exit(1)

    print("Starting interactive YouTube OAuth flow...")
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), YOUTUBE_SCOPES)
    creds = flow.run_local_server(port=8080)
    with open(token_file, "w") as f:
        f.write(creds.to_json())
    print(f"Successfully saved credentials to {token_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Short Generator Nonstop Daemon")
    parser.add_argument("--once", action="store_true", help="Run only one cycle and exit")
    parser.add_argument("--auth-youtube", action="store_true", help="Run interactive YouTube OAuth flow")
    parser.add_argument("--stats", action="store_true", help="Display current database statistics")
    args = parser.parse_args()

    if args.auth_youtube:
        auth_youtube()
    elif args.stats:
        print("Database Statistics:", db.get_stats())
    elif args.once:
        pipeline = AutoShortPipeline()
        pipeline.run_one_cycle()
        print("Single cycle completed.")
    else:
        run_daemon()

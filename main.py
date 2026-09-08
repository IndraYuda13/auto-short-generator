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
from edit_plan import EditPlan, FramingMode
from edit_director import edit_director
from visual_framing import visual_framing
from pacing import pacing_engine
from qc import qc_evaluator
from language_gate import language_gate

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

            # Step 3b: Indonesian-Only Language Gate (Product Invariant)
            if settings.INDONESIAN_ONLY_ENABLED:
                lang_eval = language_gate.evaluate_transcript(transcript_segments)
                logger.info(
                    f"Language Gate evaluation for video {vid}: eligible={lang_eval.eligible}, "
                    f"lang={lang_eval.primary_language}, confidence={lang_eval.confidence:.2f}, "
                    f"reason='{lang_eval.reason}', id_ratio={lang_eval.id_ratio:.2f}, en_ratio={lang_eval.en_ratio:.2f}"
                )

                if not lang_eval.eligible:
                    rejection_reason = (
                        f"REJECTED_NON_INDONESIAN: primary_language='{lang_eval.primary_language}', "
                        f"confidence={lang_eval.confidence:.2f}, reason='{lang_eval.reason}'"
                    )
                    logger.warning(
                        f"[LANGUAGE GATE REJECTED] Video {vid} rejected by Indonesian-only invariant. "
                        f"Reason: {rejection_reason}. Skipping renderer, skipping uploader, not completing."
                    )
                    # Mark video as rejected in DB with clear audit trail; never mark completed
                    db.update_video_status(vid, status="rejected", error_message=rejection_reason)
                    return False

            # Step 4: Analyze with Gemini 3.8 Flash
            clips = analyzer.analyze_transcript(title, transcript_segments, num_clips=1)
            if not clips:
                raise RuntimeError("LLM did not identify any viable viral clips in transcript")

            # Step 5: Render and Upload each clip
            any_clip_succeeded = False
            for idx, clip in enumerate(clips):
                clip_label = f"{vid}_{int(clip['start_sec'])}_{int(clip['end_sec'])}"
                logger.info(f"Processing clip candidate #{idx+1}: {clip['title_clickbait']} ({clip['duration']:.1f}s)")

                start_sec = clip["start_sec"]
                end_sec = clip["end_sec"]
                clip_duration = end_sec - start_sec

                # Step 5a: Clip-local Word Alignment via Whisper
                clip_subtitles = transcript_segments
                try:
                    logger.info(f"Attempting clip-local word alignment for clip {clip_label}...")
                    whisper_words = transcriber.transcribe_clip_words(
                        audio_path=source_audio_path,
                        start_sec=start_sec,
                        end_sec=end_sec
                    )
                    if whisper_words:
                        clip_subtitles = whisper_words
                        logger.info(f"Clip-local word alignment succeeded with {len(whisper_words)} segments")
                except Exception as e:
                    logger.warning(
                        f"Clip-local word alignment failed: {e}. "
                        "Falling back to phrase-level segments."
                    )

                # Step 5b: EditPlan generation via EditDirector
                edit_plan = None
                try:
                    edit_plan = edit_director.create_plan_for_clip(
                        clip_id=clip_label,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        transcript_segments=clip_subtitles,
                        video_title=title,
                        hook_reason=clip.get("hook_reason", "")
                    )
                except Exception as e:
                    logger.warning(f"EditDirector failed: {e}. Using default PODCAST_CLEAN plan.")
                    edit_plan = EditPlan.create_default(
                        clip_id=clip_label,
                        duration=clip_duration,
                        framing_mode=FramingMode.BLURRED_FALLBACK
                    )

                # Step 5c: Visual Framing Analysis (Face-tracked vs Blurred fallback)
                if settings.FACE_TRACKING_ENABLED:
                    try:
                        framing_mode, keyframes = visual_framing.analyze_clip_framing(
                            video_path=source_video_path,
                            start_sec=start_sec,
                            end_sec=end_sec
                        )
                        edit_plan.framing_mode = framing_mode
                        edit_plan.crop_keyframes = keyframes
                        logger.info(f"Visual framing resolved: {framing_mode.value} ({len(keyframes)} keyframes)")
                    except Exception as e:
                        logger.warning(f"Visual framing analysis failed: {e}. Falling back to BLURRED_FALLBACK.")
                        edit_plan.framing_mode = FramingMode.BLURRED_FALLBACK
                        edit_plan.crop_keyframes = []

                # Step 5d: Render video with Renderer V2
                rendered_path = renderer.render_short(
                    source_video_path=source_video_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    clip_id=clip_label,
                    subtitle_segments=clip_subtitles,
                    edit_plan=edit_plan
                )

                # Step 5e: Quality Control (QC) Hard Gate
                qc_report = qc_evaluator.evaluate_video(
                    video_path=rendered_path,
                    expected_duration=clip_duration
                )
                logger.info(
                    f"QC evaluation for {clip_label}: passed={qc_report.passed}, "
                    f"checks={qc_report.checks}"
                )

                if not qc_report.passed:
                    logger.error(f"[QC GATE FAILED] Refusing to upload clip {clip_label}. Errors: {qc_report.errors}")
                    db.record_clip(
                        video_id=vid,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        hook_score=clip["hook_score"],
                        title=clip["title_clickbait"],
                        description=clip["description"],
                        hashtags=clip["hashtags"],
                        rendered_path=f"[QC_FAILED: {', '.join(qc_report.errors)}]"
                    )
                    continue

                any_clip_succeeded = True

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

                # Step 6: Upload (only executed after passing QC)
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

            if not any_clip_succeeded:
                raise RuntimeError("All rendered clips failed Quality Control (QC) or rendering.")

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

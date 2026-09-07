"""Uploader module supporting YouTube Data API v3 and official TikTok Content Posting API v2."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import requests

from config import settings

logger = logging.getLogger(__name__)

# Scopes for YouTube video upload
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


class Uploader:
    def upload_clip(
        self,
        video_path: str,
        title: str,
        description: str,
        hashtags: str,
        platforms: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Uploads clip to target platforms specified in config or arguments.
        Returns dict with status and IDs per platform.
        """
        if platforms is None:
            platforms = [p.strip().lower() for p in settings.TARGET_PLATFORMS.split(",") if p.strip()]

        results = {}
        full_title = f"{title[:80]} #Shorts"
        full_desc = f"{description}\n\n{hashtags}"

        for platform in platforms:
            if platform == "youtube":
                tags_list = hashtags if isinstance(hashtags, list) else [t.strip() for t in str(hashtags).split() if t.strip()]
                results["youtube"] = self.upload_to_youtube(video_path, full_title, full_desc, tags=tags_list)
            elif platform == "tiktok":
                results["tiktok"] = self.upload_to_tiktok(video_path, full_title, full_desc)
            elif platform in ["local", "none", "local_only"]:
                results["local"] = {"status": "saved", "path": video_path}

        return results

    # ==========================
    # YOUTUBE UPLOADER (v3 API)
    # ==========================
    def upload_to_youtube(
        self,
        video_path: str,
        title: str,
        description: str,
        tags: Optional[list] = None
    ) -> Dict[str, Any]:
        """Uploads video to YouTube via resumable OAuth upload."""
        logger.info(f"Initiating YouTube Shorts upload for '{title}'...")

        token_file = settings.YOUTUBE_CREDENTIALS_FILE
        client_secrets_file = settings.YOUTUBE_CLIENT_SECRETS_FILE

        if not client_secrets_file.exists() and not token_file.exists():
            msg = (
                f"YouTube OAuth credentials not found ({client_secrets_file}). "
                "Simulating local upload or waiting for client_secrets.json to be placed."
            )
            logger.warning(msg)
            return {"status": "skipped", "reason": "no_credentials", "message": msg}

        creds = None
        if token_file.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_file), YOUTUBE_SCOPES)
            except Exception as e:
                logger.error(f"Failed to load existing YouTube token: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    with open(token_file, "w") as f:
                        f.write(creds.to_json())
                except Exception as e:
                    logger.error(f"Error refreshing YouTube token: {e}")
                    creds = None

            if not creds:
                if not client_secrets_file.exists():
                    return {"status": "failed", "reason": f"Missing {client_secrets_file}"}
                # If running headless in daemon mode, cannot open browser
                msg = f"OAuth token missing or expired. Run 'python main.py --auth-youtube' interactively once."
                logger.error(msg)
                return {"status": "auth_required", "message": msg}

        try:
            youtube = build("youtube", "v3", credentials=creds)
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags or ["shorts", "viral"],
                    "categoryId": "22",  # People & Blogs
                },
                "status": {
                    "privacyStatus": "private",
                    "selfDeclaredMadeForKids": False,
                }
            }

            media = MediaFileUpload(
                video_path,
                chunksize=1024 * 1024 * 4,
                resumable=True,
                mimetype="video/mp4"
            )

            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media
            )

            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    logger.info(f"YouTube upload progress: {int(status.progress() * 100)}%")

            video_id = response.get("id")
            logger.info(f"YouTube Shorts upload SUCCESS! URL: https://youtube.com/shorts/{video_id}")
            return {"status": "success", "video_id": video_id, "url": f"https://youtube.com/shorts/{video_id}"}

        except Exception as e:
            logger.error(f"YouTube upload error: {e}")
            return {"status": "failed", "error": str(e)}

    # ==========================
    # TIKTOK CONTENT POSTING API v2
    # ==========================
    def upload_to_tiktok(
        self,
        video_path: str,
        title: str,
        description: str
    ) -> Dict[str, Any]:
        """
        Official TikTok Content Posting API v2:
        Endpoint: POST https://open.tiktokapis.com/v2/post/publish/video/init/
        Reference: https://developers.tiktok.com/doc/content-posting-api-reference-upload-video
        """
        logger.info(f"Initiating TikTok upload for '{title}'...")

        access_token = settings.TIKTOK_ACCESS_TOKEN
        if not access_token:
            msg = (
                "TikTok Access Token is empty in .env (TIKTOK_ACCESS_TOKEN). "
                "Official TikTok Content Posting API requires a registered TikTok Developer App with video.upload scope. "
                "Saved video locally in output directory for manual upload or until token is configured."
            )
            logger.warning(msg)
            return {"status": "skipped", "reason": "no_token", "message": msg}

        file_size = os.path.getsize(video_path)

        # 1. Initialize Video Post
        init_url = "https://open.tiktokapis.com/v2/post/publish/video/init/"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8"
        }

        payload = {
            "post_info": {
                "title": f"{title}\n{description}"[:150], # TikTok title limit
                "privacy_level": "PUBLIC_TO_EVERYONE",
                "disable_duet": False,
                "disable_stitch": False,
                "disable_comment": False,
                "video_cover_timestamp_ms": 1000
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": file_size,
                "chunk_size": file_size,
                "total_chunk_count": 1
            }
        }

        try:
            resp = requests.post(init_url, headers=headers, json=payload, timeout=30)
            data = resp.json()
            if data.get("error", {}).get("code") != "ok":
                err_msg = data.get("error", {}).get("message", "Unknown TikTok API error")
                logger.error(f"TikTok init error: {err_msg}")
                return {"status": "failed", "error": err_msg}

            publish_id = data["data"]["publish_id"]
            upload_url = data["data"]["upload_url"]

            # 2. Upload video binary via PUT
            with open(video_path, "rb") as f:
                video_data = f.read()

            upload_headers = {
                "Content-Type": "video/mp4",
                "Content-Range": f"bytes 0-{file_size - 1}/{file_size}"
            }

            upload_resp = requests.put(upload_url, headers=upload_headers, data=video_data, timeout=120)
            if upload_resp.status_code in [200, 201]:
                logger.info(f"TikTok upload SUCCESS! Publish ID: {publish_id}")
                return {"status": "success", "publish_id": publish_id}
            else:
                logger.error(f"TikTok binary upload failed: {upload_resp.status_code} - {upload_resp.text}")
                return {"status": "failed", "status_code": upload_resp.status_code, "error": upload_resp.text}

        except Exception as e:
            logger.error(f"TikTok API exception: {e}")
            return {"status": "failed", "error": str(e)}


uploader = Uploader()

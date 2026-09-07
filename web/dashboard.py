"""FastAPI Web Dashboard for Auto Short Generator (Port 8450).
Implements UI/UX Pro Max OLED minimal aesthetic, real-time SSE activity feed, and SQLite integration.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Dict, Any, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure auto-short-generator root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from db import db

logger = logging.getLogger("AutoShortWeb")

app = FastAPI(
    title="Auto Short Studio Dashboard",
    description="Autonomous Video Ingestion & Vertical Clip Generation Telemetry",
    version="2.1.0"
)

# Mount static files & Jinja templates
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"
STATIC_DIR = PROJECT_ROOT / "web" / "static"
OUTPUT_DIR = settings.OUTPUT_DIR

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


def get_daemon_status() -> Dict[str, Any]:
    """Inspect systemd or process list to determine live daemon status."""
    import subprocess
    is_active = False
    details = "Idle"

    # 1. Check systemctl status autoshort
    try:
        res = subprocess.run(
            ["systemctl", "is-active", "autoshort"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2
        )
        if res.stdout.strip() == "active":
            is_active = True
    except Exception:
        pass

    # 2. Check screen or python main.py process if systemd not active yet
    if not is_active:
        try:
            ps_res = subprocess.run(
                ["pgrep", "-f", "python3 main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2
            )
            if ps_res.stdout.strip():
                is_active = True
        except Exception:
            pass

    # 3. Inspect recent video processing state from DB
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT video_id, title FROM videos WHERE status = 'processing' ORDER BY processed_at DESC LIMIT 1")
        row = cursor.fetchone()
        if row:
            details = f"Processing: {row['title'][:35]}..." if row['title'] else f"Processing {row['video_id']}"
        elif is_active:
            details = "Active / Scanning"
        else:
            details = "Offline / Stopped"

    return {
        "active": is_active,
        "state": "active" if is_active else "offline",
        "details": details
    }


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def index_page(request: Request):
    """Render main dashboard page."""
    stats = db.get_stats()
    daemon = get_daemon_status()
    clips = db.get_recent_clips(limit=50)

    # Process clips presentation
    processed_clips = []
    for c in clips:
        # Check if local file exists or if cleaned
        c_dict = dict(c)
        r_path = c_dict.get("rendered_path") or ""
        c_dict["is_cleaned"] = r_path == "[UPLOADED_AND_CLEANED]" or not os.path.exists(r_path)
        c_dict["local_available"] = bool(r_path and os.path.exists(r_path))
        
        # Local video web URL if available
        if c_dict["local_available"]:
            c_dict["web_video_url"] = f"/output/{Path(r_path).name}"
        else:
            c_dict["web_video_url"] = None

        # Format youtube short URL
        yt_id = c_dict.get("youtube_video_id")
        if yt_id:
            c_dict["youtube_url"] = f"https://youtube.com/shorts/{yt_id}"
        else:
            c_dict["youtube_url"] = None

        # YouTube thumbnail URL
        c_dict["thumbnail_url"] = f"https://img.youtube.com/vi/{c_dict['video_id']}/hqdefault.jpg"

        processed_clips.append(c_dict)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "stats": stats,
            "daemon": daemon,
            "clips": processed_clips,
        }
    )


@app.get("/api/stats")
async def api_stats():
    """Returns real-time database KPI stats and daemon status."""
    stats = db.get_stats()
    daemon = get_daemon_status()
    return {
        "stats": stats,
        "daemon": daemon
    }


@app.get("/api/clips")
async def api_clips(limit: int = 50):
    """Returns JSON list of clips for dynamic polling update."""
    clips = db.get_recent_clips(limit=limit)
    processed = []
    for c in clips:
        c_dict = dict(c)
        r_path = c_dict.get("rendered_path") or ""
        c_dict["is_cleaned"] = r_path == "[UPLOADED_AND_CLEANED]" or not os.path.exists(r_path)
        c_dict["local_available"] = bool(r_path and os.path.exists(r_path))
        yt_id = c_dict.get("youtube_video_id")
        c_dict["youtube_url"] = f"https://youtube.com/shorts/{yt_id}" if yt_id else None
        c_dict["thumbnail_url"] = f"https://img.youtube.com/vi/{c_dict['video_id']}/hqdefault.jpg"
        processed.append(c_dict)
    return {"clips": processed}


@app.get("/api/stream/sse")
async def sse_logs(request: Request):
    """
    Server-Sent Events (SSE) streaming daemon generator.log in real-time.
    Supports continuous tailing with fast initial buffer.
    """
    async def log_generator() -> AsyncGenerator[str, None]:
        log_file = settings.PROJECT_ROOT / "generator.log"
        last_pos = 0

        # Memory & Event Loop Hardening:
        # Read only up to last 16KB from disk instead of reading full multi-MB file
        if log_file.exists():
            try:
                file_size = log_file.stat().st_size
                read_start = max(0, file_size - 16384)
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(read_start)
                    content = f.read()
                    last_pos = f.tell()
                    tail_lines = content.splitlines()[-40:]
                    for line in tail_lines:
                        line_clean = line.strip()
                        if line_clean:
                            yield f"data: {json.dumps({'line': line_clean})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'line': f'[ERROR] Log read error: {e}'})}\n\n"

        # Continuous streaming loop with file rotation & truncation safety
        while True:
            if await request.is_disconnected():
                break

            if log_file.exists():
                try:
                    curr_size = log_file.stat().st_size
                    # Reset pointer if log file was rotated or truncated
                    if curr_size < last_pos:
                        last_pos = 0

                    if curr_size > last_pos:
                        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            new_data = f.read()
                            last_pos = f.tell()

                            if new_data:
                                for raw_line in new_data.splitlines():
                                    line_clean = raw_line.strip()
                                    if line_clean:
                                        yield f"data: {json.dumps({'line': line_clean})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'line': f'[ERROR] Log stream error: {e}'})}\n\n"

            await asyncio.sleep(1.0)

    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8450)

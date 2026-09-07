"""Searcher module for finding fresh potential YouTube videos to clip."""

import json
import logging
import random
import re
from typing import List, Dict, Any, Optional
import yt_dlp
from googleapiclient.discovery import build

from config import settings
from db import db
from llm_client import llm_client

logger = logging.getLogger(__name__)


class Searcher:
    def __init__(self):
        self.api_key = settings.YOUTUBE_API_KEY
        self.fallback_queries = [
            q.strip() for q in settings.SEARCH_QUERIES.split(",") if q.strip()
        ]
        self._cached_queries: List[str] = []

    def _get_cookie_opts(self) -> Dict[str, Any]:
        """Returns yt-dlp options for proxy, JS runtime (deno), and optional cookies."""
        proxy = "http://127.0.0.1:31001"
        opts: Dict[str, Any] = {
            "proxy": proxy,
            "js_runtimes": {"deno": {"path": "/usr/local/bin/deno"}},
            "remote_components": ["ejs:github"],
        }
        if proxy:
            logger.info(f"yt-dlp using local proxy: {proxy}")

        cookies_path = settings.COOKIES_FILE
        if cookies_path and cookies_path.exists():
            logger.info(f"Using YouTube cookies file for yt-dlp search: {cookies_path}")
            opts["cookiefile"] = str(cookies_path)
        else:
            logger.info(
                f"No cookies file found at {cookies_path}. "
                "yt-dlp search will run using proxy without cookies."
            )
        return opts

    def generate_ai_queries(self, count: int = 8) -> List[str]:
        """
        Ask Gemini 3.8 Flash via 9router to dynamically generate diverse, fresh,
        and general search queries for viral highlights in Indonesia.
        """
        prompt = f"""Kamu adalah AI Content Discovery & Trend Analyst profesional untuk YouTube Shorts dan TikTok di Indonesia.
Tugasmu adalah menghasilkan daftar {count} search query YouTube yang SANGAT SEGAR, BERVARIASI, dan GENERAL untuk mencari video panjang (podcast, livestreaming highlights, talkshow viral, interview mendalam, debat, stand-up comedy, gaming podcast, obrolan bisnis/kreatif, misteri/investigasi, dll) yang kaya akan potongan klip/highlight viral berpotensi tinggi.

Pastikan kategorinya berputar dan mencakup variasi luas:
1. Podcast obrolan santai, komedi, opini, dan cerita hidup unik (misal: podcast unscripted, obrolan komika, curhat kisah nyata)
2. Livestreaming highlights / momen seru streamer populer Indonesia
3. Wawancara / interview figur publik, pengusaha, kreator inspiratif
4. Debat, diskusi panas sosial/politik/hukum terkini
5. Stand-up comedy special / podcast komika
6. Kasus unik, investigasi kriminal, misteri, storytelling mendalam
7. Gaming podcast & pop culture creator obrolan seru

Keluarkan jawaban HANYA berupa valid JSON array of strings tanpa markdown penjelasan lain, contoh:
[
  "query satu",
  "query dua"
]"""
        try:
            res = llm_client.chat_completion(prompt, temperature=0.8, json_mode=True)
            text = res.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        data = v
                        break
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                queries = [q.strip() for q in data if q.strip()]
                if queries:
                    logger.info(f"Generated {len(queries)} dynamic AI search queries: {queries}")
                    return queries
        except Exception as e:
            logger.warning(f"Failed to generate dynamic queries from LLM: {e}. Falling back to config list.")

        return self.fallback_queries or ["podcast viral indonesia"]

    def search_candidates(self) -> List[Dict[str, Any]]:
        """
        Search videos using dynamic AI-generated queries.
        Uses YouTube Data API v3 if key available, else yt-dlp search.
        """
        if not self._cached_queries:
            self._cached_queries = self.generate_ai_queries(count=8)

        query = self._cached_queries.pop(0) if self._cached_queries else "podcast viral indonesia"
        logger.info(f"Searching candidate videos with dynamic query: '{query}'")

        candidates = []
        if self.api_key:
            try:
                candidates = self._search_via_api(query)
                logger.info(f"Retrieved {len(candidates)} videos from YouTube Data API v3")
            except Exception as e:
                logger.warning(f"YouTube Data API failed: {e}. Falling back to yt-dlp search.")
                candidates = self._search_via_ytdlp(query)
        else:
            candidates = self._search_via_ytdlp(query)

        # Filter out already processed videos
        unprocessed = []
        for v in candidates:
            if not db.is_video_processed(v["video_id"]):
                unprocessed.append(v)
            else:
                logger.debug(f"Skipping already processed video: {v['video_id']}")

        logger.info(f"Found {len(unprocessed)} unprocessed candidates out of {len(candidates)}")
        return unprocessed

    def _search_via_api(self, query: str) -> List[Dict[str, Any]]:
        youtube = build("youtube", "v3", developerKey=self.api_key)
        request = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=settings.MAX_SEARCH_RESULTS,
            type="video",
            videoDuration="long",  # Long podcasts/talkshows
            order="date",
        )
        response = request.execute()
        results = []
        for item in response.get("items", []):
            vid = item["id"]["videoId"]
            snippet = item["snippet"]
            results.append(
                {
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": snippet.get("title", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "description": snippet.get("description", ""),
                }
            )
        return results

    def _search_via_ytdlp(self, query: str) -> List[Dict[str, Any]]:
        """Fallback search using yt-dlp with optional cookiefile support."""
        ydl_opts = {
            "format": "18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/b/best",
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
        }
        ydl_opts.update(self._get_cookie_opts())

        search_query = f"ytsearch{settings.MAX_SEARCH_RESULTS}:{query}"
        results = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)
                for entry in info.get("entries", []):
                    vid = entry.get("id")
                    if not vid:
                        continue
                    results.append(
                        {
                            "video_id": vid,
                            "url": entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                            "title": entry.get("title", ""),
                            "channel_title": entry.get("uploader", "")
                            or entry.get("channel", ""),
                            "duration_sec": entry.get("duration", 0),
                            "published_at": entry.get("upload_date", ""),
                            "description": entry.get("description", ""),
                        }
                    )
        except Exception as e:
            logger.error(f"yt-dlp search error: {e}")
        return results

    def select_best_video(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Ask Gemini 3.8 Flash via 9router to pick the most promising video."""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        summary_list = []
        for i, c in enumerate(candidates):
            summary_list.append(
                f"[{i}] Video ID: {c['video_id']}\nTitle: {c['title']}\nChannel: {c.get('channel_title')}\nDescription: {c.get('description', '')[:200]}\n"
            )
        candidates_text = "\n".join(summary_list)

        prompt = f"""Kamu adalah kurator konten video viral profesional untuk YouTube Shorts dan TikTok.
Tugasmu adalah memilih 1 video dari daftar calon video berikut yang paling berpotensi memiliki klip perbincangan viral, insight mengejutkan, atau momen emosional/lucu yang disukai audiens Indonesia.

Daftar Calon:
{candidates_text}

Kembalikan jawaban HANYA indeks integer (misalnya 0, atau 1, atau 2) dari video terbaik tanpa penjelasan tambahan."""

        try:
            choice_str = llm_client.chat_completion(prompt, temperature=0.1)
            match = re.search(r"\d+", choice_str)
            if match:
                idx = int(match.group(0))
                if 0 <= idx < len(candidates):
                    logger.info(f"LLM selected video index {idx}: {candidates[idx]['title']}")
                    return candidates[idx]
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}. Falling back to first candidate.")

        return candidates[0]


searcher = Searcher()

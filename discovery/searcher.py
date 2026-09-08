"""Discovery Searcher module for Auto Short Generator Phase A (Blueprint Stage 1).

Responsible for discovering potential long-form source videos WITHOUT downloading full video files.
- Extracts metadata only via yt-dlp search (`ytsearch{limit}:{query}`) or YouTube Data API v3.
- Emits structured Pydantic models: `VideoSourceMeta` (and `VideoMetadata` alias).
- Connects directly with `SourceFilter` to guarantee only eligible candidates are yielded.
- Supports dynamic AI query generation via 9router LLM.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import yt_dlp
from googleapiclient.discovery import build
from pydantic import BaseModel, Field, model_validator

from config import settings
from llm_client import llm_client

logger = logging.getLogger(__name__)


class VideoSourceMeta(BaseModel):
    """Structured metadata model representing a discovered YouTube video source."""
    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    channel_title: str = ""
    duration_sec: float = 0.0
    duration: float = 0.0
    published_at: str = ""
    view_count: int = 0
    description: str = ""
    is_live: bool = False
    is_upcoming: bool = False
    is_private: bool = False
    has_captions: bool = False
    subtitles_available: List[str] = Field(default_factory=list)
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def sync_meta_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            dur = data.get("duration")
            dur_sec = data.get("duration_sec")
            if dur is not None and dur_sec is None:
                data["duration_sec"] = float(dur)
            elif dur_sec is not None and dur is None:
                data["duration"] = float(dur_sec)
            elif dur is not None and dur_sec is not None:
                if float(dur_sec) != 0.0 and float(dur) == 0.0:
                    data["duration"] = float(dur_sec)
                elif float(dur) != 0.0 and float(dur_sec) == 0.0:
                    data["duration_sec"] = float(dur)

            ch = data.get("channel")
            ch_title = data.get("channel_title")
            if ch and not ch_title:
                data["channel_title"] = ch
            elif ch_title and not ch:
                data["channel"] = ch_title
        return data


# Backward-compatible alias for existing test suites and consumers
VideoMetadata = VideoSourceMeta


class Searcher:
    """Discovers source video metadata without downloading full video media."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cookies_file: Optional[Path] = None,
        proxy: Optional[str] = None,
        source_filter: Optional[Any] = None,
    ):
        self.api_key = api_key or getattr(settings, "YOUTUBE_API_KEY", "")
        self.cookies_file = cookies_file or getattr(settings, "COOKIES_FILE", None)
        self.proxy = proxy or (settings.get_random_proxy() if hasattr(settings, "get_random_proxy") else None)
        queries_str = getattr(settings, "SEARCH_QUERIES", "")
        self.fallback_queries = [
            q.strip() for q in queries_str.split(",") if q.strip()
        ] if queries_str else ["podcast viral indonesia"]
        self._cached_queries: List[str] = []

        if source_filter is None:
            from discovery.source_filter import SourceFilter
            self.source_filter = SourceFilter()
        else:
            self.source_filter = source_filter

    def _get_ytdlp_opts(self, flat: bool = False) -> Dict[str, Any]:
        """Returns non-downloading yt-dlp options for metadata extraction only."""
        opts: Dict[str, Any] = {
            "skip_download": True,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist" if flat else False,
            "remote_components": ["ejs:github"],
        }
        if self.proxy:
            opts["proxy"] = self.proxy

        if hasattr(settings, "DENO_PATH") and Path(settings.DENO_PATH).exists():
            opts["js_runtimes"] = {"deno": {"path": settings.DENO_PATH}}

        if self.cookies_file and Path(self.cookies_file).exists():
            opts["cookiefile"] = str(self.cookies_file)

        return opts

    def generate_ai_queries(self, count: int = 6) -> List[str]:
        """Dynamically generate Indonesian viral discovery queries via 9router."""
        prompt = f"""Kamu adalah AI Content Discovery & Trend Analyst profesional untuk YouTube Shorts di Indonesia.
Hasilkan {count} search query YouTube yang segar dan bervariasi untuk mencari video panjang (podcast, talkshow, interview mendalam, komedi, debat) yang kaya akan segmen klip viral berbobot.
Keluarkan HANYA valid JSON array of strings, contoh:
[
  "podcast obrolan inspiratif indonesia",
  "interview viral komika indonesia"
]"""
        try:
            res = llm_client.chat_completion(prompt, temperature=0.7, json_mode=True)
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
                    return queries
        except Exception as e:
            logger.warning(f"Failed to generate dynamic AI search queries: {e}")

        return self.fallback_queries or ["podcast viral indonesia"]

    def search_candidates(self, limit: int = 5, query: Optional[str] = None) -> List[VideoSourceMeta]:
        """
        Discovers and filters video candidates. Connects directly to SourceFilter
        so that only eligible videos are returned.
        """
        if not query:
            if not self._cached_queries:
                self._cached_queries = self.generate_ai_queries(count=6)
            query = self._cached_queries.pop(0) if self._cached_queries else "podcast viral indonesia"

        logger.info(f"Searching candidate videos for query: '{query}' (target eligible limit={limit})")
        # Fetch a slightly larger batch to account for filtered non-eligible videos
        raw_results = self.search_videos(query=query, max_results=max(limit * 2, 8), filter_eligible=False)

        eligible_videos: List[VideoSourceMeta] = []
        for vid_meta in raw_results:
            verdict = self.source_filter.filter_video(vid_meta)
            if verdict.is_eligible:
                eligible_videos.append(vid_meta)
                if hasattr(self.source_filter, "processed_video_ids"):
                    self.source_filter.processed_video_ids.add(vid_meta.video_id)
                if len(eligible_videos) >= limit:
                    break
            else:
                logger.info(
                    f"Candidate [{vid_meta.video_id}] '{vid_meta.title[:40]}' rejected by SourceFilter: {verdict.reason} ({verdict.rejection_code})"
                )

        logger.info(f"Discovered {len(raw_results)} candidates -> {len(eligible_videos)} eligible videos passed Phase A filter.")
        return eligible_videos

    def search_videos(
        self,
        query: str,
        max_results: int = 5,
        filter_eligible: bool = True,
    ) -> List[VideoSourceMeta]:
        """
        Search YouTube videos (via Data API if key present, else yt-dlp search) without downloading media.
        When filter_eligible=True, connects to SourceFilter and filters out non-eligible videos.
        """
        logger.info(f"Executing search for query: '{query}' (max_results={max_results}, filter_eligible={filter_eligible})")
        raw_candidates: List[VideoSourceMeta] = []

        if self.api_key:
            try:
                raw_candidates = self._search_via_api(query, max_results=max(max_results * 2, 8) if filter_eligible else max_results)
            except Exception as e:
                logger.warning(f"YouTube Data API search failed ({e}); falling back to yt-dlp search.")
                raw_candidates = self._search_via_ytdlp(query, max_results=max(max_results * 2, 8) if filter_eligible else max_results)
        else:
            raw_candidates = self._search_via_ytdlp(query, max_results=max(max_results * 2, 8) if filter_eligible else max_results)

        if not filter_eligible:
            return raw_candidates[:max_results]

        eligible_candidates: List[VideoSourceMeta] = []
        for v in raw_candidates:
            verdict = self.source_filter.filter_video(v)
            if verdict.is_eligible:
                eligible_candidates.append(v)
                if len(eligible_candidates) >= max_results:
                    break
            else:
                logger.debug(f"Filtered out video {v.video_id}: {verdict.reason}")

        return eligible_candidates

    def search_eligible_videos(self, query: str, max_results: int = 5) -> List[VideoSourceMeta]:
        """Convenience method explicitly searching for Phase A eligible videos."""
        return self.search_videos(query=query, max_results=max_results, filter_eligible=True)

    def _search_via_api(self, query: str, max_results: int = 5) -> List[VideoSourceMeta]:
        """Searches YouTube using official YouTube Data API v3 without downloading media."""
        youtube = build("youtube", "v3", developerKey=self.api_key)
        req = youtube.search().list(
            q=query,
            part="snippet",
            type="video",
            maxResults=max_results,
            order="relevance",
            relevanceLanguage="id",
        )
        res = req.execute()
        items = res.get("items", [])
        if not items:
            return []

        video_ids = [it["id"]["videoId"] for it in items if "videoId" in it.get("id", {})]
        if not video_ids:
            return []

        det_req = youtube.videos().list(
            part="contentDetails,statistics,snippet",
            id=",".join(video_ids)
        )
        det_res = det_req.execute()
        detailed_items = {it["id"]: it for it in det_res.get("items", [])}

        results: List[VideoSourceMeta] = []
        for vid in video_ids:
            det = detailed_items.get(vid)
            if not det:
                continue

            snippet = det.get("snippet", {})
            content = det.get("contentDetails", {})
            stats = det.get("statistics", {})

            duration_iso = content.get("duration", "PT0S")
            duration_sec = self._parse_iso_duration(duration_iso)
            live_broadcast = snippet.get("liveBroadcastContent")
            is_live = live_broadcast == "live"
            is_upcoming = live_broadcast == "upcoming"
            has_captions = content.get("caption") == "true"

            results.append(VideoSourceMeta(
                video_id=vid,
                url=f"https://www.youtube.com/watch?v={vid}",
                title=snippet.get("title", ""),
                channel=snippet.get("channelTitle", ""),
                channel_title=snippet.get("channelTitle", ""),
                duration_sec=duration_sec,
                duration=duration_sec,
                view_count=int(stats.get("viewCount", 0)),
                description=snippet.get("description", ""),
                is_live=is_live,
                is_upcoming=is_upcoming,
                is_private=False,
                published_at=snippet.get("publishedAt", ""),
                has_captions=has_captions,
                raw_metadata=det
            ))

        return results

    def _search_via_ytdlp(self, query: str, max_results: int = 5) -> List[VideoSourceMeta]:
        """Searches YouTube using yt-dlp extract_info without downloading media."""
        search_query = f"ytsearch{max_results}:{query}"
        opts = self._get_ytdlp_opts(flat=False)

        results: List[VideoSourceMeta] = []
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(search_query, download=False)
                if not info:
                    return []

                entries = info.get("entries", [])
                for entry in entries:
                    if not entry:
                        continue
                    vid = entry.get("id", "")
                    if not vid:
                        continue

                    duration = float(entry.get("duration") or 0.0)
                    live_status = entry.get("live_status") or ""
                    is_live = bool(entry.get("is_live", False) or live_status == "is_live")
                    is_upcoming = bool(live_status == "is_upcoming")
                    subtitles = list((entry.get("subtitles") or {}).keys()) + list((entry.get("automatic_captions") or {}).keys())
                    channel_name = entry.get("channel") or entry.get("uploader") or ""

                    results.append(VideoSourceMeta(
                        video_id=vid,
                        url=entry.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}",
                        title=entry.get("title", ""),
                        channel=channel_name,
                        channel_title=channel_name,
                        duration_sec=duration,
                        duration=duration,
                        view_count=int(entry.get("view_count") or 0),
                        description=entry.get("description", "") or "",
                        is_live=is_live,
                        is_upcoming=is_upcoming,
                        is_private=bool(entry.get("availability") == "private"),
                        published_at=str(entry.get("upload_date") or ""),
                        has_captions=len(subtitles) > 0,
                        subtitles_available=subtitles,
                        raw_metadata=entry
                    ))
        except Exception as e:
            logger.error(f"yt-dlp search error: {e}")

        return results

    def get_video_metadata(self, video_id_or_url: str) -> Optional[VideoSourceMeta]:
        """Fetch metadata for a single specific video without downloading media."""
        opts = self._get_ytdlp_opts(flat=False)
        url = video_id_or_url if video_id_or_url.startswith("http") else f"https://www.youtube.com/watch?v={video_id_or_url}"
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                entry = ydl.extract_info(url, download=False)
                if not entry:
                    return None

                vid = entry.get("id", "")
                duration = float(entry.get("duration") or 0.0)
                live_status = entry.get("live_status") or ""
                is_live = bool(entry.get("is_live", False) or live_status == "is_live")
                is_upcoming = bool(live_status == "is_upcoming")
                subtitles = list((entry.get("subtitles") or {}).keys()) + list((entry.get("automatic_captions") or {}).keys())
                channel_name = entry.get("channel") or entry.get("uploader") or ""

                return VideoSourceMeta(
                    video_id=vid,
                    url=entry.get("webpage_url") or url,
                    title=entry.get("title", ""),
                    channel=channel_name,
                    channel_title=channel_name,
                    duration_sec=duration,
                    duration=duration,
                    view_count=int(entry.get("view_count") or 0),
                    description=entry.get("description", "") or "",
                    is_live=is_live,
                    is_upcoming=is_upcoming,
                    is_private=bool(entry.get("availability") == "private"),
                    published_at=str(entry.get("upload_date") or ""),
                    has_captions=len(subtitles) > 0,
                    subtitles_available=subtitles,
                    raw_metadata=entry
                )
        except Exception as e:
            logger.error(f"Failed to fetch video metadata for {video_id_or_url}: {e}")
            return None

    def select_best_video(self, candidates: List[VideoSourceMeta]) -> Optional[VideoSourceMeta]:
        """Ask Gemini 3.8 Flash via 9router to pick the most promising candidate."""
        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0]

        summary_list = []
        for i, c in enumerate(candidates):
            summary_list.append(
                f"[{i}] Video ID: {c.video_id}\nTitle: {c.title}\nChannel: {c.channel}\nDuration: {c.duration_sec}s\nDescription: {c.description[:200]}\n"
            )
        candidates_text = "\n".join(summary_list)

        prompt = f"""Kamu adalah kurator konten video viral profesional untuk YouTube Shorts dan TikTok di Indonesia.
Pilihlah 1 video dari daftar calon video berikut yang paling berpotensi memiliki obrolan viral, insight mengejutkan, atau momen emosional/lucu yang disukai audiens Indonesia.

Daftar Calon:
{candidates_text}

Kembalikan jawaban HANYA indeks integer (misalnya 0, atau 1, atau 2) dari video terbaik tanpa penjelasan tambahan."""

        try:
            choice_str = llm_client.chat_completion(prompt, temperature=0.1)
            match = re.search(r"\d+", choice_str)
            if match:
                idx = int(match.group(0))
                if 0 <= idx < len(candidates):
                    logger.info(f"LLM selected video index {idx}: {candidates[idx].title}")
                    return candidates[idx]
        except Exception as e:
            logger.warning(f"LLM selection failed ({e}). Falling back to first candidate.")

        return candidates[0]

    @staticmethod
    def _parse_iso_duration(duration_str: str) -> float:
        """Parses ISO 8601 duration (e.g. PT1H2M30S) into seconds."""
        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
        if not match:
            return 0.0
        h = int(match.group(1) or 0)
        m = int(match.group(2) or 0)
        s = int(match.group(3) or 0)
        return float(h * 3600 + m * 60 + s)

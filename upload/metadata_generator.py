"""Surgical YouTube Shorts Metadata Generator (Bab 17 & Blueprint V3.1).

Generates high-performing, ethical clickbait titles, engaging descriptions,
and targeted hashtags using Gemini 3.8 Flash via 9router.
Strictly prevents raw transcript copying and generic templates.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from config import settings
from llm_client import llm_client, LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Kamu adalah Senior Video Editor dan Growth Strategist spesialis konten short-form "
    "(YouTube Shorts & TikTok Reels) dengan pengalaman memproduksi miliaran views. "
    "Keahlian utamamu adalah meracik metadata (judul clickbait beretika, deskripsi engaging, "
    "dan hashtags relevan) yang memancing rasa penasaran tinggi tanpa menipu "
    "(curiosity gap tanpa clickbait palsu)."
)


def _normalize_text(text: str) -> str:
    """Strips punctuation and normalizes whitespace/casing for similarity matching."""
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def is_raw_transcript(title: str, transcript: str) -> bool:
    """Detects whether title is a verbatim or near-verbatim copy of the transcript.

    Returns True if title is directly excerpted from transcript, False if editorial.
    """
    if not title or not transcript:
        return False

    norm_title = _normalize_text(title)
    norm_trans = _normalize_text(transcript)

    if not norm_title or not norm_trans:
        return False

    title_words = norm_title.split()
    trans_words = norm_trans.split()

    # Direct substring check for phrase >= 3 words
    if len(title_words) >= 3:
        if norm_title in norm_trans:
            return True

    # Character-level prefix match (title starts with transcript start)
    prefix_title = norm_title.replace(" ", "")[:25]
    prefix_trans = norm_trans.replace(" ", "")[:25]
    if len(prefix_title) >= 15 and prefix_trans.startswith(prefix_title):
        return True

    # Word-level prefix match (first 4 consecutive words match transcript start)
    if len(title_words) >= 4 and len(trans_words) >= 4:
        if title_words[:4] == trans_words[:4]:
            return True

    return False


class ShortsMetadataQuality(BaseModel):
    """Quality self-check results for generated metadata."""
    title_is_not_raw_transcript: bool = True
    title_is_truthful: bool = True
    title_has_curiosity: bool = True
    description_is_relevant: bool = True
    not_clickbait: bool = True


# Alias for backward-compatibility with tests
MetadataQualityCheck = ShortsMetadataQuality


class ShortsMetadata(BaseModel):
    """Structured result containing generated metadata for YouTube Shorts."""
    title_candidates: List[str] = Field(default_factory=list, description="5 title candidates ranked/evaluated")
    selected_title: str = Field(..., description="The chosen winning title (<= 85 chars)")
    selection_reason: str = Field(..., description="Justification for selecting this title")
    description: str = Field(..., description="Engaging 1-3 paragraph description with context")
    hashtags: List[str] = Field(default_factory=list, description="3-6 targeted hashtags")
    quality: ShortsMetadataQuality = Field(default_factory=ShortsMetadataQuality)
    is_fallback: bool = Field(default=False, description="True if generated via deterministic fallback")


class ShortsMetadataGenerator:
    """Generates optimized metadata (title, description, hashtags) for YouTube Shorts."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        max_retries: int = 1,
    ):
        self.client = client or llm_client
        self.model = model or getattr(settings, "LLM_MODEL", "ag/gemini-3.8-flash-high")
        self.max_retries = max_retries

    def is_raw_transcript_copy(self, title: str, transcript: str) -> bool:
        """Instance method alias for is_raw_transcript."""
        return is_raw_transcript(title, transcript)

    def _sanitize_title(self, raw_title: str, max_length: int = 85) -> str:
        """Cleans, normalizes, and truncates title to meet YouTube Shorts constraints."""
        title = raw_title.strip().strip('"\'')
        title = re.sub(r"\s+", " ", title)
        if len(title) > max_length:
            truncated = title[: max_length - 3]
            last_space = truncated.rfind(" ")
            if last_space > 30:
                title = truncated[:last_space].strip() + "..."
            else:
                title = truncated.strip() + "..."
        return title

    def _sanitize_hashtags(self, hashtags: List[str]) -> List[str]:
        """Ensures valid hashtag list starting with #Shorts and clamped to 3-6 items."""
        clean: List[str] = []
        for tag in hashtags:
            if not isinstance(tag, str):
                continue
            t = tag.strip()
            if not t:
                continue
            if not t.startswith("#"):
                t = f"#{t}"
            t = re.sub(r"[^\w#]", "", t)
            if t and t.lower() not in [c.lower() for c in clean]:
                clean.append(t)

        # Ensure #Shorts is the very first hashtag
        shorts_idx = -1
        for i, t in enumerate(clean):
            if t.lower() == "#shorts":
                shorts_idx = i
                break

        if shorts_idx >= 0:
            shorts_tag = clean.pop(shorts_idx)
            clean.insert(0, shorts_tag)
        else:
            clean.insert(0, "#Shorts")

        # Fill minimum 3 tags
        default_fillers = ["#PodcastIndonesia", "#Indonesia", "#ViralIndonesia"]
        for filler in default_fillers:
            if len(clean) >= 3:
                break
            if filler.lower() not in [c.lower() for c in clean]:
                clean.append(filler)

        return clean[:6]

    def _format_description(self, raw_description: str, hashtags: List[str]) -> str:
        """Formats description ensuring clean paragraphs and appended hashtags."""
        desc = (raw_description or "").strip()
        tags_str = " ".join(hashtags)
        if tags_str and tags_str not in desc:
            desc = f"{desc}\n\n{tags_str}" if desc else tags_str
        return desc

    def _build_fallback_metadata(
        self,
        source_title: str,
        source_channel: str = "",
        clip_summary: str = "",
        clip_transcript: str = "",
    ) -> ShortsMetadata:
        """Deterministic fallback when LLM fails or generates invalid output."""
        logger.warning("Generating deterministic fallback metadata for YouTube Shorts.")

        clean_src = re.sub(r"\[.*?\]|\(.*?\)", "", source_title).strip()
        clean_src = clean_src.split("|")[0].strip()
        if not clean_src or len(clean_src) < 4:
            clean_src = "Obrolan Inspiratif"

        candidates = [
            self._sanitize_title(f"Pelajaran Penting: {clean_src}", max_length=85),
            self._sanitize_title(f"Rahasia di Balik: {clean_src}", max_length=85),
            self._sanitize_title(f"Kisah Nyata: {clean_src}", max_length=85),
            self._sanitize_title(f"Insight Menarik: {clean_src}", max_length=85),
            self._sanitize_title(f"Highlight Spesial: {clean_src}", max_length=85),
        ]
        selected_title = candidates[0]

        tags = self._sanitize_hashtags(["#Shorts", "#PodcastIndonesia", "#Inspirasi", "#ViralIndonesia"])

        channel_name = source_channel.strip() or "Channel Sumber"
        desc_lines = [
            f"Cuplikan obrolan menarik dan sarat makna dari {source_title}.",
            f"Simak sudut pandang dan perbincangan selengkapnya di video original {channel_name}.",
            " ".join(tags),
        ]
        description = "\n\n".join(desc_lines)

        return ShortsMetadata(
            title_candidates=candidates,
            selected_title=selected_title,
            selection_reason="Deterministic fallback derived safely from source video metadata.",
            description=description,
            hashtags=tags,
            quality=ShortsMetadataQuality(
                title_is_not_raw_transcript=True,
                title_is_truthful=True,
                title_has_curiosity=True,
                description_is_relevant=True,
                not_clickbait=True,
            ),
            is_fallback=True,
        )

    # Alias for internal use
    _build_fallback = _build_fallback_metadata

    def _build_prompt(
        self,
        clip_transcript: str,
        source_title: str,
        source_channel: str,
        clip_summary: str,
        warning_note: Optional[str] = None,
    ) -> str:
        """Constructs the structured prompt for Gemini."""
        warning_block = ""
        if warning_note:
            warning_block = f"PERHATIAN KHUSUS DARI EVALUASI SEBELUMNYA:\n{warning_note}\n\n"

        prompt = f"""Judul Video Sumber: "{source_title}"
Channel / Pembicara: "{source_channel}"
Ringkasan / Hook Momen Klip: "{clip_summary}"

Transkrip Dialog Klip Terpilih:
---
{clip_transcript.strip()}
---

Tugasmu:
Buatlah paket metadata lengkap untuk YouTube Shorts berdasarkan klip dialog di atas.

Instruksi Khusus & Batasan Ketat:
1. JUDUL (Title Candidates & Selected Title):
   - Buat 5 kandidat judul unik dalam Bahasa Indonesia natural.
   - Panjang masing-masing judul MAKSIMAL 85 karakter.
   - Judul harus memiliki CURIOSITY GAP tinggi (memancing rasa ingin tahu yang kuat), relevan dengan topik inti klip, dan jujur (tidak menipu / bukan hoax).
   - DILARANG KERAS menyalin ucapan transkrip mentah kata-demi-kata (contoh SALAH: "Kau tahu, Gue sampai sempet bilang..."). Judul harus berupa kemasan sudut pandang, insight inti, atau pertanyaan tajam.
   - Jangan ALL CAPS berlebihan. Jangan spam emoji (maksimal 0-1 emoji relevan).
   - Evaluasi kelima judul berdasarkan clarity, curiosity, relevance, naturalness, dan shorts appeal, lalu pilih 1 judul terbaik sebagai `selected_title` beserta alasannya di `selection_reason`.

2. DESKRIPSI (Description):
   - Tulis 1–3 paragraf pendek yang menjelaskan konteks dan insight utama klip ini secara santai dan menarik.
   - Berikan sedikit latar belakang agar penonton memahami bobot percakapan.
   - JANGAN gunakan template kaku/generik (misal: "Auto Short from...").
   - Di akhir deskripsi, sertakan hashtags yang telah dipilih.

3. HASHTAGS:
   - Pilih 3–6 hashtags relevan.
   - Wajib ada #Shorts pada urutan pertama.
   - Sertakan hashtag niche (contoh: #PodcastIndonesia jika format podcast/obrolan) serta topik atau nama tokoh yang dibahas.

4. QUALITY SELF-CHECK:
   - Evaluasi outputmu sendiri secara jujur pada field `quality`.

{warning_block}Kembalikan jawaban HANYA dalam format JSON valid (tanpa teks pengantar atau markdown tambahan di luar blok JSON):
{{
  "title_candidates": [
    "Kandidat 1",
    "Kandidat 2",
    "Kandidat 3",
    "Kandidat 4",
    "Kandidat 5"
  ],
  "selected_title": "Judul Terbaik Pilihan (<= 85 karakter)",
  "selection_reason": "Alasan objektif mengapa judul ini paling memikat dan relevan",
  "description": "Deskripsi engaging 1-3 paragraf...\\n\\n#Shorts #PodcastIndonesia #Topik",
  "hashtags": [
    "#Shorts",
    "#PodcastIndonesia",
    "#TopikSpesifik"
  ],
  "quality": {{
    "title_is_not_raw_transcript": true,
    "title_is_truthful": true,
    "title_has_curiosity": true,
    "description_is_relevant": true,
    "not_clickbait": true
  }}
}}"""
        return prompt

    def generate(
        self,
        clip_transcript: str,
        source_title: str,
        source_channel: str,
        clip_summary: str = "",
    ) -> ShortsMetadata:
        """Generates YouTube Shorts metadata with retry on raw transcript detection."""
        transcript = (clip_transcript or "").strip()
        s_title = (source_title or "").strip()
        s_channel = (source_channel or "").strip()
        c_summary = (clip_summary or "").strip()

        warning_note: Optional[str] = None

        for attempt in range(self.max_retries + 1):
            prompt = self._build_prompt(
                clip_transcript=transcript,
                source_title=s_title,
                source_channel=s_channel,
                clip_summary=c_summary,
                warning_note=warning_note,
            )

            try:
                raw_response = self.client.chat_completion(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.4,
                    json_mode=True,
                )
                parsed = self.client.extract_json(raw_response)
                if not isinstance(parsed, dict):
                    logger.warning(f"[Attempt {attempt+1}] Response did not parse as JSON dict.")
                    warning_note = "Format output sebelumnya tidak valid JSON. Harap kembalikan HANYA JSON valid."
                    continue

                candidates = parsed.get("title_candidates") or []
                selected_title = parsed.get("selected_title") or (candidates[0] if candidates else "")
                selection_reason = parsed.get("selection_reason") or "Dipilih berdasarkan skor rasa penasaran dan relevansi tertinggi."
                description = parsed.get("description") or ""
                hashtags = parsed.get("hashtags") or []
                quality_dict = parsed.get("quality") or {}

                clean_title = self._sanitize_title(selected_title)
                clean_tags = self._sanitize_hashtags(hashtags)
                clean_desc = self._format_description(description, clean_tags)

                # Quality self-check evaluation
                is_raw = self.is_raw_transcript_copy(clean_title, transcript)
                declared_not_raw = quality_dict.get("title_is_not_raw_transcript", True)

                if is_raw or not declared_not_raw:
                    logger.warning(
                        f"[Attempt {attempt+1}] Title '{clean_title}' flagged as raw transcript copy "
                        f"(detected={is_raw}, declared_not_raw={declared_not_raw}). Retrying..."
                    )
                    warning_note = (
                        f"Judul sebelumnya ('{clean_title}') ditolak karena terlalu mirip salinan transkrip ucapan mentah. "
                        "BUATLAH JUDUL BARU yang merupakan KEMASAN INTISARI / PERTANYAAN / SUDUT PANDANG MENARIK, "
                        "BUKAN mengulang apa yang diucapkan pembicara!"
                    )
                    continue

                quality = ShortsMetadataQuality(
                    title_is_not_raw_transcript=True,
                    title_is_truthful=bool(quality_dict.get("title_is_truthful", True)),
                    title_has_curiosity=bool(quality_dict.get("title_has_curiosity", True)),
                    description_is_relevant=bool(quality_dict.get("description_is_relevant", True)),
                    not_clickbait=bool(quality_dict.get("not_clickbait", True)),
                )

                logger.info(f"Successfully generated Shorts metadata: '{clean_title}'")
                return ShortsMetadata(
                    title_candidates=[self._sanitize_title(c) for c in candidates] or [clean_title],
                    selected_title=clean_title,
                    selection_reason=str(selection_reason),
                    description=clean_desc,
                    hashtags=clean_tags,
                    quality=quality,
                    is_fallback=False,
                )

            except Exception as e:
                logger.warning(f"[Attempt {attempt+1}] Failed to call LLM for metadata: {e}")
                warning_note = f"Terjadi kesalahan teknis: {e}"

        logger.error("ShortsMetadataGenerator exhausted retries. Returning fallback metadata.")
        return self._build_fallback_metadata(
            source_title=s_title,
            source_channel=s_channel,
            clip_transcript=transcript,
            clip_summary=c_summary,
        )

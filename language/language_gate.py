"""Language Eligibility Gate for Indonesian-Only Product Invariant.

Deterministic validation of spoken language from speech/transcript segments or audio samples.
Enforces:
- PRIMARY_LANGUAGE must be Indonesian (id).
- Slang & colloquial Indonesian (gue, lu, bang, nih, kok, wkwk, mantap, dll) accepted.
- Natural Indonesian-English code-switching accepted (e.g. 'Gue waktu itu basically belum ngerti PMF').
- English-dominant or other foreign languages rejected.
- Unknown / empty speech fail-safe skip/rejection.
- Spoken speech analysis overrides contradictory title/metadata.
"""

import os
import re
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Core Indonesian lexicon (common formal, function words, pronouns, verbs, adjectives)
INDONESIAN_CORE_WORDS = {
    "yang", "dan", "di", "ini", "itu", "dari", "ke", "ada", "saya", "kita",
    "kami", "kamu", "dia", "mereka", "dengan", "untuk", "pada", "adalah", "sebagai",
    "dalam", "bisa", "akan", "sudah", "tidak", "juga", "karena", "lebih", "hanya",
    "bukan", "seperti", "kalau", "banyak", "orang", "saat", "oleh", "harus",
    "tahun", "hari", "kemudian", "sangat", "lagi", "secara", "setelah", "dapat",
    "antara", "menjadi", "lain", "semua", "tentang", "jadi", "hampir", "selalu",
    "mungkin", "ketika", "sampai", "membuat", "kata", "tahu", "punya", "bilang",
    "pikir", "tapi", "namun", "terus", "baru", "sedang", "masih", "belum",
    "mengapa", "bagaimana", "dimana", "kapan", "siapa", "apa", "kenapa", "saja",
    "sudah", "pernah", "begitu", "begini", "sini", "situ", "sana", "mau",
    "ingin", "sama", "buat", "biar", "supaya", "agar", "balik",
    "jalan", "makan", "minum", "tidur", "bicara", "dengar", "lihat",
    "kerja", "hidup", "masuk", "keluar", "bawa", "tanya", "jawab", "baca", "tulis",
    "waktu", "tempat", "cerita", "hal", "soal", "masalah", "dunia", "anak", "teman"
}

# Indonesian colloquial / slang lexicon
INDONESIAN_SLANG_WORDS = {
    "gue", "gua", "gw", "lu", "lo", "elu", "luorang",
    "bang", "mas", "mbak", "bro", "sis", "gan", "kak", "boskuu", "cuy",
    "nggak", "ngga", "ga", "gak", "nggaklah", "enggak", "ogah",
    "banget", "bgt", "parah", "gokil", "keren", "mantap", "mantul", "anjir", "anjing",
    "bener", "beneran", "emang", "emangnya", "kan", "dong", "sih", "nih", "tuh", "deh",
    "kok", "loh", "lho", "lah", "yah", "yaudah", "gimana", "ngapain", "kek", "kayak",
    "kayaknya", "gitu", "gini", "gini-gini", "gitu-gitu", "aja", "ajalah", "doang",
    "cuma", "pake", "pakek", "makanya", "gara-gara", "sampe", "udahan", "kelar",
    "bikin", "ngomong", "ngeliat", "ngedenger", "mikir", "nyari", "nanya", "nemu",
    "dapet", "ngasih", "kasih", "ambil", "narik", "nyalain", "matiin", "baper",
    "kepo", "pansos", "curhat", "nongkrong", "tongkrongan", "wkwk", "wkwkwk", "haha",
    "hahaha", "santai", "selow", "sans", "rebahan", "mager", "gabut", "bucin", "alay"
}

# Common English stopwords to gauge English dominance
ENGLISH_CORE_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
    "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
    "than", "then", "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first", "well", "way",
    "even", "new", "want", "because", "any", "these", "give", "day", "most", "us"
}

ALL_INDONESIAN_WORDS = INDONESIAN_CORE_WORDS.union(INDONESIAN_SLANG_WORDS)


class LanguageGateResult(BaseModel):
    """Result of language evaluation."""
    eligible: bool
    primary_language: str
    confidence: float
    reason: str
    id_ratio: float = 0.0
    en_ratio: float = 0.0
    total_analyzed_words: int = 0
    detected_indonesian_tokens: int = 0
    detected_english_tokens: int = 0


class LanguageGate:
    """Deterministic language gate enforcing Indonesian-Only content."""

    MIN_TOTAL_WORDS: int = 6           # Fail safe on empty / near-empty audio
    MIN_ID_RATIO: float = 0.10         # Minimum Indonesian token ratio to qualify
    MAX_EN_RATIO: float = 0.50         # English dominance threshold

    def evaluate_text_sample(self, text: str) -> LanguageGateResult:
        """Evaluates raw text string directly."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = [w for w in cleaned.split() if w and not w.isdigit()]
        return self._evaluate_tokens(tokens)

    def evaluate_transcript(
        self,
        transcript_segments: Optional[List[Union[Dict[str, Any], Any]]],
        explicit_whisper_lang: Optional[str] = None
    ) -> LanguageGateResult:
        """
        Evaluates speech/transcript tokens deterministically.
        Handles:
        - Indonesian formal and slang
        - Indonesian + English code-switching
        - English-dominant rejection
        - Empty or foreign language rejection
        """
        if not transcript_segments:
            return LanguageGateResult(
                eligible=False,
                primary_language="unknown",
                confidence=0.0,
                reason="Empty or missing transcript segments",
                id_ratio=0.0,
                en_ratio=0.0,
                total_analyzed_words=0
            )

        # Extract all words from segments
        tokens: List[str] = []
        for seg in transcript_segments:
            if isinstance(seg, dict):
                text = seg.get("text", "")
                if not text and "words" in seg:
                    text = " ".join(w.get("word", "") for w in seg.get("words", []))
            else:
                text = getattr(seg, "text", "")
                if not text and hasattr(seg, "words"):
                    text = " ".join(getattr(w, "word", "") for w in seg.words)

            cleaned = re.sub(r"[^\w\s]", " ", text.lower())
            tokens.extend([w for w in cleaned.split() if w and not w.isdigit()])

        return self._evaluate_tokens(tokens, explicit_whisper_lang=explicit_whisper_lang)

    def _evaluate_tokens(
        self,
        tokens: List[str],
        explicit_whisper_lang: Optional[str] = None
    ) -> LanguageGateResult:
        total_words = len(tokens)
        if total_words < self.MIN_TOTAL_WORDS:
            return LanguageGateResult(
                eligible=False,
                primary_language="unknown",
                confidence=0.1,
                reason=f"Insufficient speech content for language determination ({total_words} words)",
                id_ratio=0.0,
                en_ratio=0.0,
                total_analyzed_words=total_words
            )

        id_matches = 0
        en_matches = 0

        for t in tokens:
            if t in ALL_INDONESIAN_WORDS:
                id_matches += 1
            elif t in ENGLISH_CORE_WORDS:
                en_matches += 1

        id_ratio = round(id_matches / total_words, 4)
        en_ratio = round(en_matches / total_words, 4)

        # 1. Foreign Language Check: If neither Indonesian nor English has substantial presence
        if id_matches == 0 and en_matches == 0:
            return LanguageGateResult(
                eligible=False,
                primary_language="foreign",
                confidence=0.9,
                reason="No recognizable Indonesian or English words detected; foreign speech",
                id_ratio=id_ratio,
                en_ratio=en_ratio,
                total_analyzed_words=total_words,
                detected_indonesian_tokens=id_matches,
                detected_english_tokens=en_matches
            )

        # 2. English Dominant: English words outnumber Indonesian words significantly
        if en_matches > id_matches and en_ratio >= 0.25 and id_ratio < 0.25:
            return LanguageGateResult(
                eligible=False,
                primary_language="en",
                confidence=round(min(0.99, max(0.6, en_ratio / (id_ratio + 0.05))), 3),
                reason=f"English-dominant content rejected (EN ratio {en_ratio:.2f} > ID ratio {id_ratio:.2f})",
                id_ratio=id_ratio,
                en_ratio=en_ratio,
                total_analyzed_words=total_words,
                detected_indonesian_tokens=id_matches,
                detected_english_tokens=en_matches
            )

        # 3. Indonesian Dominant or Valid Code-Switching:
        if id_matches >= en_matches and id_ratio >= self.MIN_ID_RATIO:
            confidence = round(min(0.99, max(0.65, id_ratio + 0.3)), 3)
            reason = "Indonesian speech is dominant" if en_matches == 0 else "Indonesian speech is dominant with natural English code-switching"
            return LanguageGateResult(
                eligible=True,
                primary_language="id",
                confidence=confidence,
                reason=reason,
                id_ratio=id_ratio,
                en_ratio=en_ratio,
                total_analyzed_words=total_words,
                detected_indonesian_tokens=id_matches,
                detected_english_tokens=en_matches
            )

        # 4. Borderline / Ambiguous check
        if id_matches > 0 and id_matches >= en_matches:
            return LanguageGateResult(
                eligible=True,
                primary_language="id",
                confidence=0.6,
                reason="Indonesian speech tokens detected above English count",
                id_ratio=id_ratio,
                en_ratio=en_ratio,
                total_analyzed_words=total_words,
                detected_indonesian_tokens=id_matches,
                detected_english_tokens=en_matches
            )

        return LanguageGateResult(
            eligible=False,
            primary_language="other",
            confidence=0.7,
            reason=f"Insufficient Indonesian speech presence (ID ratio {id_ratio:.2f}, EN ratio {en_ratio:.2f})",
            id_ratio=id_ratio,
            en_ratio=en_ratio,
            total_analyzed_words=total_words,
            detected_indonesian_tokens=id_matches,
            detected_english_tokens=en_matches
        )

    def evaluate_audio_sample(
        self,
        audio_or_video_path: str,
        sample_duration_sec: float = 30.0
    ) -> LanguageGateResult:
        """
        Samples the first 30 seconds of audio using faster-whisper to evaluate language.
        """
        if not os.path.exists(audio_or_video_path):
            return LanguageGateResult(
                eligible=False,
                primary_language="unknown",
                confidence=0.0,
                reason=f"File not found: {audio_or_video_path}"
            )

        tmp_wav = f"/tmp/lang_sample_{Path(audio_or_video_path).stem}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_or_video_path),
            "-t", str(sample_duration_sec),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            tmp_wav
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
            if res.returncode != 0 or not os.path.exists(tmp_wav):
                return LanguageGateResult(
                    eligible=False,
                    primary_language="unknown",
                    confidence=0.0,
                    reason="Failed to extract audio sample via ffmpeg"
                )

            from faster_whisper import WhisperModel
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(tmp_wav, language=None, beam_size=1)
            seg_list = [{"text": s.text} for s in segments]
            return self.evaluate_transcript(seg_list, explicit_whisper_lang=info.language)
        except Exception as e:
            logger.error(f"Error evaluating audio sample: {e}")
            return LanguageGateResult(
                eligible=False,
                primary_language="unknown",
                confidence=0.0,
                reason=f"Exception in audio transcription: {e}"
            )
        finally:
            if os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass


language_gate = LanguageGate()

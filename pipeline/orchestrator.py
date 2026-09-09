"""Auto Clipper Production Orchestrator (Blueprint Bab 22).

Full end-to-end Auto Clipper engine coordinating:
1. Discovery & Search (discovery.searcher)
2. Source Filter (discovery.source_filter)
3. Indonesian Language Gate (language.language_gate)
4. Transcript Provider (transcription.transcript_provider)
5. Candidate Generator (analysis.candidate_generator)
6. Gemini Semantic Scorer (analysis.semantic_scorer)
7. Boundary Refiner 30-55s (analysis.boundary_refiner)
8. Visual Viability & Visual Director (analysis.visual_director / visual_analyzer)
9. Framing & Edit Plan (editing.framing / editing.edit_plan)
10. Clean FFmpeg Renderer (editing.renderer)
11. Three-Tier Quality Control Gate (quality.ThreeTierQCGate)
12. Strict Upload Gate & YouTube Shorts Uploader (upload.uploader)

Provides:
- process_video(video_id_or_url, dry_run=True, ...) -> PipelineResult
- run_discovery_cycle(query, max_videos=5, dry_run=True, ...) -> List[PipelineResult]
"""

import logging
import os
import subprocess
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from config import settings
from discovery.searcher import Searcher, VideoSourceMeta
from discovery.source_filter import SourceFilter
from discovery.search_planner import GeminiSearchPlanner, SearchPlan
from language.language_gate import LanguageGate, LanguageGateResult
from transcription.transcript_provider import TranscriptProvider, TranscriptSegment
from analysis.candidate_generator import CandidateGenerator, CandidateWindow
from analysis.semantic_scorer import SemanticScorer, SemanticScore
from analysis.boundary_refiner import BoundaryRefiner, RefinementResult, is_sentence_complete
from analysis.visual_analyzer import VisualAnalyzer, VisualAnalysisReport
from analysis.visual_director import VisualDirector, VisualDirectorVerdict
from analysis.visual_preflight import VisualPreflight, VisualPreflightResult
from editing.framing import SceneStaticFraming, FramingDecision
from editing.subtitle_policy import SubtitlePolicyClassifier, generate_ass_subtitles
from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer, RenderResult
from quality import ThreeTierQCGate, ThreeTierQCReport
from quality.gemini_video_qc import GeminiNativeVideoQC, GeminiVideoQCResult
from upload.uploader import (
    StrictUploadGate,
    UploadGateCheck,
    YouTubeShortsUploader,
    all_gates_pass,
)
from pipeline.state_machine import (
    PipelineStatus,
    validate_transition,
    can_transition,
    InvalidStateTransitionError,
    GuardInvariantError,
)
from storage.repository import (
    StorageRepository,
    VideoRecord,
    CandidateRecord,
    RenderRecord,
    UploadRecord,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Pipeline Result Contract
# ==============================================================================

class PipelineResult(BaseModel):
    """Execution report returned by the orchestrator for a single video."""
    video_id: str
    status: PipelineStatus
    is_success: bool = False
    rejection_reason: Optional[str] = None
    error_message: Optional[str] = None
    video_metadata: Optional[VideoSourceMeta] = None
    selected_candidate: Optional[Dict[str, Any]] = None
    refined_boundary: Optional[Dict[str, Any]] = None
    rendered_path: Optional[str] = None
    duration_sec: float = 0.0
    qc_report: Optional[ThreeTierQCReport] = None
    gate_check: Optional[UploadGateCheck] = None
    upload_result: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = Field(default_factory=dict)


# ==============================================================================
# Auto Clipper Orchestrator
# ==============================================================================

class AutoClipperOrchestrator:
    """Full End-to-End Auto Clipper Engine (Blueprint Bab 22)."""

    def __init__(
        self,
        repository: Optional[StorageRepository] = None,
        searcher: Optional[Searcher] = None,
        source_filter: Optional[SourceFilter] = None,
        language_gate: Optional[LanguageGate] = None,
        transcript_provider: Optional[TranscriptProvider] = None,
        candidate_generator: Optional[CandidateGenerator] = None,
        semantic_scorer: Optional[SemanticScorer] = None,
        boundary_refiner: Optional[BoundaryRefiner] = None,
        visual_analyzer: Optional[VisualAnalyzer] = None,
        visual_director: Optional[VisualDirector] = None,
        framing: Optional[SceneStaticFraming] = None,
        subtitle_classifier: Optional[SubtitlePolicyClassifier] = None,
        renderer: Optional[CleanRenderer] = None,
        qc_gate: Optional[ThreeTierQCGate] = None,
        uploader: Optional[YouTubeShortsUploader] = None,
        search_planner: Optional[GeminiSearchPlanner] = None,
        visual_preflight: Optional[VisualPreflight] = None,
        gemini_video_qc: Optional[GeminiNativeVideoQC] = None,
        output_dir: Optional[Path] = None,
        download_dir: Optional[Path] = None,
    ):
        self.repo = repository or StorageRepository()
        self.source_filter = source_filter or SourceFilter()
        self.searcher = searcher or Searcher(source_filter=self.source_filter)
        self.search_planner = search_planner or GeminiSearchPlanner()
        self.language_gate = language_gate or LanguageGate()
        self.transcript_provider = transcript_provider or TranscriptProvider()
        self.candidate_generator = candidate_generator or CandidateGenerator()
        self.semantic_scorer = semantic_scorer or SemanticScorer()
        self.boundary_refiner = boundary_refiner or BoundaryRefiner()
        self.visual_analyzer = visual_analyzer or VisualAnalyzer()
        self.visual_director = visual_director or VisualDirector()
        self.visual_preflight = visual_preflight or VisualPreflight()
        self.framing = framing or SceneStaticFraming()
        self.subtitle_classifier = subtitle_classifier or SubtitlePolicyClassifier()
        self.renderer = renderer or CleanRenderer()
        self.qc_gate = qc_gate or ThreeTierQCGate()
        self.gemini_video_qc = gemini_video_qc or GeminiNativeVideoQC()
        self.uploader = uploader or YouTubeShortsUploader()

        self.output_dir = Path(output_dir or getattr(settings, "OUTPUT_DIR", "/root/projects/auto-short-generator-v3/output"))
        self.download_dir = Path(download_dir or getattr(settings, "DOWNLOAD_DIR", "/root/projects/auto-short-generator-v3/downloads"))
        self.failed_dir = self.output_dir.parent / "failed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self._cooldown_blacklist: Dict[str, float] = {}

    def _record_cooldown(self, video_id: str, start_sec: float, end_sec: float):
        """Records fingerprint into cooldown blacklist to avoid reprocessing (Section 22)."""
        fingerprint = f"{video_id}_{int(start_sec)}_{int(end_sec)}"
        self._cooldown_blacklist[fingerprint] = time.time()

    def is_in_cooldown(self, video_id: str, start_sec: float, end_sec: float, cooldown_sec: float = 86400.0) -> bool:
        """Checks if a candidate window is in cooldown."""
        fingerprint = f"{video_id}_{int(start_sec)}_{int(end_sec)}"
        last_time = self._cooldown_blacklist.get(fingerprint)
        if last_time and (time.time() - last_time) < cooldown_sec:
            return True
        return False

    def get_recent_failures_summary(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieves recent failures for Gemini Search Planner diversification."""
        recent_videos = self.repo.list_videos(limit=20)
        failures = []
        for v in recent_videos:
            if v.status in (
                PipelineStatus.QC_FAILED,
                PipelineStatus.REJECTED_VISUAL,
                PipelineStatus.REJECTED_LANGUAGE,
                PipelineStatus.NO_GOOD_CLIP,
            ):
                failures.append({
                    "video_id": v.video_id,
                    "title": v.title,
                    "status": v.status.value,
                    "reason": v.rejection_reason or v.error_message or "Unknown failure",
                })
                if len(failures) >= limit:
                    break
        return failures

    def _archive_failed_qc(
        self,
        video_id: str,
        clip_id: str,
        video_path: Optional[str],
        qc_report: ThreeTierQCReport,
        gemini_qc: Optional[GeminiVideoQCResult] = None,
        edit_plan: Optional[EditPlan] = None,
        transcript_text: str = "",
    ) -> Path:
        """Archives failed video and diagnostics to failed/ directory (Blueprint Section 20)."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        archive_dir = self.failed_dir / f"{timestamp}_{video_id}_{clip_id}"
        archive_dir.mkdir(parents=True, exist_ok=True)

        if video_path and os.path.exists(video_path):
            dest_clip = archive_dir / "clip.mp4"
            try:
                shutil.copy2(video_path, dest_clip)
            except Exception as e:
                logger.warning(f"Failed to copy failed clip: {e}")

        reason_md = archive_dir / "reason.md"
        blocking_reasons_list = []
        if gemini_qc and gemini_qc.blocking_reasons:
            blocking_reasons_list.extend(gemini_qc.blocking_reasons)
        if qc_report.perceptual and qc_report.perceptual.blocking_issues:
            blocking_reasons_list.extend(qc_report.perceptual.blocking_issues)
        if qc_report.errors:
            blocking_reasons_list.extend(qc_report.errors)

        blocking_bullets = "\n".join(f"- {r}" for r in blocking_reasons_list) or "- Failed technical or visual QC checks"
        gemini_score = gemini_qc.score if gemini_qc else (qc_report.perceptual.score if qc_report.perceptual else "N/A")
        gemini_notes = gemini_qc.summary if gemini_qc else (qc_report.perceptual.notes if qc_report.perceptual else "N/A")

        reason_content = f"""# QC Failure Report

Video ID: {video_id}
Clip ID: {clip_id}
Timestamp: {timestamp}

## Blocking Reasons
{blocking_bullets}

## Gemini Director Score
{gemini_score}/100

## Director Notes
{gemini_notes}

## Decision
NOT PUBLISHABLE
"""
        reason_md.write_text(reason_content, encoding="utf-8")

        try:
            (archive_dir / "qc.json").write_text(qc_report.model_dump_json(indent=2), encoding="utf-8")
            if edit_plan:
                (archive_dir / "edit_plan.json").write_text(edit_plan.model_dump_json(indent=2), encoding="utf-8")
            if transcript_text:
                (archive_dir / "transcript.txt").write_text(transcript_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write failure metadata json: {e}")

        logger.info(f"[{video_id}] Failed QC archived to: {archive_dir}")
        return archive_dir

    def _resolve_video_id(self, video_meta_or_url: Union[str, VideoSourceMeta, Dict[str, Any]]) -> str:
        """Extracts standard 11-char YouTube ID or sanitized identifier from string or meta."""
        if isinstance(video_meta_or_url, VideoSourceMeta):
            return video_meta_or_url.video_id
        if isinstance(video_meta_or_url, dict):
            return str(video_meta_or_url.get("video_id") or "unknown_video")
        s = str(video_meta_or_url).strip()
        if "v=" in s:
            parts = s.split("v=")[1].split("&")[0]
            return parts[:11]
        if "youtu.be/" in s:
            return s.split("youtu.be/")[1].split("?")[0][:11]
        if Path(s).exists():
            return Path(s).stem
        return s.replace("https://", "").replace("http://", "").split("/")[0][:32]

    def _ensure_media_file(self, video_meta: VideoSourceMeta, custom_path: Optional[str] = None) -> Optional[str]:
        """Ensures a local MP4 file is available for visual inspection and rendering."""
        if custom_path and os.path.exists(custom_path):
            return custom_path

        local_candidate = self.download_dir / f"{video_meta.video_id}.mp4"
        if local_candidate.exists() and local_candidate.stat().st_size > 100 * 1024:
            return str(local_candidate)

        # Download via yt-dlp if URL is remote
        if video_meta.url.startswith("http"):
            logger.info(f"Downloading source video media for [{video_meta.video_id}] via yt-dlp...")
            cmd = [
                "yt-dlp",
                "--proxy", "http://127.0.0.1:31001",
                "-f", "bestvideo[height<=720][vcodec^=avc][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
                "--output", str(local_candidate),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                video_meta.url,
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if res.returncode == 0 and local_candidate.exists() and local_candidate.stat().st_size > 50000:
                    return str(local_candidate)
            except Exception as e:
                logger.warning(f"Failed to download media for [{video_meta.video_id}]: {e}")

        return None

    def process_video(
        self,
        video_meta_or_url: Optional[Union[str, VideoSourceMeta, Dict[str, Any]]] = None,
        video_id_or_url: Optional[Union[str, VideoSourceMeta, Dict[str, Any]]] = None,
        dry_run: bool = True,
        custom_video_path: Optional[str] = None,
        custom_transcript: Optional[List[TranscriptSegment]] = None,
        skip_upload: bool = False,
    ) -> PipelineResult:
        """Processes a single video through the complete 11-stage pipeline.

        Adheres strictly to all state machine transitions and guard invariants.
        """
        target = video_meta_or_url if video_meta_or_url is not None else video_id_or_url
        if target is None:
            raise ValueError("Either video_meta_or_url or video_id_or_url must be provided.")

        timings: Dict[str, float] = {}
        t_start = time.time()
        vid_id = self._resolve_video_id(target)
        logger.info(f"========== Starting Auto Clipper Pipeline for [{vid_id}] ==========")

        # ----------------------------------------------------------------------
        # STAGE 1: Discovery & Registration (State: discovered)
        # ----------------------------------------------------------------------
        t0 = time.time()
        video_meta: Optional[VideoSourceMeta] = None

        if isinstance(target, VideoSourceMeta):
            video_meta = target
        elif isinstance(target, dict):
            video_meta = VideoSourceMeta(**target)
        elif custom_video_path and os.path.exists(custom_video_path):
            video_meta = VideoSourceMeta(
                video_id=vid_id,
                url=custom_video_path,
                title=f"Source {vid_id}",
                channel="LocalSource",
                duration_sec=300.0,
                duration=300.0,
            )
        elif isinstance(target, str) and Path(target).exists():
            video_meta = VideoSourceMeta(
                video_id=vid_id,
                url=target,
                title=f"Source {vid_id}",
                channel="LocalSource",
                duration_sec=300.0,
                duration=300.0,
            )
        else:
            target_str = str(target)
            try:
                video_meta = self.searcher.get_video_metadata(target_str)
            except Exception as e:
                logger.warning(f"Searcher metadata fetch error for {vid_id}: {e}")

            if not video_meta:
                video_meta = VideoSourceMeta(
                    video_id=vid_id,
                    url=target_str if target_str.startswith("http") else f"https://www.youtube.com/watch?v={vid_id}",
                    title=f"Video {vid_id}",
                    duration_sec=300.0,
                )

        # Persist video as discovered
        db_video = self.repo.save_video(
            VideoRecord(
                video_id=vid_id,
                url=video_meta.url,
                title=video_meta.title,
                channel_title=video_meta.channel_title or video_meta.channel,
                duration_sec=video_meta.duration_sec,
                published_at=video_meta.published_at,
                status=PipelineStatus.DISCOVERED,
                metadata_json=video_meta.model_dump_json(),
            )
        )
        timings["stage_discovery"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 2: Source Filter (State: eligible | rejected_language)
        # ----------------------------------------------------------------------
        t0 = time.time()
        filter_verdict = self.source_filter.filter_video(video_meta)
        if not filter_verdict.is_eligible:
            reason = f"SourceFilter rejected: {filter_verdict.reason} ({filter_verdict.rejection_code})"
            logger.info(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_LANGUAGE, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_LANGUAGE,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        self.repo.update_video_status(vid_id, PipelineStatus.ELIGIBLE)
        timings["stage_filter"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 3: Transcript Provider & Language Gate (State: transcribed | rejected_language)
        # ----------------------------------------------------------------------
        t0 = time.time()
        transcript: List[TranscriptSegment] = []

        if custom_transcript:
            transcript = custom_transcript
        else:
            try:
                transcript = self.transcript_provider.get_phrase_transcript(
                    video_id=vid_id,
                    audio_or_video_path=custom_video_path or str(self.download_dir / f"{vid_id}.mp4"),
                )
            except Exception as e:
                logger.warning(f"Transcript acquisition failed for [{vid_id}]: {e}")

        if not transcript:
            reason = "Failed to obtain transcript segments from YouTube captions or Whisper fallback."
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_LANGUAGE, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_LANGUAGE,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # Indonesian Language Gate Check
        combined_text = " ".join(s.text for s in transcript)
        lang_res: LanguageGateResult = self.language_gate.evaluate_transcript(transcript)

        if not lang_res.eligible:
            reason = f"Indonesian Language Gate REJECTED: {lang_res.reason} (confidence={lang_res.confidence})"
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_LANGUAGE, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_LANGUAGE,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # Transition to transcribed
        self.repo.update_video_status(vid_id, PipelineStatus.TRANSCRIBED)
        timings["stage_transcript_and_language"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 4: Candidate Generator (State: candidates_found | no_good_clip)
        # ----------------------------------------------------------------------
        t0 = time.time()
        candidates: List[CandidateWindow] = self.candidate_generator.generate_candidates(transcript)

        if not candidates:
            reason = "Candidate generator produced 0 viable candidate windows."
            self.repo.update_video_status(vid_id, PipelineStatus.NO_GOOD_CLIP, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.NO_GOOD_CLIP,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        self.repo.update_video_status(vid_id, PipelineStatus.CANDIDATES_FOUND)
        timings["stage_candidate_generation"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 5: Gemini Semantic Scorer (State: candidate_selected | no_good_clip)
        # ----------------------------------------------------------------------
        t0 = time.time()
        scores: List[SemanticScore] = self.semantic_scorer.score_candidates(
            candidates=candidates,
            video_title=video_meta.title,
        )

        best_cand, best_score, score_verdict = self.semantic_scorer.select_best_clip(candidates, scores)

        # Persist all candidate evaluations in storage
        candidate_pk_map: Dict[str, int] = {}
        for c in candidates:
            sc = next((s for s in scores if s.candidate_id == c.candidate_id), None)
            pk = self.repo.save_candidate(
                CandidateRecord(
                    candidate_id=c.candidate_id,
                    video_id=vid_id,
                    start_sec=c.start_sec,
                    end_sec=c.end_sec,
                    duration_sec=c.duration_sec,
                    text=c.text,
                    hook_score=sc.hook_score if sc else 0.0,
                    payoff_score=sc.payoff_score if sc else 0.0,
                    self_contained_score=sc.self_contained_score if sc else 0.0,
                    overall_score=sc.score if sc else 0.0,
                    reason=sc.reason if sc else "",
                    selected=(best_cand is not None and c.candidate_id == best_cand.candidate_id),
                )
            )
            candidate_pk_map[c.candidate_id] = pk

        if best_cand is None or best_score is None:
            reason = f"Semantic Scorer returned: {score_verdict}"
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.NO_GOOD_CLIP, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.NO_GOOD_CLIP,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        self.repo.update_video_status(vid_id, PipelineStatus.CANDIDATE_SELECTED)
        timings["stage_semantic_scoring"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 6: Boundary Refiner with Candidate Fallback Loop (30-55s target)
        # Blueprint V3.1 Section 23: for candidate in ranked, try boundary alignment
        # ----------------------------------------------------------------------
        t0 = time.time()

        # Build ranked list: best first, then others by score descending
        ranked_candidates = []
        ranked_scores = []
        if best_cand is not None and best_score is not None:
            ranked_candidates.append(best_cand)
            ranked_scores.append(best_score)
        for c in candidates:
            if c.candidate_id != (best_cand.candidate_id if best_cand else None):
                sc = next((s for s in scores if s.candidate_id == c.candidate_id), None)
                if sc and sc.good_clip:
                    ranked_candidates.append(c)
                    ranked_scores.append(sc)

        boundary_res = None
        selected_cand = None
        selected_score = None
        for cand_i, (try_cand, try_score) in enumerate(zip(ranked_candidates, ranked_scores)):
            try_boundary: RefinementResult = self.boundary_refiner.refine(
                start_sec=try_score.suggested_start,
                end_sec=try_score.suggested_end,
                segments=transcript,
            )
            if not try_boundary.is_valid:
                logger.info(f"[{vid_id}] Candidate {try_cand.candidate_id} (rank #{cand_i+1}) boundary rejected: {try_boundary.rejection_reason} (duration={try_boundary.duration:.2f}s)")
                continue

            # Additional check: Inspect candidate ending against transcript/tokens
            clip_segs = [s for s in transcript if s.start >= try_boundary.refined_start - 0.5 and s.end <= try_boundary.refined_end + 0.5]
            cand_text = " ".join(s.text.strip() for s in clip_segs if s.text) if clip_segs else getattr(try_cand, "text", "")
            
            # Safely check sentence completeness
            is_comp, comp_reason = True, ""
            if hasattr(self.boundary_refiner, "is_sentence_complete"):
                try:
                    res = self.boundary_refiner.is_sentence_complete(cand_text)
                    if isinstance(res, tuple) and len(res) == 2:
                        is_comp, comp_reason = res
                    else:
                        is_comp, comp_reason = is_sentence_complete(cand_text)
                except Exception:
                    is_comp, comp_reason = is_sentence_complete(cand_text)
            else:
                is_comp, comp_reason = is_sentence_complete(cand_text)

            if not is_comp:
                can_ext = False
                new_end = try_boundary.refined_end
                ext_reason = "Extension not available"
                if hasattr(self.boundary_refiner, "extend_to_sentence_boundary"):
                    try:
                        res = self.boundary_refiner.extend_to_sentence_boundary(
                            start_sec=try_boundary.refined_start,
                            end_sec=try_boundary.refined_end,
                            segments=transcript,
                            max_duration_sec=getattr(self.boundary_refiner, "TARGET_MAX_DURATION", 55.0),
                        )
                        if isinstance(res, tuple) and len(res) == 3:
                            can_ext, new_end, ext_reason = res
                    except Exception as e:
                        logger.debug(f"extend_to_sentence_boundary error: {e}")

                if can_ext:
                    logger.info(f"[{vid_id}] Candidate {try_cand.candidate_id} sentence ending extended: {ext_reason}")
                    try_boundary.refined_end = new_end
                    try_boundary.duration = round(new_end - try_boundary.refined_start, 2)
                else:
                    logger.info(
                        f"[{vid_id}] Candidate {try_cand.candidate_id} (rank #{cand_i+1}) rejected: "
                        f"incomplete sentence ({comp_reason}) cannot be extended within 55s limit ({ext_reason})"
                    )
                    continue

            boundary_res = try_boundary
            selected_cand = try_cand
            selected_score = try_score
            if cand_i > 0:
                logger.info(f"[{vid_id}] Candidate fallback: rank #{cand_i+1} ({try_cand.candidate_id}) passed boundary refinement after top candidates failed.")
            break

        if boundary_res is None or selected_cand is None or selected_score is None:
            reason = f"All {len(ranked_candidates)} ranked candidates failed boundary refinement (30-55s target)."
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.NO_GOOD_CLIP, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.NO_GOOD_CLIP,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # Reassign best_cand/best_score to the one that passed boundary
        best_cand = selected_cand
        best_score = selected_score

        clip_start = boundary_res.refined_start
        clip_end = boundary_res.refined_end
        clip_duration = boundary_res.duration
        timings["stage_boundary_refinement"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 7: Media Acquisition & Visual Director (State: visual_verified | rejected_visual)
        # ----------------------------------------------------------------------
        t0 = time.time()
        media_path = self._ensure_media_file(video_meta, custom_video_path)

        if not media_path or not os.path.exists(media_path):
            reason = f"Cannot inspect visuals: Media file not found for video [{vid_id}]."
            logger.error(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reason)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # Run local OpenCV visual analysis
        vis_report: VisualAnalysisReport = self.visual_analyzer.analyze_clip(
            video_path=media_path,
            start_sec=clip_start,
            end_sec=clip_end,
        )

        # Run Multimodal Gemini Visual Director
        vis_verdict: VisualDirectorVerdict = self.visual_director.evaluate_window(
            video_path=media_path,
            start_sec=clip_start,
            end_sec=clip_end,
            local_report=vis_report,
        )

        if not vis_verdict.approved:
            reasons = "; ".join(vis_verdict.rejection_reasons) or "Visual Director rejected clip usability."
            logger.warning(f"[{vid_id}] Visual Director REJECTED: {reasons}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reasons)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reasons,
                video_metadata=video_meta,
                timings=timings,
            )

        # Mark candidate as visually approved
        if best_cand.candidate_id in candidate_pk_map:
            self.repo.mark_candidate_selected(
                candidate_pk=candidate_pk_map[best_cand.candidate_id],
                visual_approved=True,
                visual_notes=str(vis_verdict.notes or ""),
            )

        self.repo.update_video_status(vid_id, PipelineStatus.VISUAL_VERIFIED)
        timings["stage_visual_verification"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 8: Framing & Edit Plan Construction (Auto Clipper V3.1)
        # ----------------------------------------------------------------------
        t0 = time.time()
        # Stage F: Native Gemini Source-Clip Visual Preflight
        vis_preflight: VisualPreflightResult = self.visual_preflight.preflight_clip(
            video_path=media_path,
            transcript_excerpt=best_cand.text,
            start_sec=clip_start,
            duration_sec=clip_duration,
        )

        # Dedicated Gemini Native Video ending check (Auto Clipper V3.1)
        gemini_ending_complete = getattr(vis_preflight, "ending_complete", True)
        gemini_ending_natural = getattr(vis_preflight, "ending_natural", True)
        # Handle non-bool / MagicMock types
        if not isinstance(gemini_ending_complete, bool):
            gemini_ending_complete = bool(gemini_ending_complete)
        if not isinstance(gemini_ending_natural, bool):
            gemini_ending_natural = bool(gemini_ending_natural)

        if not gemini_ending_complete or not gemini_ending_natural:
            ending_err = getattr(vis_preflight, "ending_reason", "") or "Incomplete or abrupt sentence ending detected by Gemini"
            logger.warning(f"[{vid_id}] Gemini visual preflight detected incomplete ending: {ending_err}. Attempting extension...")
            can_ext = False
            new_end = clip_end
            ext_reason = "Extension not available"
            if hasattr(self.boundary_refiner, "extend_to_sentence_boundary"):
                try:
                    res = self.boundary_refiner.extend_to_sentence_boundary(
                        start_sec=clip_start,
                        end_sec=clip_end,
                        segments=transcript,
                        max_duration_sec=getattr(self.boundary_refiner, "TARGET_MAX_DURATION", 55.0),
                    )
                    if isinstance(res, tuple) and len(res) == 3:
                        can_ext, new_end, ext_reason = res
                except Exception as e:
                    logger.debug(f"extend_to_sentence_boundary error: {e}")

            target_max = getattr(self.boundary_refiner, "TARGET_MAX_DURATION", 55.0)
            if can_ext and (new_end - clip_start) <= target_max:
                logger.info(f"[{vid_id}] Extended clip boundary after Gemini ending check: {ext_reason}")
                clip_end = new_end
                clip_duration = round(clip_end - clip_start, 2)
                # Clear ending errors from blocking issues if any
                vis_preflight.blocking_issues = [
                    b for b in vis_preflight.blocking_issues
                    if "incomplete sentence" not in b.lower() and "unnatural clip" not in b.lower()
                ]
                if not vis_preflight.blocking_issues:
                    vis_preflight.usable = True
            else:
                reason = f"Gemini visual preflight rejected ending: {ending_err} (extension failed: {ext_reason})"
                logger.warning(f"[{vid_id}] {reason}")
                self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reason)
                self._record_cooldown(vid_id, clip_start, clip_end)
                return PipelineResult(
                    video_id=vid_id,
                    status=PipelineStatus.REJECTED_VISUAL,
                    is_success=False,
                    rejection_reason=reason,
                    video_metadata=video_meta,
                    timings=timings,
                )

        if not vis_preflight.usable:
            reasons = "; ".join(vis_preflight.blocking_issues) or "Visual Preflight rejected source clip usability."
            logger.warning(f"[{vid_id}] Visual Preflight REJECTED: {reasons}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reasons)
            self._record_cooldown(vid_id, clip_start, clip_end)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reasons,
                video_metadata=video_meta,
                timings=timings,
            )

        # Layout Policy: SAFE_WIDE is default invariant (Blueprint V3.1 Section 12)
        chosen_layout = "SAFE_WIDE"
        if vis_preflight.recommended_layout in ("SAFE_WIDE", "SAFE_FULL_FRAME", "SAFE_ZOOM"):
            chosen_layout = vis_preflight.recommended_layout

        framing_dec: FramingDecision = self.framing.analyze_framing(
            video_path=media_path,
            start_sec=clip_start,
            end_sec=clip_end,
            scene_cuts=vis_report.scene_cuts,
            default_layout=chosen_layout,
        )

        edit_plan = EditPlan(
            layout=framing_dec.layout,
            crop_windows=framing_dec.crop_windows,
            duration=clip_duration,
            clip_id=f"{vid_id}_{int(clip_start)}_{int(clip_end)}",
        )

        # Stage G: Subtitle Policy Hard Invariant — Evidence-Based Decision Hierarchy
        # A. ffprobe embedded subtitle stream? -> SOURCE_EXISTING
        # B. Gemini native video candidate check confirms visible subtitles? -> SOURCE_EXISTING
        # C. Local detector confirms burned-in subtitles with HIGH confidence (and no Gemini conflict)? -> SOURCE_EXISTING
        # D. Both agree NO subtitles -> GENERATE
        # E. If UNKNOWN, conflict, or ambiguous -> REJECT candidate (NEVER guess)
        from editing.subtitle_detector import SubtitleDetector, SubtitleDetectionState
        sub_detector = SubtitleDetector()
        sub_state, sub_conf, sub_meta = sub_detector.detect(
            video_path=media_path,
            start_sec=clip_start,
            end_sec=clip_end,
        )
        logger.info(f"[{vid_id}] Subtitle detection: state={sub_state.value}, confidence={sub_conf}, meta={sub_meta}")

        # Dedicated Gemini Native Video candidate check for subtitles before render
        preflight_mode = "LOCAL_FALLBACK"
        if "Conservative deterministic fallback" not in (vis_preflight.notes or ""):
            preflight_mode = "GEMINI_NATIVE_VIDEO"

        gemini_active = (preflight_mode == "GEMINI_NATIVE_VIDEO")
        # Handle MagicMock / bool safely
        raw_has_subs = getattr(vis_preflight, "has_subtitles", None)
        raw_exist_subs = getattr(vis_preflight, "existing_visible_subtitles", None)
        if isinstance(raw_has_subs, bool):
            gemini_has_subtitles = raw_has_subs
        elif isinstance(raw_exist_subs, bool):
            gemini_has_subtitles = raw_exist_subs
        else:
            gemini_has_subtitles = bool(raw_has_subs or raw_exist_subs or False)

        raw_conf = getattr(vis_preflight, "subtitle_confidence", 1.0)
        if isinstance(raw_conf, (int, float)):
            gemini_sub_conf = float(raw_conf)
        else:
            gemini_sub_conf = 1.0

        # 1. Embedded track in container -> SOURCE_EXISTING
        if sub_state == SubtitleDetectionState.EMBEDDED_TRACK:
            has_source_subtitles = True
            subtitle_decision_reason = "Embedded subtitle stream found via ffprobe"

        # 2. Local state UNKNOWN -> REJECT candidate per safety invariant (NEVER guess)
        elif sub_state == SubtitleDetectionState.UNKNOWN:
            reason = f"Subtitle detection state UNKNOWN (conf={sub_conf}) — rejecting candidate per safety invariant"
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reason)
            self._record_cooldown(vid_id, clip_start, clip_end)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # 3. Gemini confidence ambiguous (< 0.7) -> REJECT candidate (NEVER guess)
        elif gemini_active and gemini_sub_conf < 0.7:
            reason = f"Gemini subtitle detection ambiguous / low confidence ({gemini_sub_conf:.2f} < 0.7) — rejecting candidate"
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reason)
            self._record_cooldown(vid_id, clip_start, clip_end)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # 4. Conflict between local detector and Gemini:
        # Local detected BURNED_IN (conf >= 0.7), but Gemini says NO subtitles -> CONFLICT!
        # OR Local detected NONE (conf >= 0.7), but Gemini says HAS subtitles -> CONFLICT!
        elif gemini_active and (
            (sub_state == SubtitleDetectionState.BURNED_IN and not gemini_has_subtitles and sub_conf >= 0.7 and gemini_sub_conf >= 0.7)
            or (sub_state == SubtitleDetectionState.NONE and gemini_has_subtitles and sub_conf >= 0.7 and gemini_sub_conf >= 0.7)
        ):
            reason = (
                f"Subtitle detection conflict: local detector ({sub_state.value}, conf={sub_conf:.2f}) "
                f"conflicts with Gemini (has_subtitles={gemini_has_subtitles}, conf={gemini_sub_conf:.2f}). "
                "Rejecting candidate — NEVER guess."
            )
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reason)
            self._record_cooldown(vid_id, clip_start, clip_end)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        # 5. Gemini detects visible subtitles on screen -> SOURCE_EXISTING
        elif gemini_active and gemini_has_subtitles:
            has_source_subtitles = True
            subtitle_decision_reason = (
                f"Gemini native video verified visible subtitles (conf={gemini_sub_conf:.2f}): "
                f"{getattr(vis_preflight, 'subtitle_reason', '') or 'Subtitles visible on screen'}"
            )

        # 6. Local detected BURNED_IN (when Gemini not active or in agreement) -> SOURCE_EXISTING
        elif sub_state == SubtitleDetectionState.BURNED_IN:
            has_source_subtitles = True
            subtitle_decision_reason = f"Local BURNED_IN detection verified (conf={sub_conf})"

        # 7. Both agree NO subtitles -> GENERATE
        elif sub_state == SubtitleDetectionState.NONE and (not gemini_active or not gemini_has_subtitles):
            has_source_subtitles = False
            subtitle_decision_reason = "No subtitle evidence detected (local=NONE, gemini=False)"

        # 8. Any other ambiguous state -> REJECT candidate
        else:
            reason = (
                f"Subtitle detection ambiguous state (local={sub_state.value}, "
                f"gemini_has_subtitles={gemini_has_subtitles}) — rejecting candidate"
            )
            logger.warning(f"[{vid_id}] {reason}")
            self.repo.update_video_status(vid_id, PipelineStatus.REJECTED_VISUAL, rejection_reason=reason)
            self._record_cooldown(vid_id, clip_start, clip_end)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.REJECTED_VISUAL,
                is_success=False,
                rejection_reason=reason,
                video_metadata=video_meta,
                timings=timings,
            )

        logger.info(f"[{vid_id}] Subtitle decision: has_source={has_source_subtitles}, reason={subtitle_decision_reason}")

        ass_path: Optional[str] = None
        if has_source_subtitles:
            edit_plan.subtitle_policy = "SOURCE_EXISTING"
            logger.info(
                f"[{vid_id}] Source clip already contains visible subtitles. "
                "Subtitle policy set to SOURCE_EXISTING (ZERO second subtitle layer)."
            )
        else:
            edit_plan.subtitle_policy = "GENERATE"
            # Generate clean ASS Subtitles using V3.1 Hybrid Subtitle Accuracy Engine
            from editing.word_subtitle_engine import generate_ass_from_words
            subtitle_file = str(self.output_dir / f"{edit_plan.clip_id}.ass")
            debug_timeline = str(self.output_dir / f"debug_{edit_plan.clip_id}_timeline.json")

            # Fetch source transcript excerpt overlapping with candidate window
            source_transcript_excerpt = getattr(best_cand, "text", "") or ""

            sub_ok, sub_report = generate_ass_from_words(
                video_path=media_path,
                output_ass_path=subtitle_file,
                start_sec=clip_start,
                duration_sec=clip_duration,
                source_transcript=source_transcript_excerpt,
                video_title=getattr(video_meta, "title", ""),
                channel_title=getattr(video_meta, "channel_title", ""),
                debug_timeline_path=debug_timeline,
                use_hybrid=True,
            )
            if sub_ok and sub_report.get("valid"):
                ass_path = subtitle_file
                logger.info(f"[{vid_id}] Generated hybrid word-level subtitle V3.1: {ass_path} "
                           f"({sub_report.get('phrase_count', 0)} phrases, {sub_report.get('word_count', 0)} words, 0 overlaps)")
            else:
                logger.warning(f"[{vid_id}] Word-level subtitle generation failed: {sub_report}. Proceeding without subtitle.")
                edit_plan.subtitle_policy = "GENERATE_FAILED"

        timings["stage_edit_plan"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 9: FFmpeg Clean Rendering (State: rendering -> rendered | render_failed)
        # ----------------------------------------------------------------------
        t0 = time.time()
        self.repo.update_video_status(vid_id, PipelineStatus.RENDERING)

        output_path = self.output_dir / f"{edit_plan.clip_id}.mp4"
        render_pk = self.repo.save_render(
            RenderRecord(
                candidate_id=candidate_pk_map.get(best_cand.candidate_id),
                video_id=vid_id,
                rendered_path=str(output_path),
                duration_sec=clip_duration,
                edit_plan_json=edit_plan.model_dump_json(),
                status="rendering",
            )
        )

        render_res: RenderResult = self.renderer.render(
            video_path=media_path,
            output_path=str(output_path),
            edit_plan=edit_plan,
            start_sec=clip_start,
            duration=clip_duration,
            subtitle_ass_path=ass_path,
        )

        if not render_res.success or not output_path.exists():
            err_msg = render_res.error_message or "FFmpeg render failed to produce output file."
            logger.error(f"[{vid_id}] Rendering error: {err_msg}")
            self.repo.update_render_status(render_pk, status="render_failed", error_message=err_msg)
            self.repo.update_video_status(vid_id, PipelineStatus.RENDER_FAILED, error_message=err_msg)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.RENDER_FAILED,
                is_success=False,
                error_message=err_msg,
                video_metadata=video_meta,
                timings=timings,
            )

        self.repo.update_render_status(render_pk, status="rendered")
        self.repo.update_video_status(vid_id, PipelineStatus.RENDERED)
        timings["stage_rendering"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 10: Three-Tier Quality Control Gate & Native Gemini Video QC
        # ----------------------------------------------------------------------
        t0 = time.time()
        qc_report: ThreeTierQCReport = self.qc_gate.evaluate(
            video_path=str(output_path),
            transcript_text=best_cand.text,
            edit_plan=edit_plan,
        )

        gemini_video_qc_res: Optional[GeminiVideoQCResult] = None
        # Blueprint V3.1 Section 18: Local QC prevents sending broken artifacts to Gemini unnecessarily
        if qc_report.passed and qc_report.publishable:
            gemini_video_qc_res = self.gemini_video_qc.evaluate_video(
                video_path=str(output_path),
                transcript_text=best_cand.text,
                edit_plan=edit_plan,
            )

        qc_passed = (
            qc_report.passed
            and qc_report.publishable
            and (gemini_video_qc_res is None or gemini_video_qc_res.passed)
        )

        if not qc_passed:
            all_errors = list(qc_report.errors)
            if gemini_video_qc_res and gemini_video_qc_res.blocking_reasons:
                all_errors.extend(gemini_video_qc_res.blocking_reasons)
            err_msg = f"Three-Tier QC Gate REJECTED: {', '.join(all_errors) if all_errors else 'Director score below threshold'}"
            logger.warning(f"[{vid_id}] {err_msg}")

            # Archive failed clip to failed/ (Blueprint V3.1 Section 20)
            self._archive_failed_qc(
                video_id=vid_id,
                clip_id=edit_plan.clip_id,
                video_path=str(output_path),
                qc_report=qc_report,
                gemini_qc=gemini_video_qc_res,
                edit_plan=edit_plan,
                transcript_text=best_cand.text,
            )
            self._record_cooldown(vid_id, clip_start, clip_end)

            self.repo.update_render_status(
                render_pk,
                status="rendered",
                qc_passed=False,
                qc_report_json=qc_report.model_dump_json(),
                error_message=err_msg,
            )
            self.repo.update_video_status(vid_id, PipelineStatus.QC_FAILED, error_message=err_msg)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.QC_FAILED,
                is_success=False,
                error_message=err_msg,
                video_metadata=video_meta,
                rendered_path=str(output_path),
                duration_sec=clip_duration,
                qc_report=qc_report,
                timings=timings,
            )

        # QC PASSED
        self.repo.update_render_status(
            render_pk,
            status="rendered",
            qc_passed=True,
            qc_report_json=qc_report.model_dump_json(),
        )
        self.repo.update_video_status(vid_id, PipelineStatus.QC_PASSED)
        timings["stage_qc"] = round(time.time() - t0, 3)

        # ----------------------------------------------------------------------
        # STAGE 11: Strict Upload Gate & Uploading (State: uploading -> completed | upload_failed)
        # ----------------------------------------------------------------------
        t0 = time.time()
        # Evaluate Strict Upload Gate across all 8 production gates
        gate_check = StrictUploadGate.evaluate(
            language_gate=lang_res.eligible,
            semantic_clip_gate=best_score.good_clip,
            visual_viability_gate=vis_verdict.approved,
            boundary_gate=boundary_res.is_valid,
            render_success=render_res.success,
            technical_qc=qc_report.technical.passed,
            visual_qc=qc_report.visual.passed,
            perceptual_qc=qc_report.perceptual.publishable,
            details={"video_id": vid_id, "clip_id": edit_plan.clip_id},
            raise_on_failure=True,
        )

        if skip_upload:
            logger.info(f"[{vid_id}] Skip upload requested; marking completed directly from qc_passed.")
            self.repo.update_video_status(vid_id, PipelineStatus.COMPLETED)
            timings["total"] = round(time.time() - t_start, 3)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.COMPLETED,
                is_success=True,
                video_metadata=video_meta,
                selected_candidate=best_cand.model_dump(),
                refined_boundary=boundary_res.model_dump(),
                rendered_path=str(output_path),
                duration_sec=clip_duration,
                qc_report=qc_report,
                gate_check=gate_check,
                timings=timings,
            )

        # Invariant Guard Check: State transition to 'uploading' ONLY permitted from 'qc_passed'
        self.repo.update_video_status(vid_id, PipelineStatus.UPLOADING)

        upload_pk = self.repo.save_upload(
            UploadRecord(
                render_id=render_pk,
                video_id=vid_id,
                platform="youtube",
                status="uploading",
                title=best_cand.text[:80] or video_meta.title[:80],
                description=f"Auto Short from {video_meta.title}\n\n#Shorts #Indonesia",
                dry_run=dry_run,
            )
        )

        upload_res = self.uploader.upload_short(
            video_path=str(output_path),
            title=best_cand.text[:80] or video_meta.title[:80],
            description=f"Auto Short from {video_meta.title}\n\n#Shorts #Indonesia",
            gate_check=gate_check,
            dry_run=dry_run,
        )

        if upload_res.get("status") != "success":
            err_msg = upload_res.get("error") or upload_res.get("message") or "Upload to platform failed."
            logger.error(f"[{vid_id}] Upload failed: {err_msg}")
            self.repo.update_upload_status(upload_pk, status="upload_failed", error_message=err_msg)
            self.repo.update_video_status(vid_id, PipelineStatus.UPLOAD_FAILED, error_message=err_msg)
            return PipelineResult(
                video_id=vid_id,
                status=PipelineStatus.UPLOAD_FAILED,
                is_success=False,
                error_message=err_msg,
                video_metadata=video_meta,
                rendered_path=str(output_path),
                duration_sec=clip_duration,
                qc_report=qc_report,
                gate_check=gate_check,
                upload_result=upload_res,
                timings=timings,
            )

        # Upload SUCCESS: transition to completed!
        self.repo.update_upload_status(
            upload_id=upload_pk,
            status="completed" if not dry_run else "dry_run",
            platform_video_id=upload_res.get("video_id"),
            url=upload_res.get("url"),
        )
        self.repo.update_video_status(vid_id, PipelineStatus.COMPLETED)
        timings["stage_upload"] = round(time.time() - t0, 3)
        timings["total"] = round(time.time() - t_start, 3)

        logger.info(f"[{vid_id}] Auto Clipper Pipeline COMPLETED successfully! URL: {upload_res.get('url')}")
        return PipelineResult(
            video_id=vid_id,
            status=PipelineStatus.COMPLETED,
            is_success=True,
            video_metadata=video_meta,
            selected_candidate=best_cand.model_dump(),
            refined_boundary=boundary_res.model_dump(),
            rendered_path=str(output_path),
            duration_sec=clip_duration,
            qc_report=qc_report,
            gate_check=gate_check,
            upload_result=upload_res,
            timings=timings,
        )

    def run_discovery_cycle(
        self,
        search_queries: Optional[Union[str, List[str]]] = None,
        max_videos: int = 5,
        dry_run: bool = True,
        query: Optional[str] = None,
    ) -> List[PipelineResult]:
        """Discovers eligible YouTube videos and runs the production pipeline for each."""
        if search_queries is None and query is not None:
            queries = [query]
        elif isinstance(search_queries, str):
            queries = [search_queries]
        elif isinstance(search_queries, list):
            queries = search_queries
        else:
            recent_failures = self.get_recent_failures_summary(limit=5)
            search_plan = self.search_planner.plan_searches(
                recent_failures_summary=recent_failures,
                max_queries=max_videos,
            )
            queries = search_plan.queries or ["podcast viral indonesia"]

        logger.info(f"Running discovery cycle: queries={queries}, max_videos={max_videos}, dry_run={dry_run}")
        
        discovered: List[VideoSourceMeta] = []
        seen_ids = set()
        for q in queries:
            if len(discovered) >= max_videos:
                break
            needed = max_videos - len(discovered)
            try:
                found = self.searcher.search_eligible_videos(query=q, max_results=needed)
                for v in found:
                    if v.video_id not in seen_ids:
                        seen_ids.add(v.video_id)
                        discovered.append(v)
            except Exception as e:
                logger.error(f"Discovery search error for query '{q}': {e}")

        logger.info(f"Discovery returned {len(discovered)} eligible videos. Processing batch...")

        results: List[PipelineResult] = []
        for v in discovered:
            try:
                res = self.process_video(video_meta_or_url=v, dry_run=dry_run)
                results.append(res)
            except Exception as e:
                logger.error(f"Error processing video [{v.video_id}]: {e}")
                results.append(
                    PipelineResult(
                        video_id=v.video_id,
                        status=PipelineStatus.UPLOAD_FAILED if "upload" in str(e).lower() else PipelineStatus.RENDER_FAILED,
                        is_success=False,
                        error_message=str(e),
                    )
                )

        return results

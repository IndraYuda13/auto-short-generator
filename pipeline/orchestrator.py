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
from analysis.boundary_refiner import BoundaryRefiner, RefinementResult
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
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
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
        # STAGE 6: Boundary Refiner (30-55s target)
        # ----------------------------------------------------------------------
        t0 = time.time()
        boundary_res: RefinementResult = self.boundary_refiner.refine(
            start_sec=best_score.suggested_start,
            end_sec=best_score.suggested_end,
            segments=transcript,
        )

        if not boundary_res.is_valid:
            reason = f"Boundary Refiner REJECTED: {boundary_res.rejection_reason} (duration={boundary_res.duration:.2f}s)"
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
                visual_notes=vis_verdict.notes,
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

        # Stage G: Subtitle Policy Hard Invariant
        # Case 1: If source already contains visible subtitles -> DO NOT GENERATE NEW SUBTITLES!
        burned_present, _ = self.subtitle_classifier.check_burned_in_subtitles(
            video_path=media_path,
            start_sec=clip_start,
            end_sec=clip_end,
        )
        has_source_subtitles = vis_preflight.existing_visible_subtitles or burned_present

        ass_path: Optional[str] = None
        if has_source_subtitles:
            edit_plan.subtitle_policy = "SOURCE_EXISTING"
            logger.info(
                f"[{vid_id}] Source clip already contains visible subtitles. "
                "Subtitle policy set to SOURCE_EXISTING (ZERO second subtitle layer)."
            )
        else:
            edit_plan.subtitle_policy = "GENERATE"
            # Generate clean ASS Subtitles
            clip_segments = [
                {
                    "start": max(0.0, round(s.start - clip_start, 2)),
                    "end": max(0.0, round(s.end - clip_start, 2)),
                    "text": s.text,
                }
                for s in transcript
                if s.start >= clip_start - 0.5 and s.end <= clip_end + 0.5
            ]
            subtitle_file = self.output_dir / f"{edit_plan.clip_id}.ass"
            generate_ass_subtitles(
                phrases_or_words=clip_segments,
                output_path=str(subtitle_file),
            )
            ass_path = str(subtitle_file)

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

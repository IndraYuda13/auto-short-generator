"""Live Audit Script: End-to-End Orchestrator Verification on Real Media Samples.
Audits Sample A and Sample B through the full 11-stage AutoClipperOrchestrator.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/root/projects/auto-short-generator-v3")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import AutoClipperOrchestrator, PipelineResult
from storage.repository import StorageRepository
from discovery.searcher import VideoSourceMeta
from transcription.transcript_provider import TranscriptProvider, TranscriptSegment
from analysis.candidate_generator import CandidateGenerator, CandidateWindow
from analysis.semantic_scorer import SemanticScorer, SemanticScore
from analysis.boundary_refiner import BoundaryRefiner
from analysis.visual_analyzer import VisualAnalyzer
from analysis.visual_director import VisualDirector
from editing.framing import SceneStaticFraming
from editing.subtitle_policy import SubtitlePolicyClassifier
from editing.renderer import CleanRenderer
from quality import ThreeTierQCGate
from upload.uploader import YouTubeShortsUploader
from pipeline.state_machine import PipelineStatus


class ApprovedCandidateSemanticScorer(SemanticScorer):
    """Wrapper that returns good_clip=True on the primary candidate window for live render audit."""
    def score_candidates(self, candidates, video_title=""):
        scores = super().score_candidates(candidates, video_title=video_title)
        # Ensure at least the primary candidate is approved for rendering verification
        if candidates:
            c = candidates[0]
            # Override primary candidate to enable render pipeline verification
            scores[0] = SemanticScore(
                candidate_id=c.candidate_id,
                good_clip=True,
                score=88.0,
                hook_score=85.0,
                payoff_score=85.0,
                self_contained_score=90.0,
                reason="Live audit: Candidate window approved for full downstream verification.",
                suggested_start=c.start_sec,
                suggested_end=min(c.start_sec + 35.0, c.end_sec),
            )
        return scores


def run_live_audit():
    print("================================================================================")
    print("PRISM LIVE AUDIT: Auto Clipper Orchestrator End-to-End on Real Media")
    print("================================================================================")

    output_dir = PROJECT_ROOT / "output" / "live_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = PROJECT_ROOT / "data" / "live_audit.db"
    if db_path.exists():
        db_path.unlink()

    repo = StorageRepository(db_path=db_path)
    semantic_scorer = ApprovedCandidateSemanticScorer()
    
    orch = AutoClipperOrchestrator(
        repository=repo,
        semantic_scorer=semantic_scorer,
        output_dir=output_dir,
        download_dir=PROJECT_ROOT / "downloads",
    )

    results = {}

    samples = [
        {
            "id": "sample_a_live_audit",
            "path": str(PROJECT_ROOT / "downloads" / "sample_a_h264_clip.mp4"),
            "title": "Sample A: Indonesian Talking Head Single Speaker",
            "channel": "IndoSpeakerA",
        },
        {
            "id": "sample_b_live_audit",
            "path": str(PROJECT_ROOT / "downloads" / "sample_b_h264_clip.mp4"),
            "title": "Sample B: Indonesian Two-Person Podcast",
            "channel": "IndoPodcastB",
        }
    ]

    for s in samples:
        vid_id = s["id"]
        vpath = s["path"]
        print(f"\n>>> EXECUTING PIPELINE FOR [{vid_id}] ({s['title']})...")
        meta = VideoSourceMeta(
            video_id=vid_id,
            url=vpath,
            title=s["title"],
            channel=s["channel"],
            duration_sec=300.0,
        )

        t0 = time.time()
        res: PipelineResult = orch.process_video(
            video_meta_or_url=meta,
            custom_video_path=vpath,
            dry_run=True,
        )
        elapsed = time.time() - t0
        print(f"[{vid_id}] Finished in {elapsed:.2f}s | Status: {res.status.value} | Success: {res.is_success}")
        
        if res.rendered_path and os.path.exists(res.rendered_path):
            print(f"[{vid_id}] Rendered short produced: {res.rendered_path}")
            # Run ffprobe verification
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", res.rendered_path
            ]
            probe = subprocess.run(cmd, capture_output=True, text=True)
            probe_data = json.loads(probe.stdout)
            v = next((x for x in probe_data["streams"] if x["codec_type"] == "video"), {})
            a = next((x for x in probe_data["streams"] if x["codec_type"] == "audio"), {})
            print(f"[{vid_id}] Technical Specs:")
            print(f"       Video: {v.get('width')}x{v.get('height')}, codec={v.get('codec_name')}, fps={v.get('r_frame_rate')}")
            print(f"       Audio: {a.get('channels')}ch, codec={a.get('codec_name')}, sample_rate={a.get('sample_rate')}Hz")
            print(f"       Duration: {float(probe_data['format'].get('duration', 0)):.2f}s")
            
            # EBU R128 loudness measurement
            ebur_cmd = [
                "ffmpeg", "-nostats", "-i", res.rendered_path,
                "-filter_complex", "ebur128=peak=true", "-f", "null", "-"
            ]
            ebur_run = subprocess.run(ebur_cmd, capture_output=True, text=True)
            ebur_out = ebur_run.stderr
            # Extract Integrated Loudness and True Peak
            lufs = "N/A"
            tp = "N/A"
            for line in ebur_out.splitlines():
                if "I:" in line and "LUFS" in line:
                    lufs = line.strip()
                if "Peak:" in line and "dBFS" in line:
                    tp = line.strip()
            print(f"[{vid_id}] Audio Loudness: {lufs} | {tp}")
        
        if res.qc_report:
            qc = res.qc_report
            print(f"[{vid_id}] Three-Tier QC: Passed={qc.passed}, Publishable={qc.publishable}")
            print(f"       Tier 1 (Technical): Passed={qc.technical.passed}, errors={qc.technical.errors}")
            print(f"       Tier 2 (Visual): Passed={qc.visual.passed}, subject_ratio={qc.visual.subject_present_ratio:.2f}, subtitle_safe={qc.visual.subtitle_safe}, errors={qc.visual.errors}")
            print(f"       Tier 3 (Perceptual): Passed={qc.perceptual.passed}, Score={qc.perceptual.score}/100, blocking={qc.perceptual.blocking_issues}")
            print(f"       Director Notes: {qc.perceptual.notes}")

        if res.gate_check:
            print(f"[{vid_id}] Strict Upload Gate: Passed={res.gate_check.passed}")
            print(f"       Checks: {res.gate_check.model_dump()}")

        results[vid_id] = {
            "status": res.status.value,
            "is_success": res.is_success,
            "rendered_path": res.rendered_path,
            "timings": res.timings,
            "duration": res.duration_sec,
            "qc_passed": res.qc_report.passed if res.qc_report else False,
            "publishable": res.qc_report.publishable if res.qc_report else False,
            "gate_passed": res.gate_check.passed if res.gate_check else False,
        }

    summary_file = output_dir / "live_audit_summary.json"
    summary_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved live audit summary to {summary_file}")
    return results

if __name__ == "__main__":
    run_live_audit()

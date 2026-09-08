"""Auto Clipper V3.1 Stable Release Stress Test Runner.

Target: 3 SUCCESSFULLY PUBLISHED PRIVATE CLIPS via fully automatic pipeline.
Source Code: FROZEN at RC_COMMIT (no mutations allowed during test).

Behavior:
- Loops search -> transcript -> candidate -> preflight -> render -> QC -> upload
- On content-quality failure: archive + blacklist + immediate new search (no 300s sleep)
- On technical failure (network/API): bounded backoff
- Stops when 3 successful private uploads are achieved
- Generates stable_review/ pack with all evidence
"""

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path("/root/projects/auto-short-generator-v3")
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import AutoClipperOrchestrator, PipelineResult, PipelineStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(PROJECT_ROOT / "stable_review" / "stress_test.log"), mode="a"),
    ],
)
logger = logging.getLogger("stress_test")

REVIEW_DIR = PROJECT_ROOT / "stable_review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SUCCESSES = 3
MAX_CYCLES = 15  # safety cap to prevent infinite loop
TECHNICAL_BACKOFF_SEC = 30


def run_stress_test():
    successes = []
    failures = []
    cycle_count = 0

    orchestrator = AutoClipperOrchestrator()

    logger.info("=" * 80)
    logger.info(f"STRESS TEST START — Target: {TARGET_SUCCESSES} successful private uploads")
    logger.info(f"RC_COMMIT: 92856c4d10974995f0b4ed886fc406d386e8d2cf")
    logger.info("=" * 80)

    while len(successes) < TARGET_SUCCESSES and cycle_count < MAX_CYCLES:
        cycle_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE #{cycle_count} | Successes so far: {len(successes)}/{TARGET_SUCCESSES}")
        logger.info(f"{'='*60}")

        try:
            results = orchestrator.run_discovery_cycle(
                max_videos=3,
                dry_run=False,
            )
        except Exception as e:
            logger.error(f"Cycle #{cycle_count} crashed with exception: {e}")
            failures.append({
                "cycle": cycle_count,
                "type": "technical_crash",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            logger.info(f"Technical backoff: {TECHNICAL_BACKOFF_SEC}s...")
            time.sleep(TECHNICAL_BACKOFF_SEC)
            continue

        if not results:
            logger.warning(f"Cycle #{cycle_count}: No results returned (all filtered or search empty).")
            failures.append({
                "cycle": cycle_count,
                "type": "no_results",
                "error": "Discovery cycle returned 0 results",
                "timestamp": datetime.now().isoformat(),
            })
            continue

        for res in results:
            if res.is_success and res.status == PipelineStatus.COMPLETED:
                success_idx = len(successes) + 1
                logger.info(f"✅ SUCCESS #{success_idx}: video_id={res.video_id}, path={res.rendered_path}")

                # Save to review pack
                success_dir = REVIEW_DIR / f"success_{success_idx:02d}"
                success_dir.mkdir(parents=True, exist_ok=True)

                # Copy rendered clip
                if res.rendered_path and os.path.exists(res.rendered_path):
                    shutil.copy2(res.rendered_path, success_dir / "clip.mp4")

                # Save evidence artifacts
                evidence = {
                    "video_id": res.video_id,
                    "status": res.status.value,
                    "rendered_path": res.rendered_path,
                    "duration_sec": res.duration_sec,
                    "cycle": cycle_count,
                    "timestamp": datetime.now().isoformat(),
                }
                if res.video_metadata:
                    evidence["source"] = {
                        "title": res.video_metadata.title,
                        "url": res.video_metadata.url,
                        "duration_sec": res.video_metadata.duration_sec,
                    }
                    (success_dir / "source.json").write_text(json.dumps(evidence["source"], indent=2))

                if res.selected_candidate:
                    (success_dir / "edit_plan.json").write_text(json.dumps(res.selected_candidate, indent=2))

                if res.refined_boundary:
                    evidence["boundary"] = res.refined_boundary

                if res.qc_report:
                    qc_data = {
                        "technical_passed": res.qc_report.technical.passed if res.qc_report.technical else None,
                        "visual_passed": res.qc_report.visual.passed if res.qc_report.visual else None,
                        "perceptual_passed": res.qc_report.perceptual.passed if res.qc_report.perceptual else None,
                        "perceptual_score": res.qc_report.perceptual.score if res.qc_report.perceptual else None,
                    }
                    evidence["qc"] = qc_data

                if res.upload_result:
                    (success_dir / "upload.json").write_text(json.dumps(res.upload_result, indent=2))
                    evidence["upload"] = res.upload_result

                if res.timings:
                    evidence["timings"] = res.timings

                (success_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))

                successes.append(evidence)

                if len(successes) >= TARGET_SUCCESSES:
                    break
            else:
                fail_entry = {
                    "cycle": cycle_count,
                    "video_id": res.video_id,
                    "status": res.status.value,
                    "type": "content_rejection" if res.status in (
                        PipelineStatus.NO_GOOD_CLIP,
                        PipelineStatus.REJECTED_LANGUAGE,
                        PipelineStatus.REJECTED_VISUAL,
                        PipelineStatus.QC_FAILED,
                    ) else "pipeline_failure",
                    "reason": res.rejection_reason or res.error_message or "Unknown",
                    "timestamp": datetime.now().isoformat(),
                }
                failures.append(fail_entry)
                logger.info(f"❌ FAIL: video_id={res.video_id}, status={res.status.value}, reason={fail_entry['reason'][:120]}")

    # Generate reports
    logger.info("\n" + "=" * 80)
    logger.info(f"STRESS TEST COMPLETE: {len(successes)}/{TARGET_SUCCESSES} successes, {len(failures)} failures, {cycle_count} cycles")
    logger.info("=" * 80)

    # stress_test_report.md
    report_lines = [
        "# Auto Clipper V3.1 Stress Test Report\n",
        f"- **RC Commit**: `5626c14cc383deb3aea5af6fa91569eb12bad654`",
        f"- **Test Date**: {datetime.now().isoformat()[:19]}",
        f"- **Total Cycles**: {cycle_count}",
        f"- **Total Successes**: {len(successes)}",
        f"- **Total Failures/Rejections**: {len(failures)}",
        "",
    ]
    for i, s in enumerate(successes, 1):
        report_lines.append(f"\n## Success #{i}")
        report_lines.append(f"- Video ID: `{s['video_id']}`")
        if "source" in s:
            report_lines.append(f"- Source Title: {s['source'].get('title', 'N/A')}")
        if "upload" in s:
            report_lines.append(f"- YouTube URL: {s['upload'].get('url', 'N/A')}")
        report_lines.append(f"- Duration: {s.get('duration_sec', 'N/A')}s")

    (REVIEW_DIR / "stress_test_report.md").write_text("\n".join(report_lines))

    # failed_summary.md
    fail_lines = [
        "# Failed/Rejected Clips Summary\n",
        f"Total: {len(failures)}\n",
    ]
    content_rejections = [f for f in failures if f.get("type") == "content_rejection"]
    technical_failures = [f for f in failures if f.get("type") != "content_rejection"]
    fail_lines.append(f"- Content Rejections (expected): {len(content_rejections)}")
    fail_lines.append(f"- Technical/Pipeline Failures: {len(technical_failures)}")
    fail_lines.append("")
    for i, f in enumerate(failures, 1):
        fail_lines.append(f"### Failure #{i}")
        fail_lines.append(f"- Cycle: {f.get('cycle')}")
        fail_lines.append(f"- Video: `{f.get('video_id', 'N/A')}`")
        fail_lines.append(f"- Type: {f.get('type')}")
        fail_lines.append(f"- Status: {f.get('status', 'N/A')}")
        fail_lines.append(f"- Reason: {f.get('reason', 'N/A')}")
        fail_lines.append("")

    (REVIEW_DIR / "failed_summary.md").write_text("\n".join(fail_lines))

    # Final JSON manifest
    manifest = {
        "rc_commit": "92856c4d10974995f0b4ed886fc406d386e8d2cf",
        "test_completed": datetime.now().isoformat(),
        "total_cycles": cycle_count,
        "successes": successes,
        "failure_count": len(failures),
        "content_rejection_count": len(content_rejections),
        "technical_failure_count": len(technical_failures),
        "target_met": len(successes) >= TARGET_SUCCESSES,
    }
    (REVIEW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return len(successes) >= TARGET_SUCCESSES


if __name__ == "__main__":
    success = run_stress_test()
    sys.exit(0 if success else 1)

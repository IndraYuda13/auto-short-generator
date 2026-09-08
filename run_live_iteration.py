"""Run 1 Real Iteration of Auto Clipper V3.1 Native Gemini Decision Pipeline.

Performs:
1. Search via Gemini Search Planner -> YouTube Searcher
2. End-to-End Processing of top eligible video
3. Verification of all gates and upload to YouTube (Private status)
4. Displays complete telemetry and output video path
"""

import os
import sys
import json
import logging
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path("/root/projects/auto-short-generator-v3")
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.orchestrator import AutoClipperOrchestrator, PipelineResult
from storage.repository import StorageRepository

# Setup rich console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("live_iteration")

def main():
    print("=" * 80)
    print("STARTING 1 LIVE ITERATION: AUTO CLIPPER V3.1 NATIVE PIPELINE")
    print("=" * 80)

    orchestrator = AutoClipperOrchestrator()

    # Step 1: Run discovery cycle with Gemini Search Planner
    logger.info("Executing discovery cycle with Gemini Search Planner...")
    start_time = time.time()
    results = orchestrator.run_discovery_cycle(
        max_videos=1,
        dry_run=False,      # REAL UPLOAD (private status)
    )
    total_duration = time.time() - start_time

    print("\n" + "=" * 80)
    print("ITERATION EXECUTION COMPLETED")
    print("=" * 80)

    if not results:
        logger.warning("No videos were processed in this cycle (all filtered or no candidates).")
        return

    for idx, res in enumerate(results, 1):
        print(f"\n--- Result #{idx} for Video ID: {res.video_id} ---")
        print(f"Status: {res.status.value}")
        print(f"Success: {res.is_success}")
        if res.error_message:
            print(f"Error/Reject Reason: {res.error_message}")
        if res.rejection_reason:
            print(f"Rejection Reason: {res.rejection_reason}")
        if res.rendered_path:
            print(f"Rendered Output Path: {res.rendered_path}")
            if os.path.exists(res.rendered_path):
                size_mb = os.path.getsize(res.rendered_path) / (1024 * 1024)
                print(f"File Size: {size_mb:.2f} MB")
        if res.duration_sec:
            print(f"Duration: {res.duration_sec:.2f} seconds")
        if res.qc_report:
            print(f"Technical QC: {res.qc_report.technical.passed}")
            print(f"Visual QC: {res.qc_report.visual.passed}")
            if res.qc_report.perceptual:
                print(f"Perceptual QC Score: {res.qc_report.perceptual.score}/100")
                print(f"Perceptual Notes: {res.qc_report.perceptual.notes}")
        if res.gate_check:
            print(f"Upload Gate Passed: {res.gate_check.passed}")
        if res.timings:
            print(f"Timings: {json.dumps(res.timings, indent=2)}")

if __name__ == "__main__":
    main()

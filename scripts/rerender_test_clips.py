"""Rerender Test Clips Runner (Auto Clipper V3.1 Hybrid Subtitle Accuracy).

Executes end-to-end verification for target clips:
1. fQbpsIQpi08 (308.0 - 350.0, 42.0s)
2. UxAHTdGR7do (402.0 - 443.0, 41.0s)

Validates critical phrases:
- 'muka bumi' (fixes 'mukabomi')
- 'pertama' (fixes 'peren pertama')
- 'who are you? siapa kamu?' (fixes 'huayu')
- 'nomor tiga' (fixes 'nama tiga')
"""

import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set storage cache for huggingface
os.environ.setdefault("HF_HOME", "/mnt/storage/hermes_home_orion/.cache/huggingface")

from editing.edit_plan import EditPlan
from editing.renderer import CleanRenderer
from editing.word_subtitle_engine import generate_hybrid_subtitles
from quality.gemini_video_qc import GeminiNativeVideoQC, GeminiVideoQCResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("rerender_test_clips")


def get_db_source_transcript(video_id: str, start_sec: float, end_sec: float) -> str:
    """Retrieves source transcript excerpt from app_v3.db candidates table."""
    db_path = PROJECT_ROOT / "data" / "app_v3.db"
    if not db_path.exists():
        return ""
    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        c.execute(
            "SELECT text FROM candidates WHERE video_id = ? "
            "ORDER BY ABS(start_sec - ?) ASC LIMIT 1",
            (video_id, start_sec),
        )
        row = c.fetchone()
        conn.close()
        if row and row[0]:
            return str(row[0]).strip()
    except Exception as e:
        logger.warning(f"Failed to query DB for {video_id}: {e}")
    return ""


def evaluate_with_retry(qc: GeminiNativeVideoQC, video_path: str, transcript: str, plan: EditPlan, max_retries: int = 3) -> GeminiVideoQCResult:
    """Evaluates video with retry against transient 503/server errors."""
    for attempt in range(max_retries):
        try:
            res = qc.evaluate_video(video_path, transcript_text=transcript, edit_plan=plan)
            if "503" not in res.summary and "unavailable" not in res.summary.lower():
                return res
            logger.warning(f"Transient 503 from 9router on attempt {attempt+1}, backing off 5s...")
            time.sleep(5)
        except Exception as e:
            logger.warning(f"Exception during QC attempt {attempt+1}: {e}")
            time.sleep(5)
    return qc.evaluate_video(video_path, transcript_text=transcript, edit_plan=plan)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", choices=["clip1", "clip2", "all"], default="all", help="Target clip to render")
    args, _ = parser.parse_known_args()

    clips = [
        {
            "id": "clip1",
            "video_id": "fQbpsIQpi08",
            "start_sec": 308.0,
            "end_sec": 350.0,
            "duration": 42.0,
            "source_video": str(PROJECT_ROOT / "downloads" / "fQbpsIQpi08.mp4"),
            "output_ass": str(PROJECT_ROOT / "output" / "fQbpsIQpi08_308_350_v3.ass"),
            "output_mp4": str(PROJECT_ROOT / "output" / "fQbpsIQpi08_308_350_v3.mp4"),
            "timeline_json": str(PROJECT_ROOT / "output" / "fQbpsIQpi08_308_350_v3_timeline.json"),
            "video_title": "Jika MUAK Dengan Keadaan Hidup , Konsumsi Podcast Ini! Dijamin POV-mu Berubah 100% | Peeptlk #33",
            "channel_title": "Peep Talk",
            "critical_terms": ["senyumnya sumringah", "satpam", "ongkir", "wanita ini", "aku S1"],
        },
        {
            "id": "clip2",
            "video_id": "UxAHTdGR7do",
            "start_sec": 402.0,
            "end_sec": 443.0,
            "duration": 41.0,
            "source_video": str(PROJECT_ROOT / "downloads" / "UxAHTdGR7do.mp4"),
            "output_ass": str(PROJECT_ROOT / "output" / "UxAHTdGR7do_402_443_v3.ass"),
            "output_mp4": str(PROJECT_ROOT / "output" / "UxAHTdGR7do_402_443_v3.mp4"),
            "timeline_json": str(PROJECT_ROOT / "output" / "UxAHTdGR7do_402_443_v3_timeline.json"),
            "video_title": "Kalau Kamu Mau HIDUPMU Gini-Gini Terus, SKIP Aja Video Ini! | SUARA BERKELAS #172",
            "channel_title": "SUARA BERKELAS",
            "critical_terms": ["muka bumi", "pertama", "who are you? siapa kamu?", "nomor tiga"],
        },
    ]

    if args.clip != "all":
        clips = [c for c in clips if c["id"] == args.clip]

    renderer = CleanRenderer()
    qc_evaluator = GeminiNativeVideoQC()
    summary_path = PROJECT_ROOT / "output" / "rerender_test_summary.json"
    results = {}
    if summary_path.exists():
        try:
            results = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    for clip in clips:
        cid = clip["id"]
        vid = clip["video_id"]
        logger.info(f"\n==================================================")
        logger.info(f"PROCESSING {cid}: {vid} [{clip['start_sec']}s - {clip['end_sec']}s]")
        logger.info(f"==================================================")

        source_video = clip["source_video"]
        if not os.path.exists(source_video):
            logger.error(f"Source video missing: {source_video}")
            continue

        # 1. Source transcript
        source_transcript = get_db_source_transcript(vid, clip["start_sec"], clip["end_sec"])
        logger.info(f"Source Transcript Excerpt ({len(source_transcript)} chars): {source_transcript[:120]}...")

        # 2. Generate Hybrid Subtitles
        t0 = time.time()
        sub_ok, sub_report, phrases = generate_hybrid_subtitles(
            video_path=source_video,
            output_ass_path=clip["output_ass"],
            start_sec=clip["start_sec"],
            duration_sec=clip["duration"],
            source_transcript_excerpt=source_transcript,
            video_title=clip["video_title"],
            channel_title=clip["channel_title"],
            model_size="large-v3",
            debug_timeline_path=clip["timeline_json"],
        )
        t_sub = time.time() - t0
        logger.info(f"Subtitle Generation: ok={sub_ok}, phrases={len(phrases)}, time={t_sub:.2f}s")
        logger.info(f"ASS Report: {sub_report}")

        if not sub_ok:
            logger.error(f"Failed to generate subtitles for {vid}")
            continue

        # 3. Render using CleanRenderer (SAFE_WIDE layout)
        edit_plan = EditPlan(
            layout="SAFE_WIDE",
            subtitle_policy="GENERATE",
            duration=clip["duration"],
            clip_id=f"{vid}_{int(clip['start_sec'])}_{int(clip['end_sec'])}_v3",
        )

        t0 = time.time()
        render_res = renderer.render(
            video_path=source_video,
            output_path=clip["output_mp4"],
            edit_plan=edit_plan,
            start_sec=clip["start_sec"],
            duration=clip["duration"],
            subtitle_ass_path=clip["output_ass"],
        )
        t_render = time.time() - t0
        logger.info(f"Rendering: success={render_res.success}, output={clip['output_mp4']}, time={t_render:.2f}s")

        if not render_res.success or not os.path.exists(clip["output_mp4"]):
            logger.error(f"Render failed for {vid}: {render_res.error_message}")
            continue

        # 4. Native Gemini Video QC via 9router
        logger.info(f"Running Gemini Native Video QC on {clip['output_mp4']}...")
        t0 = time.time()
        qc_result = evaluate_with_retry(
            qc=qc_evaluator,
            video_path=clip["output_mp4"],
            transcript=source_transcript,
            plan=edit_plan,
        )
        t_qc = time.time() - t0
        logger.info(f"Gemini Video QC: passed={qc_result.passed}, score={qc_result.score}, time={t_qc:.2f}s")
        logger.info(f"  subtitle_timing: {qc_result.subtitle_timing}")
        logger.info(f"  subtitle_overlap: {qc_result.subtitle_overlap}")
        logger.info(f"  subtitle_linger: {qc_result.subtitle_linger}")
        logger.info(f"  subtitle_text_accuracy: {qc_result.subtitle_text_accuracy}")
        logger.info(f"  obvious_transcription_errors: {qc_result.obvious_transcription_errors}")
        logger.info(f"  blocking_reasons: {qc_result.blocking_reasons}")
        logger.info(f"  summary: {qc_result.summary}")

        results[vid] = {
            "clip_id": f"{vid}_{int(clip['start_sec'])}_{int(clip['end_sec'])}",
            "ass_path": clip["output_ass"],
            "mp4_path": clip["output_mp4"],
            "sub_report": sub_report,
            "phrases": phrases,
            "qc_result": qc_result.model_dump(),
        }

        # Save run summary progressively after each clip
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Progressive run summary saved to {summary_path}")

    # Final summary logged
    logger.info(f"\nFinal run summary saved to {summary_path}")


if __name__ == "__main__":
    main()

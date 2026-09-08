"""FFmpeg Clean Filtergraph Executor (Blueprint Bab 15).

Consumes EditPlan, applies scene-static crop windows, subtle punch-ins, ASS subtitles,
and broadcast audio mastering chain to produce 1080x1920 H.264 + AAC 48kHz stereo.
"""

import os
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from editing.edit_plan import EditPlan, SceneCrop, PunchInEvent
from editing.audio import AUDIO_FILTER_CHAIN

logger = logging.getLogger(__name__)


class RenderResult(BaseModel):
    """Result summary of a render execution."""
    output_path: str
    duration: float = 0.0
    width: int = 1080
    height: int = 1920
    filtergraph: str = ""
    command: List[str] = Field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None


class CleanRenderer:
    """Clean filtergraph generator and FFmpeg renderer for short-form content."""

    def __init__(self):
        pass

    def build_filtergraph(
        self,
        edit_plan: EditPlan,
        duration: float,
        start_sec: float = 0.0,
        subtitle_ass_path: Optional[str] = None
    ) -> str:
        """Constructs FFmpeg filtergraph string from EditPlan.

        Handles:
        - Layout: PORTRAIT_9_16 (single or multi scene-static crops) or SAFE_FULL_FRAME
        - Punch-in events: max 2 subtle zoom crops
        - Subtitle ASS integration
        - Audio mastering chain (-16 LUFS, TP -1.5 dB, 48kHz stereo)
        """
        if edit_plan.layout == "REJECT":
            raise ValueError("Cannot construct filtergraph: EditPlan layout is 'REJECT'.")

        filter_parts: List[str] = []

        # --- 1. Video Framing Filter ---
        if edit_plan.layout in ("SAFE_WIDE", "SAFE_FULL_FRAME"):
            # Safe wide / full-frame: preserve 100% original width on 1080x1920 canvas with blurred background
            filter_parts.append(
                "[0:v]split[bg][fg];"
                "[bg]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=10:2,scale=1080:1920[bg_blur];"
                "[fg]scale=1080:-1[fg_scaled];"
                "[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2[v_base]"
            )
        else:
            # PORTRAIT_9_16
            crops = edit_plan.crop_windows
            if len(crops) == 0:
                # Fallback center crop
                filter_parts.append(
                    "[0:v]crop=min(iw\\,ih*9/16):ih:(iw-ow)/2:0,scale=1080:1920[v_base]"
                )
            elif len(crops) == 1:
                # Single scene crop held constant for entire clip
                c = crops[0]
                filter_parts.append(
                    f"[0:v]crop={c.crop_w}:{c.crop_h}:{c.crop_x}:{c.crop_y},scale=1080:1920[v_base]"
                )
            else:
                # Multiple scene cuts: scene-static crops concatenated
                concat_labels: List[str] = []
                for i, c in enumerate(crops):
                    # Clip-local scene start and end
                    seg_s = max(0.0, c.scene_start)
                    seg_e = min(duration, c.scene_end)
                    if seg_e <= seg_s:
                        continue
                    filter_parts.append(
                        f"[0:v]trim=start={seg_s:.2f}:end={seg_e:.2f},setpts=PTS-STARTPTS,"
                        f"crop={c.crop_w}:{c.crop_h}:{c.crop_x}:{c.crop_y},scale=1080:1920[vseg_{i}]"
                    )
                    concat_labels.append(f"[vseg_{i}]")

                num_segments = len(concat_labels)
                if num_segments == 1:
                    filter_parts.append(f"{concat_labels[0]}null[v_base]")
                elif num_segments > 1:
                    concat_str = "".join(concat_labels)
                    filter_parts.append(f"{concat_str}concat=n={num_segments}:v=1:a=0[v_base]")
                else:
                    filter_parts.append(
                        "[0:v]crop=min(iw\\,ih*9/16):ih:(iw-ow)/2:0,scale=1080:1920[v_base]"
                    )

        current_v = "v_base"

        # --- 2. Punch-in Events ---
        punch_events = edit_plan.punch_in_events[:2]
        if punch_events:
            if len(punch_events) == 1:
                ev = punch_events[0]
                t1 = max(0.0, ev.start_time)
                t2 = min(duration, ev.end_time)
                scale = ev.scale
                # Calculate cropped dimensions for zoom (ensure even)
                pw = (int(round(1080.0 / scale)) // 2) * 2
                ph = (int(round(1920.0 / scale)) // 2) * 2

                crop_expr = (
                    f"crop='if(between(t,{t1:.2f},{t2:.2f}),{pw},1080)':"
                    f"'if(between(t,{t1:.2f},{t2:.2f}),{ph},1920)':"
                    f"'(in_w-out_w)/2':'(in_h-out_h)/2',scale=1080:1920"
                )
                filter_parts.append(f"[{current_v}]{crop_expr}[v_punch]")
                current_v = "v_punch"
            elif len(punch_events) >= 2:
                ev1, ev2 = punch_events[0], punch_events[1]
                t1_a = max(0.0, ev1.start_time)
                t2_a = min(duration, ev1.end_time)
                pw_a = (int(round(1080.0 / ev1.scale)) // 2) * 2
                ph_a = (int(round(1920.0 / ev1.scale)) // 2) * 2

                t1_b = max(0.0, ev2.start_time)
                t2_b = min(duration, ev2.end_time)
                pw_b = (int(round(1080.0 / ev2.scale)) // 2) * 2
                ph_b = (int(round(1920.0 / ev2.scale)) // 2) * 2

                w_expr = f"if(between(t,{t1_a:.2f},{t2_a:.2f}),{pw_a},if(between(t,{t1_b:.2f},{t2_b:.2f}),{pw_b},1080))"
                h_expr = f"if(between(t,{t1_a:.2f},{t2_a:.2f}),{ph_a},if(between(t,{t1_b:.2f},{t2_b:.2f}),{ph_b},1920))"

                crop_expr = f"crop='{w_expr}':'{h_expr}':'(in_w-out_w)/2':'(in_h-out_h)/2',scale=1080:1920"
                filter_parts.append(f"[{current_v}]{crop_expr}[v_punch]")
                current_v = "v_punch"

        # --- 3. Subtitle ASS Integration ---
        should_burn_subtitles = (
            edit_plan.subtitle_policy == "GENERATE"
            and subtitle_ass_path is not None
            and os.path.exists(subtitle_ass_path)
        )
        if should_burn_subtitles and subtitle_ass_path:
            escaped_path = subtitle_ass_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
            filter_parts.append(f"[{current_v}]subtitles='{escaped_path}'[v_out]")
        else:
            filter_parts.append(f"[{current_v}]null[v_out]")

        # --- 4. Audio Mastering Filter Chain ---
        if edit_plan.audio_mastering:
            filter_parts.append(f"[0:a]{AUDIO_FILTER_CHAIN}[a_out]")
        else:
            filter_parts.append("[0:a]aformat=sample_rates=48000:channel_layouts=stereo[a_out]")

        return ";".join(filter_parts)

    def render(
        self,
        edit_plan: Optional[EditPlan] = None,
        input_video_path: Optional[str] = None,
        output_video_path: Optional[str] = None,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        subtitle_ass_path: Optional[str] = None,
        video_path: Optional[str] = None,
        output_path: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> RenderResult:
        """Executes FFmpeg with the compiled clean filtergraph.

        Produces 1080x1920 H.264 + AAC 48kHz stereo output.
        """
        in_path = input_video_path or video_path or ""
        out_path_str = output_video_path or output_path or ""
        if edit_plan is None:
            raise ValueError("edit_plan must be provided to render")

        if edit_plan.layout == "REJECT":
            return RenderResult(
                output_path=out_path_str,
                success=False,
                error_message="EditPlan layout is 'REJECT' (candidate unviable)"
            )

        if not os.path.exists(in_path):
            return RenderResult(
                output_path=out_path_str,
                success=False,
                error_message=f"Input video not found: {in_path}"
            )

        # Probe input video duration if not given
        clip_duration = duration if duration is not None else edit_plan.duration
        if clip_duration is None:
            if end_sec is not None and end_sec > start_sec:
                clip_duration = end_sec - start_sec
            else:
                # Default duration fallback
                clip_duration = 30.0

        filtergraph = self.build_filtergraph(
            edit_plan=edit_plan,
            duration=clip_duration,
            start_sec=start_sec,
            subtitle_ass_path=subtitle_ass_path
        )

        out_path = Path(out_path_str)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["ffmpeg", "-y"]

        # Fast seek to start if specified
        if start_sec > 0:
            cmd.extend(["-ss", f"{start_sec:.3f}"])

        cmd.extend([
            "-i", in_path,
            "-filter_complex", filtergraph,
            "-map", "[v_out]",
            "-map", "[a_out]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
        ])

        # Always add duration limit to avoid processing entire source file
        if clip_duration is not None:
            cmd.extend(["-t", f"{clip_duration:.3f}"])

        cmd.extend([
            "-movflags", "+faststart",
            str(out_path)
        ])

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if res.returncode != 0:
                logger.error(f"FFmpeg render failed: {res.stderr}")
                return RenderResult(
                    output_path=str(out_path),
                    duration=clip_duration,
                    filtergraph=filtergraph,
                    command=cmd,
                    success=False,
                    error_message=f"FFmpeg exit {res.returncode}: {res.stderr[-500:]}"
                )

            return RenderResult(
                output_path=str(out_path),
                duration=clip_duration,
                width=1080,
                height=1920,
                filtergraph=filtergraph,
                command=cmd,
                success=out_path.exists()
            )
        except Exception as e:
            logger.error(f"Render exception: {e}")
            return RenderResult(
                output_path=str(out_path),
                duration=clip_duration,
                filtergraph=filtergraph,
                command=cmd,
                success=False,
                error_message=str(e)
            )

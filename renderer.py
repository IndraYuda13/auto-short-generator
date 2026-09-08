"""Renderer V2 Module: Consumes typed EditPlan, applies face-tracked crop or blurred fallback,
calm ASS subtitles, and standardized broadcast audio mastering chain.
"""

import os
import random
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import settings
from edit_plan import EditPlan, FramingMode, EditEventType
from subtitle import subtitle_generator

logger = logging.getLogger(__name__)


class HookType:
    CAMERA_STOMP = "camera_stomp"       # Legacy Hook 1
    PAPER_TEAR = "paper_tear"           # Legacy Hook 2
    BREAKING_NEWS = "breaking_news"     # Legacy Hook 3
    ZOOM_PUNCH = "zoom_punch"           # Legacy Hook 4
    GLITCH_REWIND = "glitch_rewind"     # Legacy Hook 5


class Renderer:
    """Video rendering engine consuming EditPlan."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or settings.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fonts_dir = settings.PROJECT_ROOT / "assets" / "fonts"
        self.fonts_dir.mkdir(parents=True, exist_ok=True)

    def render_short(
        self,
        source_video_path: str,
        start_sec: float,
        end_sec: float,
        clip_id: str,
        subtitle_segments: Optional[List[Dict[str, Any]]] = None,
        edit_plan: Optional[EditPlan] = None,
        hook_type: Optional[str] = None
    ) -> str:
        """
        Renders a short-form video from source:
        - If edit_plan is provided, runs Renderer V2 (deterministic, content-aware framing & punch-ins).
        - If hook_type is explicitly provided or settings.LEGACY_RANDOM_HOOKS_ENABLED is True,
          can run legacy hook pipeline.
        - Default path: Deterministic EditPlan (PODCAST_CLEAN, face-tracked crop or blurred fallback, no random choice).
        """
        output_path = self.output_dir / f"short_{clip_id}.mp4"
        duration = max(0.1, end_sec - start_sec)

        # Legacy hook support strictly opt-in
        use_legacy_hooks = bool(hook_type) or (settings.LEGACY_RANDOM_HOOKS_ENABLED and not settings.EDITOR_V2_ENABLED)

        if use_legacy_hooks:
            available_hooks = [
                HookType.CAMERA_STOMP,
                HookType.PAPER_TEAR,
                HookType.BREAKING_NEWS,
                HookType.ZOOM_PUNCH,
                HookType.GLITCH_REWIND,
            ]
            chosen_hook = hook_type if hook_type in available_hooks else random.choice(available_hooks)
            logger.info(f"[LEGACY] Rendering short {clip_id} using legacy hook '{chosen_hook}'...")
            return self._render_legacy(
                source_video_path=source_video_path,
                start_sec=start_sec,
                end_sec=end_sec,
                clip_id=clip_id,
                subtitle_segments=subtitle_segments,
                chosen_hook=chosen_hook,
                output_path=output_path
            )

        # -------------------------------------------------------------
        # Renderer V2 Path: Deterministic, EditPlan-driven
        # -------------------------------------------------------------
        if edit_plan is None:
            edit_plan = EditPlan.create_default(
                clip_id=clip_id,
                duration=duration,
                framing_mode=FramingMode.BLURRED_FALLBACK
            )

        logger.info(
            f"[V2] Rendering short {clip_id}: duration={duration:.2f}s, "
            f"framing={edit_plan.framing_mode.value}, events={len(edit_plan.edit_events)}"
        )

        # 1. Generate Subtitles via SubtitleGeneratorV2 if allowed by EditPlan
        ass_path = None
        # Invariant: Never generate new subtitles if source already has subtitles
        should_generate_subtitles = edit_plan.generate_new_subtitle and not edit_plan.existing_subtitle
        if should_generate_subtitles and subtitle_segments:
            # Check whether segments have word-level timestamps
            has_words = any(len(s.get("words", [])) > 0 for s in subtitle_segments)
            ass_path = self.output_dir / f"sub_{clip_id}.ass"
            subtitle_generator.generate_ass(
                subtitle_data=subtitle_segments,
                edit_plan=edit_plan,
                output_path=ass_path,
                has_word_timestamps=has_words
            )
        elif edit_plan.existing_subtitle:
            logger.info(f"Existing subtitle present ({edit_plan.subtitle_source}) - skipping ASS generation (NO DUPLICATE SUBTITLES).")

        # 2. Build V2 Filtergraph & Audio Mastering
        filter_complex, extra_inputs, map_audio = self._build_v2_pipeline(
            edit_plan=edit_plan,
            ass_path=ass_path,
            duration=duration
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(source_video_path),
        ]
        cmd.extend(extra_inputs)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", map_audio,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_path)
        ])

        logger.info(f"Executing FFmpeg V2 render for clip {clip_id}...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg render failed:\n{result.stderr[-1500:]}")
            raise RuntimeError(f"FFmpeg render failed with exit code {result.returncode}")

        logger.info(f"Short successfully rendered: {output_path} ({os.path.getsize(output_path)} bytes)")
        return str(output_path)

    def _build_v2_pipeline(
        self,
        edit_plan: EditPlan,
        ass_path: Optional[Path],
        duration: float
    ) -> tuple[str, list[str], str]:
        """
        Builds FFmpeg filtergraph for Renderer V2:
        - Layout: FACE_TRACKED crop or BLURRED_FALLBACK
        - Subtle semantic punch-ins (115% zoom) only if defined in edit_plan
        - ASS subtitle overlay
        - Broadcast audio mastering chain (high-pass, gentle compression, loudness normalization)
        """
        extra_inputs = []
        filter_parts = []

        # 1. Base Visual Layout
        if edit_plan.framing_mode == FramingMode.FACE_TRACKED and edit_plan.crop_keyframes:
            # Check if keyframes have distinct scene segments or single segment
            kfs = edit_plan.crop_keyframes
            # Build time-dependent crop expression or average center crop
            # If multiple keyframes, build nested if piecewise constant expression to eliminate boundary gaps/overlaps
            if len(kfs) == 1:
                avg_center_x = kfs[0].crop_center_x
                crop_expr = (
                    f"crop='ih*9/16':'ih':'min(max(0, iw*{avg_center_x:.4f} - (ih*9/32)), iw - ih*9/16)':'0',"
                    f"scale=1080:1920:flags=bicubic"
                )
            else:
                # Build piecewise nested if expression based on keyframe timestamps:
                # if(lt(t, t1), c0, if(lt(t, t2), c1, ... c_{N-1}))
                # This guarantees contiguous, gap-free, non-overlapping intervals across all t.
                nested_expr = f"{kfs[-1].crop_center_x:.4f}"
                for i in reversed(range(len(kfs) - 1)):
                    t_boundary = kfs[i + 1].time
                    val = kfs[i].crop_center_x
                    nested_expr = f"if(lt(t,{t_boundary:.3f}),{val:.4f},{nested_expr})"

                crop_expr = (
                    f"crop='ih*9/16':'ih':'min(max(0, iw*({nested_expr}) - (ih*9/32)), iw - ih*9/16)':'0',"
                    f"scale=1080:1920:flags=bicubic"
                )

            layout_filter = f"[0:v]{crop_expr}[base_v]"
            filter_parts.append(layout_filter)
            current_v = "[base_v]"
        elif edit_plan.framing_mode == FramingMode.SUBTITLE_SAFE_FULL_WIDTH:
            # Subtitle-Safe Full Width Framing:
            # Ensures 100% of source 16:9 width is preserved inside 9:16 portrait canvas so that
            # horizontal burned-in subtitles are NEVER truncated on left/right edges.
            # Background layer strictly excludes the lower subtitle region (ih*0.70) to ensure
            # ZERO ghost subtitles appear in blurred upper/lower bars.
            # Foreground is full source scaled to 1080 width centered vertically.
            safe_layout = (
                "[0:v]split=2[bg_in][fg_in];"
                "[bg_in]crop=iw:ih*0.70:0:0,scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=5:2,scale=1080:1920:flags=bicubic[bg];"
                "[fg_in]scale=1080:-2:flags=bicubic[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[base_v]"
            )
            filter_parts.append(safe_layout)
            current_v = "[base_v]"
        else:
            # Blurred background fallback:
            # Exclude lower 30% from background blur as well to avoid ghost text
            base_layout = (
                "[0:v]split=2[bg_in][fg_in];"
                "[bg_in]crop=iw:ih*0.70:0:0,scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=5:2,scale=1080:1920:flags=bicubic[bg];"
                "[fg_in]scale=1080:-2[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[base_v]"
            )
            filter_parts.append(base_layout)
            current_v = "[base_v]"

        # 2. Subtle Semantic Punch-In Events (from EditPlan)
        punch_events = [e for e in edit_plan.edit_events if e.type == EditEventType.PUNCH_IN]
        if punch_events:
            # Build conditional crop zoom for each punch-in
            # E.g. at 115% zoom: crop to (1080/1.15 = 939, 1920/1.15 = 1669)
            conds = []
            for ev in punch_events[:2]:  # Max 2 subtle punch-ins
                e_start = ev.time
                e_end = ev.time + (ev.duration or 1.5)
                conds.append(f"between(t,{e_start:.2f},{e_end:.2f})")

            if conds:
                punch_cond = "+".join(conds)
                punch_filter = (
                    f"{current_v}crop="
                    f"'if({punch_cond}, 939, 1080)':"
                    f"'if({punch_cond}, 1669, 1920)':"
                    f"(in_w-out_w)/2:(in_h-out_h)/2,scale=1080:1920[punch_v]"
                )
                filter_parts.append(punch_filter)
                current_v = "[punch_v]"

        # 3. Subtitle Overlay
        if ass_path and ass_path.exists():
            escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            escaped_fonts = str(self.fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            sub_filter = f"{current_v}ass='{escaped_ass}':fontsdir='{escaped_fonts}'[outv]"
            filter_parts.append(sub_filter)
        else:
            passthrough = f"{current_v}null[outv]"
            filter_parts.append(passthrough)

        # 4. Audio Mastering Chain
        map_audio = "0:a"
        if settings.AUDIO_MASTERING_ENABLED and edit_plan.audio_profile:
            ap = edit_plan.audio_profile
            # Voice-first broadcast mastering with standardized 48kHz stereo output:
            # 1. High-pass filter at 80Hz (removes low-end rumble)
            # 2. Mild dynamic range compression (acompressor: ratio=3:1, threshold=-18dB, attack=15ms, release=100ms)
            # 3. Loudnorm: 1-pass integrated loudness normalization to target LUFS (-16.0) with true peak (-1.5dBTP)
            # 4. aresample=48000, aformat=channel_layouts=stereo for social platform compliance
            audio_chain = (
                f"[0:a]highpass=f={ap.highpass_freq},"
                f"acompressor=threshold=-18dB:ratio=3:attack=15:release=100:makeup=2,"
                f"loudnorm=I={ap.loudness_target_lufs:.1f}:TP={ap.true_peak_db:.1f}:LRA=11,"
                f"aresample=48000,aformat=channel_layouts=stereo[aout]"
            )
            filter_parts.append(audio_chain)
            map_audio = "[aout]"

        filter_complex = ";".join(filter_parts)
        return filter_complex, extra_inputs, map_audio

    def _render_legacy(
        self,
        source_video_path: str,
        start_sec: float,
        end_sec: float,
        clip_id: str,
        subtitle_segments: Optional[List[Dict[str, Any]]],
        chosen_hook: str,
        output_path: Path
    ) -> str:
        """Legacy rendering pipeline with 5 visual hook engines for backwards compatibility."""
        duration = end_sec - start_sec
        ass_path = None
        if subtitle_segments is not None:
            ass_path = self._generate_legacy_ass(
                segments=subtitle_segments,
                clip_start=start_sec,
                clip_end=end_sec,
                clip_id=clip_id,
                hook_type=chosen_hook
            )

        filter_complex, extra_inputs, map_audio = self._build_legacy_pipeline(
            ass_path=ass_path,
            duration=duration,
            hook_type=chosen_hook
        )

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(source_video_path),
        ]
        cmd.extend(extra_inputs)
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-map", map_audio,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(output_path)
        ])

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"Legacy FFmpeg render failed:\n{result.stderr[-1000:]}")
            raise RuntimeError(f"Legacy FFmpeg render failed with exit code {result.returncode}")
        return str(output_path)

    def _build_legacy_pipeline(
        self,
        ass_path: Optional[Path],
        duration: float,
        hook_type: str
    ) -> tuple[str, list[str], str]:
        extra_inputs = []
        map_audio = "0:a"
        fg_filter = "scale=1080:-2"
        base_layout = (
            "[0:v]split=2[bg_in][fg_in];"
            "[bg_in]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=5:2,scale=1080:1920:flags=bicubic[bg];"
            f"[fg_in]{fg_filter}[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[merged]"
        )
        current_v = "[merged]"
        hook_filters = []

        if hook_type == HookType.CAMERA_STOMP:
            stomp_v = (
                f"{current_v}scale=1140:2026,"
                f"crop=1080:1920:'(in_w-out_w)/2+26*exp(-7*t)*sin(36*PI*t)*lte(t,0.6)':"
                f"'(in_h-out_h)/2+32*exp(-7*t)*cos(36*PI*t)*lte(t,0.6)'[stomp_v]"
            )
            hook_filters.append(stomp_v)
            current_v = "[stomp_v]"
            extra_inputs.extend([
                "-f", "lavfi",
                "-i", "aevalsrc=sin(2*PI*55*t)*exp(-8*t)*0.85:s=44100:d=0.7"
            ])
            audio_filter = "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]"
            hook_filters.append(audio_filter)
            map_audio = "[aout]"

        elif hook_type == HookType.ZOOM_PUNCH:
            mid_start = 5.0
            mid_end = 6.2 if duration >= 8.0 else -1.0
            punch_v = (
                f"{current_v}crop="
                f"'if(lte(t,1.5), 939, if(between(t,{mid_start},{mid_end}), 1000, 1080))':"
                f"'if(lte(t,1.5), 1669, if(between(t,{mid_start},{mid_end}), 1777, 1920))':"
                f"(in_w-out_w)/2:(in_h-out_h)/2,scale=1080:1920[punch_v]"
            )
            hook_filters.append(punch_v)
            current_v = "[punch_v]"

        elif hook_type == HookType.GLITCH_REWIND:
            glitch_v = f"{current_v}rgbashift=rh=-18:bv=18:gh=8:enable='between(t,0,0.6)'[glitch_v]"
            hook_filters.append(glitch_v)
            current_v = "[glitch_v]"

        if ass_path and os.path.exists(ass_path):
            escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            escaped_fonts = str(self.fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            sub_filter = f"{current_v}ass='{escaped_ass}':fontsdir='{escaped_fonts}'[outv]"
            hook_filters.append(sub_filter)
        else:
            passthrough = f"{current_v}null[outv]"
            hook_filters.append(passthrough)

        all_parts = [base_layout] + hook_filters
        return ";".join(all_parts), extra_inputs, map_audio

    def _generate_legacy_ass(
        self,
        segments: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        clip_id: str,
        hook_type: str
    ) -> Path:
        """Legacy ASS generation."""
        ass_path = self.output_dir / f"sub_{clip_id}.ass"
        header = """[Script Info]
Title: Auto Short Dynamic Subtitles & Hook Engine
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Hormozi,Montserrat Black,44,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,2,0,1,4,2,2,60,60,520,1
Style: BreakingBannerBg,Montserrat Black,40,&H001414D0,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: BreakingBannerText,Montserrat Black,42,&H00FFFFFF,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,2,0,1,3,2,8,0,0,140,1
Style: BreakingTickerBg,Montserrat,36,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,1,0,0,0,1
Style: BreakingTickerText,Rubik Black,36,&H0000E6FF,&H00000000,&H00000000,&H90000000,-1,0,0,0,100,100,1,0,1,3,1,2,30,30,420,1
Style: PaperLeft,Arial,10,&H00EBEFF2,&H00000000,&H00707070,&H50000000,0,0,0,0,100,100,0,0,1,2,4,7,0,0,0,1
Style: PaperRight,Arial,10,&H00DFE4E8,&H00000000,&H00707070,&H50000000,0,0,0,0,100,100,0,0,1,2,4,7,0,0,0,1
Style: RewindOSD,Rubik Black,50,&H0000E6FF,&H00000000,&H00000000,&HA0000000,-1,0,0,0,100,100,2,0,1,4,2,7,80,0,120,1
Style: RewindScanLine,Arial,10,&H70FFFFFF,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events = []
        if hook_type == HookType.BREAKING_NEWS:
            events.append("Dialogue: 1,0:00:00.00,0:00:03.00,BreakingBannerBg,,0,0,0,,{\\p1\\pos(60,110)}m 0 0 l 960 0 l 960 110 l 0 110{\\p0}")
            events.append("Dialogue: 2,0:00:00.00,0:00:03.00,BreakingBannerText,,0,0,0,,{\\pos(540,145)}⚠ BREAKING NEWS ⚠")
            events.append("Dialogue: 1,0:00:00.00,0:00:03.00,BreakingTickerBg,,0,0,0,,{\\p1\\pos(0,1440)}m 0 0 l 1080 0 l 1080 80 l 0 80{\\p0}")
            events.append("Dialogue: 2,0:00:00.00,0:00:03.00,BreakingTickerText,,0,0,0,,{\\pos(540,1460)}🔴 VIRAL UPDATE • SAKSIKAN SAMPAI SELESAI")

        content = header + "\n".join(events) + "\n"
        ass_path.write_text(content, encoding="utf-8")
        return ass_path


renderer = Renderer()

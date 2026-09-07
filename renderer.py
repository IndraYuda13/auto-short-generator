"""Renderer module using FFmpeg: 16:9 to 9:16 blurred background with Hormozi/MrBeast style dynamic subtitles
and 5 viral visual hook techniques (Camera Stomp, Paper Tear, Breaking News, Zoom Punch-In, Glitch Rewind).
"""

import os
import random
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from config import settings

logger = logging.getLogger(__name__)


class HookType:
    CAMERA_STOMP = "camera_stomp"       # Hook 1: 0.0s - 0.6s damped oscillation shake + sub-bass thud SFX
    PAPER_TEAR = "paper_tear"           # Hook 2: 0.0s - 0.8s jagged paper tear reveal
    BREAKING_NEWS = "breaking_news"     # Hook 3: 0.0s - 3.0s crimson top banner + bottom ticker
    ZOOM_PUNCH = "zoom_punch"           # Hook 4: 0.0s - 1.5s 115% snap punch-in + punchline micro-zoom
    GLITCH_REWIND = "glitch_rewind"     # Hook 5: 0.0s - 0.6s rgbashift chromatic aberration + OSD rewind


class Renderer:
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
        hook_type: Optional[str] = None
    ) -> str:
        """
        Takes source 16:9 video, crops & creates 9:16 layout (1080x1920) with blurred background,
        applies dynamic .ass karaoke style subtitles in safe zone Y: 1380px (MarginV=520, 44pt),
        applies 1 of 5 visual hook engines, and renders final MP4.
        """
        output_path = self.output_dir / f"short_{clip_id}.mp4"
        duration = end_sec - start_sec

        # Determine visual hook technique (either explicit or random rotation)
        available_hooks = [
            HookType.CAMERA_STOMP,
            HookType.PAPER_TEAR,
            HookType.BREAKING_NEWS,
            HookType.ZOOM_PUNCH,
            HookType.GLITCH_REWIND,
        ]
        chosen_hook = hook_type if hook_type in available_hooks else random.choice(available_hooks)

        logger.info(
            f"Rendering short {clip_id}: start={start_sec}s, duration={duration:.1f}s, "
            f"hook='{chosen_hook}' -> {output_path}"
        )

        # 1. Create Subtitle (.ass) file with active hook visual vector layers
        ass_path = None
        if subtitle_segments is not None:
            ass_path = self._generate_ass_subtitles(
                segments=subtitle_segments,
                clip_start=start_sec,
                clip_end=end_sec,
                clip_id=clip_id,
                hook_type=chosen_hook
            )

        # 2. Build FFmpeg Filtergraph & Audio Pipeline
        filter_complex, extra_inputs, map_audio = self._build_ffmpeg_pipeline(
            ass_path=ass_path,
            duration=duration,
            hook_type=chosen_hook
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-t", str(duration),
            "-i", str(source_video_path),
        ]

        # Append any auxiliary inputs (e.g. sub-bass thud generator)
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

        logger.info(f"Executing FFmpeg render command for hook '{chosen_hook}'...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"FFmpeg render failed:\n{result.stderr[-1000:]}")
            raise RuntimeError(f"FFmpeg render failed with exit code {result.returncode}")

        logger.info(f"Short successfully rendered: {output_path} ({os.path.getsize(output_path)} bytes)")
        return str(output_path)

    def _build_ffmpeg_pipeline(
        self,
        ass_path: Optional[Path],
        duration: float,
        hook_type: str
    ) -> tuple[str, list[str], str]:
        """
        Constructs the FFmpeg filter_complex, auxiliary inputs, and audio mapping
        tailored to the chosen visual hook.
        """
        extra_inputs = []
        map_audio = "0:a"

        # Base layout:
        # Background: downscaled & blurred with boxblur for fast render, then scaled to 1080x1920
        # Foreground: 1080 width, centered vertically
        # Overlay: foreground placed over blurred background -> [merged]
        fg_filter = "scale=1080:-2"
        base_layout = (
            "[0:v]split=2[bg_in][fg_in];"
            "[bg_in]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=5:2,scale=1080:1920:flags=bicubic[bg];"
            f"[fg_in]{fg_filter}[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2[merged]"
        )

        current_v = "[merged]"
        hook_filters = []

        # -------------------------------------------------------------
        # Hook 1: Camera Stomp Shockwave (0.0s - 0.6s)
        # Screen shake damped oscillation + sub-bass 55Hz thud SFX
        # -------------------------------------------------------------
        if hook_type == HookType.CAMERA_STOMP:
            # Scale slightly up (1140x2026) so shake won't reveal black borders
            stomp_v = (
                f"{current_v}scale=1140:2026,"
                f"crop=1080:1920:'(in_w-out_w)/2+26*exp(-7*t)*sin(36*PI*t)*lte(t,0.6)':"
                f"'(in_h-out_h)/2+32*exp(-7*t)*cos(36*PI*t)*lte(t,0.6)'[stomp_v]"
            )
            hook_filters.append(stomp_v)
            current_v = "[stomp_v]"

            # Synthesize sub-bass thud (55Hz sine wave with rapid exponential decay)
            extra_inputs.extend([
                "-f", "lavfi",
                "-i", "aevalsrc=sin(2*PI*55*t)*exp(-8*t)*0.85:s=44100:d=0.7"
            ])
            # Audio mix: input 0:a and input 1:a
            audio_filter = "[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[aout]"
            hook_filters.append(audio_filter)
            map_audio = "[aout]"

        # -------------------------------------------------------------
        # Hook 4: Content Transition & Zoom Punch-In (0.0s - 1.5s snap cut)
        # 115% snap cut at 0.0s-1.5s, snap back to 100% at >1.5s,
        # plus a punchline micro-zoom (108%) around mid-video if duration > 8s
        # -------------------------------------------------------------
        elif hook_type == HookType.ZOOM_PUNCH:
            # At 0.0 - 1.5s: 115% scale = crop to (1080/1.15 = 939, 1920/1.15 = 1669)
            # At mid-video (e.g. 5.0 - 6.2s if duration >= 8s): micro-zoom 108% (1000x1777)
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

        # -------------------------------------------------------------
        # Hook 5: Rewind Lecek / Glitch Tape Rewind (0.0s - 0.6s)
        # rgbashift chromatic aberration + rewind visual
        # -------------------------------------------------------------
        elif hook_type == HookType.GLITCH_REWIND:
            glitch_v = (
                f"{current_v}rgbashift=rh=-18:bv=18:gh=8:enable='between(t,0,0.6)'[glitch_v]"
            )
            hook_filters.append(glitch_v)
            current_v = "[glitch_v]"

        # -------------------------------------------------------------
        # Subtitle & Vector Graphics Overlay (.ass)
        # -------------------------------------------------------------
        if ass_path and os.path.exists(ass_path):
            escaped_ass = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            escaped_fonts = str(self.fonts_dir).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
            sub_filter = f"{current_v}ass='{escaped_ass}':fontsdir='{escaped_fonts}'[outv]"
            hook_filters.append(sub_filter)
        else:
            passthrough = f"{current_v}null[outv]"
            hook_filters.append(passthrough)

        all_parts = [base_layout] + hook_filters
        filter_complex = ";".join(all_parts)

        return filter_complex, extra_inputs, map_audio

    def _generate_ass_subtitles(
        self,
        segments: List[Dict[str, Any]],
        clip_start: float,
        clip_end: float,
        clip_id: str,
        hook_type: str
    ) -> Path:
        """
        Generate Advanced SubStation Alpha (.ass) file:
        - Font: Montserrat Black / Rubik Black
        - Size: 44pt
        - Position: Y=1380px (MarginV=520 with Alignment 2)
        - Highlight: Neon Yellow (&H0000E6FF) + pop scale 108% (\\fscx108\\fscy108)
        - Chunking: 3-4 words per line with dynamic active word pop
        - Injected visual hook layers for Paper Tear, Breaking News, or Glitch Rewind OSD
        """
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

        # -------------------------------------------------------------
        # Inject Visual Hook Vector/OSD Elements into ASS
        # -------------------------------------------------------------
        if hook_type == HookType.BREAKING_NEWS:
            # Hook 3: Breaking News Crimson Top Banner + Bottom Ticker Bar (0.0s - 3.0s)
            events.append("Dialogue: 1,0:00:00.00,0:00:03.00,BreakingBannerBg,,0,0,0,,{\\p1\\pos(60,110)}m 0 0 l 960 0 l 960 110 l 0 110{\\p0}")
            events.append("Dialogue: 2,0:00:00.00,0:00:03.00,BreakingBannerText,,0,0,0,,{\\pos(540,145)}⚠ BREAKING NEWS ⚠")
            events.append("Dialogue: 1,0:00:00.00,0:00:03.00,BreakingTickerBg,,0,0,0,,{\\p1\\pos(0,1440)}m 0 0 l 1080 0 l 1080 80 l 0 80{\\p0}")
            events.append("Dialogue: 2,0:00:00.00,0:00:03.00,BreakingTickerText,,0,0,0,,{\\pos(540,1460)}🔴 VIRAL UPDATE • SAKSIKAN SAMPAI SELESAI")

        elif hook_type == HookType.PAPER_TEAR:
            # Hook 2: Paper Tear Reveal (0.0s - 0.8s)
            # Left jagged paper sheet moves from x=0 to x=-900
            # Right jagged paper sheet moves from x=0 to x=+900
            events.append(
                "Dialogue: 6,0:00:00.00,0:00:00.80,PaperLeft,,0,0,0,,"
                "{\\move(0,0,-920,0,0,800)\\p1}m 0 0 l 550 0 l 530 180 l 565 380 l 525 580 l 560 780 "
                "l 530 980 l 570 1180 l 535 1380 l 565 1580 l 530 1780 l 550 1920 l 0 1920{\\p0}"
            )
            events.append(
                "Dialogue: 6,0:00:00.00,0:00:00.80,PaperRight,,0,0,0,,"
                "{\\move(0,0,920,0,0,800)\\p1}m 1080 0 l 550 0 l 530 180 l 565 380 l 525 580 l 560 780 "
                "l 530 980 l 570 1180 l 535 1380 l 565 1580 l 530 1780 l 550 1920 l 1080 1920{\\p0}"
            )

        elif hook_type == HookType.GLITCH_REWIND:
            # Hook 5: Glitch Tape Rewind OSD and flickering scan lines (0.0s - 0.6s)
            events.append("Dialogue: 3,0:00:00.00,0:00:00.60,RewindOSD,,0,0,0,,{\\an7\\pos(80,120)}⏪ REW 00:00:02")
            events.append("Dialogue: 2,0:00:00.00,0:00:00.20,RewindScanLine,,0,0,0,,{\\p1\\pos(0,380)}m 0 0 l 1080 0 l 1080 6 l 0 6{\\p0}")
            events.append("Dialogue: 2,0:00:00.20,0:00:00.40,RewindScanLine,,0,0,0,,{\\p1\\pos(0,960)}m 0 0 l 1080 0 l 1080 8 l 0 8{\\p0}")
            events.append("Dialogue: 2,0:00:00.40,0:00:00.60,RewindScanLine,,0,0,0,,{\\p1\\pos(0,1420)}m 0 0 l 1080 0 l 1080 5 l 0 5{\\p0}")

        # -------------------------------------------------------------
        # Generate Subtitle Dialogues with Safe Zone & Active Word Pop
        # -------------------------------------------------------------
        total_clip_duration = clip_end - clip_start
        for s in segments:
            s_start = s["start"] - clip_start
            s_end = s["end"] - clip_start

            if s_end <= 0 or s_start >= total_clip_duration:
                continue

            s_start = max(0.0, s_start)
            s_end = min(total_clip_duration, s_end)

            words = s.get("words", [])
            text = s.get("text", "").strip()

            if words:
                # 1. Word-level timestamps provided (Whisper)
                # Chunk into 3-4 words per group
                chunk_size = 3
                for i in range(0, len(words), chunk_size):
                    chunk = words[i:i + chunk_size]
                    c_start = max(0.0, chunk[0]["start"] - clip_start)
                    c_end = min(total_clip_duration, chunk[-1]["end"] - clip_start)
                    if c_end <= c_start:
                        continue

                    # Active word highlight karaoke style
                    for cur_idx, cur_word in enumerate(chunk):
                        w_start = max(c_start, cur_word["start"] - clip_start)
                        w_end = min(c_end, cur_word["end"] - clip_start)
                        if w_end <= w_start:
                            continue

                        w_start_str = self._format_ass_time(w_start)
                        w_end_str = self._format_ass_time(w_end)

                        formatted_parts = []
                        for idx, w in enumerate(chunk):
                            clean_word = w["word"].strip().upper()
                            if idx == cur_idx:
                                # Neon Yellow &H0000E6FF + 108% scale pop
                                formatted_parts.append(
                                    f"{{\\c&H0000E6FF\\fscx108\\fscy108}}{clean_word}{{\\c&H00FFFFFF\\fscx100\\fscy100}}"
                                )
                            else:
                                formatted_parts.append(clean_word)

                        line_text = " ".join(formatted_parts)
                        events.append(f"Dialogue: 0,{w_start_str},{w_end_str},Hormozi,,0,0,0,,{{\\b1}}{line_text}")

            else:
                # 2. Segment-level timestamps (YouTube API)
                # Split text into 3-4 words chunks and interpolate timing evenly
                words_in_text = text.split()
                if not words_in_text:
                    continue

                chunk_size = 3
                chunks = [words_in_text[i:i + chunk_size] for i in range(0, len(words_in_text), chunk_size)]
                num_chunks = len(chunks)
                seg_duration = s_end - s_start

                for chunk_idx, chunk in enumerate(chunks):
                    sub_start = s_start + (chunk_idx / num_chunks) * seg_duration
                    sub_end = s_start + ((chunk_idx + 1) / num_chunks) * seg_duration

                    # Further sub-divide chunk duration per word for active pop effect
                    num_words = len(chunk)
                    chunk_dur = sub_end - sub_start

                    for cur_idx, _ in enumerate(chunk):
                        w_start = sub_start + (cur_idx / num_words) * chunk_dur
                        w_end = sub_start + ((cur_idx + 1) / num_words) * chunk_dur

                        w_start_str = self._format_ass_time(w_start)
                        w_end_str = self._format_ass_time(w_end)

                        formatted_parts = []
                        for idx, w in enumerate(chunk):
                            clean_w = w.strip().upper()
                            if idx == cur_idx:
                                formatted_parts.append(
                                    f"{{\\c&H0000E6FF\\fscx108\\fscy108}}{clean_w}{{\\c&H00FFFFFF\\fscx100\\fscy100}}"
                                )
                            else:
                                formatted_parts.append(clean_w)

                        line_text = " ".join(formatted_parts)
                        events.append(f"Dialogue: 0,{w_start_str},{w_end_str},Hormozi,,0,0,0,,{{\\b1}}{line_text}")

        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(header + "\n".join(events) + "\n")

        logger.info(
            f"Generated ASS subtitle file for hook '{hook_type}': "
            f"{ass_path} ({len(events)} dialogue lines)"
        )
        return ass_path

    def _format_ass_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        centisecs = int((secs - int(secs)) * 100)
        return f"{hours:d}:{minutes:02d}:{int(secs):02d}.{centisecs:02d}"


renderer = Renderer()

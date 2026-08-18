import math
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ], text=True).strip()
    return float(out)


def concat_audio(parts: list[Path], out: Path) -> None:
    listing = out.parent / "audio_concat.txt"
    listing.write_text("\n".join(f"file '{p.resolve()}'" for p in parts), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:a", "pcm_s16le", str(out)])


def _sentences(text: str) -> list[str]:
    items = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return items or [text.strip()]


def make_srt(text: str, total_seconds: float, out: Path) -> None:
    items = _sentences(text)
    weights = [max(1, len(x.split())) for x in items]
    total_w = sum(weights)
    cursor = 0.0

    def ts(sec: float) -> str:
        ms = int(round(sec * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, (sentence, w) in enumerate(zip(items, weights), start=1):
        seg = total_seconds * (w / total_w)
        end = min(total_seconds, cursor + seg)
        blocks.append(f"{i}\n{ts(cursor)} --> {ts(end)}\n{sentence}\n")
        cursor = end
    out.write_text("\n".join(blocks), encoding="utf-8")


def render_video(clips: list[Path], voice: Path, narration: str, out: Path, portrait: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "segments"
    work.mkdir(exist_ok=True)
    voice_dur = duration(voice)

    width, height = (720, 1280) if portrait else (1280, 720)
    segment_len = max(2.5, min(6.0, voice_dur / max(1, len(clips))))
    needed = max(1, math.ceil(voice_dur / segment_len))
    sequence = [clips[i % len(clips)] for i in range(needed)]

    seg_paths = []
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},setsar=1,fps=30,format=yuv420p"
    )
    for i, clip in enumerate(sequence):
        seg = work / f"seg_{i:03d}.mp4"
        run([
            "ffmpeg", "-y", "-stream_loop", "-1", "-i", str(clip),
            "-t", f"{segment_len:.3f}", "-an", "-vf", vf,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", str(seg)
        ])
        seg_paths.append(seg)

    concat_list = work / "video_concat.txt"
    concat_list.write_text("\n".join(f"file '{p.resolve()}'" for p in seg_paths), encoding="utf-8")
    base = work / "base.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(base)])

    srt = out.parent / "captions.srt"
    make_srt(narration, voice_dur, srt)
    font_size = 36 if portrait else 24
    margin_v = 110 if portrait else 55
    style = f"FontName=DejaVu Sans,FontSize={font_size},PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV={margin_v}"
    escaped = str(srt.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    run([
        "ffmpeg", "-y", "-i", str(base), "-i", str(voice),
        "-t", f"{voice_dur:.3f}",
        "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)
    ])


def make_thumbnail(video: Path, title: str, out: Path) -> None:
    frame = out.parent / "thumb_frame.jpg"
    run(["ffmpeg", "-y", "-ss", "00:00:01", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(frame)])
    img = ImageOps.fit(Image.open(frame).convert("RGB"), (1280, 720), method=Image.Resampling.LANCZOS)
    bg = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, 1280, 720), fill=(0, 0, 0, 80))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 70)

    words = title.upper().split()
    lines, line = [], []
    for word in words:
        test = " ".join(line + [word])
        if draw.textbbox((0, 0), test, font=font)[2] > 1080 and line:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    lines = lines[:3]
    block_h = len(lines) * 88
    y = (720 - block_h) // 2
    for text in lines:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=4)
        tw = bbox[2] - bbox[0]
        x = (1280 - tw) // 2
        draw.text((x, y), text, font=font, fill="white", stroke_width=5, stroke_fill="black")
        y += 88
    Image.alpha_composite(bg.convert("RGBA"), overlay).convert("RGB").save(out, quality=92)

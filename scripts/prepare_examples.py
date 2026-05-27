"""
Prepare a git-tracked bundle of matching example frames.

This script extracts the *same* moment from:
- the original input video (optional)
- the exported novel-view render (`export/render.mp4`)
- the semantic overlay render (`semantic/overlay.mp4`)

It writes PNGs into a single git-tracked folder under `examples/`.

Notes
-----
- Requires `ffmpeg` available on PATH.
- Videos are large and are gitignored; this script only writes small PNGs.
- Use `--time` to pick a timestamp in seconds. If you only know a frame index,
  use `--fps` to convert it: time = frame / fps.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def _probe_duration_seconds(video: Path) -> float:
    """
    Return video duration in seconds via ffprobe.

    We use this to avoid extracting beyond EOF, which can result in missing frames.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    s = result.stdout.strip()
    return float(s) if s else 0.0


def _extract_frame(video: Path, out_png: Path, *, time_s: float) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i is faster (seeks), good enough for visual examples.
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{time_s:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(out_png),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract matching example frames from export + semantic videos."
    )
    parser.add_argument(
        "--progress",
        type=float,
        default=None,
        help=(
            "Extract by normalised progress (0..1) within each video. "
            "This is often the easiest way to align short export/overlay videos "
            "with a long input video."
        ),
    )
    parser.add_argument(
        "--time",
        type=float,
        default=1.0,
        help="Timestamp in seconds to extract (default: 1.0).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional fps for converting a frame index to seconds.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Optional frame index to extract (requires --fps).",
    )
    parser.add_argument(
        "--input-video",
        type=str,
        default="input/video.mp4",
        help="Original input video path (default: input/video.mp4).",
    )
    parser.add_argument(
        "--render-video",
        type=str,
        default="export/render.mp4",
        help="Exported render video path (default: export/render.mp4).",
    )
    parser.add_argument(
        "--overlay-video",
        type=str,
        default="semantic/overlay.mp4",
        help="Semantic overlay video path (default: semantic/overlay.mp4).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="examples/frames",
        help="Output directory (default: examples/frames).",
    )
    parser.add_argument(
        "--skip-input",
        action="store_true",
        help="Do not extract a frame from the input video.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    render_video = Path(args.render_video)
    overlay_video = Path(args.overlay_video)
    input_video = Path(args.input_video)

    if not render_video.exists():
        raise SystemExit(f"Missing render video: {render_video}")
    if not overlay_video.exists():
        raise SystemExit(f"Missing overlay video: {overlay_video}")
    if not args.skip_input and not input_video.exists():
        raise SystemExit(f"Missing input video: {input_video} (or pass --skip-input)")

    durations = {
        "input": _probe_duration_seconds(input_video) if not args.skip_input else 0.0,
        "render": _probe_duration_seconds(render_video),
        "overlay": _probe_duration_seconds(overlay_video),
    }

    if args.progress is not None:
        if not (0.0 <= args.progress <= 1.0):
            raise SystemExit("--progress must be between 0 and 1.")
        times = {
            k: (v * args.progress if v > 0 else 0.0) for k, v in durations.items()
        }
        selection = f"progress={args.progress:.3f}"
    else:
        if args.frame is not None:
            if args.fps is None or args.fps <= 0:
                raise SystemExit("--frame requires a positive --fps.")
            base_time = args.frame / args.fps
            selection = f"frame={args.frame} @ fps={args.fps}"
        else:
            base_time = args.time
            selection = f"time_s={base_time:.3f}"

        # Clamp extraction time per video to stay within bounds.
        times = {}
        for k, dur in durations.items():
            if dur <= 0:
                times[k] = base_time
            else:
                times[k] = min(base_time, max(0.0, dur - 1e-3))

    if not args.skip_input:
        _extract_frame(input_video, out_dir / "input_frame.png", time_s=times["input"])
    _extract_frame(render_video, out_dir / "render_frame.png", time_s=times["render"])
    _extract_frame(
        overlay_video,
        out_dir / "semantic_overlay_frame.png",
        time_s=times["overlay"],
    )

    meta = out_dir / "README.txt"
    meta.write_text(
        "\n".join(
            [
                "Extracted example frames",
                f"- selection: {selection}",
                f"- durations_s: input={durations['input']:.3f} render={durations['render']:.3f} overlay={durations['overlay']:.3f}",
                f"- extract_s: input={times['input']:.3f} render={times['render']:.3f} overlay={times['overlay']:.3f}",
                f"- input_video: {input_video}",
                f"- render_video: {render_video}",
                f"- overlay_video: {overlay_video}",
                "",
                "Files:",
                "- input_frame.png (optional)",
                "- render_frame.png",
                "- semantic_overlay_frame.png",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[examples] Wrote frames to: {out_dir}")


if __name__ == "__main__":
    main()


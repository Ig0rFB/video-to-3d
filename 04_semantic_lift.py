"""
Optional semantic lifting: Grounding DINO + SAM2 → 3D Gaussian labels.

Implement after the core pipeline produces a valid PLY export.
Requires: groundingdino-py, sam2, and checkpoints/sam2.1_hiera_large.pt
"""

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from device import get_device

TARGET_CLASSES = [
    "table",
    "chair",
    "sofa",
    "shelf",
    "door",
    "window",
    "floor",
    "wall",
    "ceiling",
    "lamp",
]


@dataclass(frozen=True)
class _Colours:
    # BGR for OpenCV
    table: tuple[int, int, int] = (0, 153, 255)
    chair: tuple[int, int, int] = (0, 255, 153)
    sofa: tuple[int, int, int] = (255, 153, 0)
    shelf: tuple[int, int, int] = (153, 0, 255)
    door: tuple[int, int, int] = (0, 0, 255)
    window: tuple[int, int, int] = (255, 0, 0)
    floor: tuple[int, int, int] = (64, 64, 64)
    wall: tuple[int, int, int] = (128, 128, 128)
    ceiling: tuple[int, int, int] = (200, 200, 200)
    lamp: tuple[int, int, int] = (0, 255, 255)


def _class_colour(name: str) -> tuple[int, int, int]:
    c = _Colours()
    if hasattr(c, name):
        return getattr(c, name)
    return (255, 255, 255)


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(path), url], check=True)


def _iter_frames(frames_dir: Path, *, stride: int, max_frames: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in exts])
    if stride > 1:
        files = files[::stride]
    if max_frames is not None:
        files = files[:max_frames]
    return files


def _boxes_cxcywh_to_xyxy_px(boxes, w: int, h: int):
    import numpy as np

    b = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else boxes
    b = b.astype("float32")
    b[:, 0] *= w
    b[:, 2] *= w
    b[:, 1] *= h
    b[:, 3] *= h
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x0 = cx - bw / 2
    y0 = cy - bh / 2
    x1 = cx + bw / 2
    y1 = cy + bh / 2
    xyxy = np.stack([x0, y0, x1, y1], axis=1)
    xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w - 1)
    xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h - 1)
    return xyxy


def _match_class(phrase: str, classes: list[str]) -> str | None:
    p = phrase.lower().strip()
    for c in classes:
        if c in p:
            return c
    return None


def lift_semantics(
    checkpoint_dir: str,
    frames_dir: str,
    output_dir: str,
    *,
    stride: int = 1,
    max_frames: int | None = None,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    overlay_alpha: float = 0.45,
    write_video: bool = True,
) -> None:
    """
    Per-frame semantic overlays:
    1. Grounding DINO on each frame → boxes + phrases
    2. SAM2 on each box → instance mask
    3. Blend coloured masks on top of the RGB frame and save overlay PNGs
    """
    device = get_device()
    if device != "cuda":
        raise SystemExit("[semantic] This script requires CUDA.")

    try:
        import cv2  # type: ignore
        import numpy as np
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[semantic] Missing runtime deps: {exc}") from exc

    try:
        from patch_groundingdino import patch_groundingdino_bertwarper

        patch_groundingdino_bertwarper()
        from groundingdino.util.inference import load_model, predict  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "[semantic] groundingdino-py is not installed (or patch failed). Install with:\n"
            "  uv add --prerelease=allow groundingdino-py\n"
            "If you see BertModel/get_head_mask errors, also run:\n"
            "  uv run --no-sync python patch_groundingdino.py\n"
            "  uv pip install --python .venv/bin/python 'transformers>=4.35,<5'\n"
        ) from exc

    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "[semantic] SAM2 is not installed. Install it with:\n"
            "  uv add \"git+https://github.com/facebookresearch/sam2.git\"\n"
        ) from exc

    repo_root = Path(__file__).resolve().parent
    frames_path = Path(frames_dir)
    out_root = Path(output_dir)
    out_overlays = out_root / "overlays"
    out_overlays.mkdir(parents=True, exist_ok=True)

    # GroundingDINO config comes from the installed package; weights downloaded to checkpoints/.
    import groundingdino  # type: ignore

    gdino_pkg = Path(groundingdino.__file__).resolve().parent
    gdino_cfg = gdino_pkg / "config" / "GroundingDINO_SwinT_OGC.py"
    gdino_weights = repo_root / "checkpoints" / "groundingdino_swint_ogc.pth"
    _download_if_missing(
        "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        gdino_weights,
    )
    if not gdino_cfg.exists():
        raise SystemExit(f"[semantic] GroundingDINO config not found: {gdino_cfg}")

    print(f"[semantic] Loading GroundingDINO: {gdino_cfg}")
    dino = load_model(str(gdino_cfg), str(gdino_weights))

    # SAM2 config is shipped in the installed package (editable or package-data).
    import sam2  # type: ignore

    sam2_pkg = Path(sam2.__file__).resolve().parent
    # Try to find configs folder by walking up a bit.
    candidates = [
        sam2_pkg / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
        sam2_pkg.parent / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
        repo_root / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml",
    ]
    sam2_cfg = next((c for c in candidates if c.exists()), None)
    if sam2_cfg is None:
        raise SystemExit(
            "[semantic] SAM2 config file not found. Expected one of:\n"
            + "\n".join(f"  - {c}" for c in candidates)
        )

    sam2_ckpt = repo_root / "checkpoints" / "sam2.1_hiera_large.pt"
    if not sam2_ckpt.exists():
        raise SystemExit(
            f"[semantic] SAM2 checkpoint missing: {sam2_ckpt}\n"
            "Download it (example):\n"
            "  mkdir -p checkpoints\n"
            "  curl -L -o checkpoints/sam2.1_hiera_large.pt \\\n"
            "    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt\n"
        )

    print(f"[semantic] Loading SAM2: {sam2_cfg}")
    sam2_model = build_sam2(str(sam2_cfg), str(sam2_ckpt), device="cuda")
    sam2_pred = SAM2ImagePredictor(sam2_model)

    files = _iter_frames(frames_path, stride=stride, max_frames=max_frames)
    if not files:
        raise SystemExit(f"[semantic] No frames found under: {frames_path}")

    caption = " . ".join(TARGET_CLASSES) + " ."
    print(f"[semantic] Frames: {len(files)}  stride={stride}  output={out_overlays}")

    # SAM2 recommends bfloat16 autocast on CUDA for speed.
    torch = __import__("torch")
    ctx = torch.autocast("cuda", dtype=torch.bfloat16)

    with torch.inference_mode(), ctx:
        for idx, path in enumerate(files):
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # GroundingDINO expects a torch tensor image; use util loader via cv2 -> numpy.
            image_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            boxes, logits, phrases = predict(
                model=dino,
                image=image_t,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device="cuda",
            )
            if len(phrases) == 0:
                out_path = out_overlays / f"{path.stem}_overlay.png"
                cv2.imwrite(str(out_path), bgr)
                continue

            xyxy = _boxes_cxcywh_to_xyxy_px(boxes, w=w, h=h)

            sam2_pred.set_image(rgb)

            overlay = bgr.copy()
            for j, phrase in enumerate(phrases):
                cls = _match_class(phrase, TARGET_CLASSES)
                if cls is None:
                    continue
                x0, y0, x1, y1 = xyxy[j]
                box = np.array([x0, y0, x1, y1], dtype=np.float32)
                masks, ious, _ = sam2_pred.predict(
                    box=box,
                    multimask_output=False,
                    return_logits=False,
                    normalize_coords=True,
                )
                if masks is None or len(masks) == 0:
                    continue
                mask = masks[0].astype(bool)
                colour = _class_colour(cls)

                # Blend overlay
                overlay[mask] = (
                    overlay[mask].astype(np.float32) * (1.0 - overlay_alpha)
                    + np.array(colour, dtype=np.float32) * overlay_alpha
                ).astype(np.uint8)

                # Draw box + label
                cv2.rectangle(
                    overlay,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    colour,
                    2,
                )
                cv2.putText(
                    overlay,
                    cls,
                    (int(x0), max(0, int(y0) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                    cv2.LINE_AA,
                )

            out_path = out_overlays / f"{path.stem}_overlay.png"
            cv2.imwrite(str(out_path), overlay)
            if (idx + 1) % 25 == 0:
                print(f"[semantic] {idx+1}/{len(files)}")

    if write_video:
        video_path = out_root / "overlay.mp4"
        # ffmpeg expects sequential frames; write a concat list for robustness.
        list_path = out_root / "frames.txt"
        with list_path.open("w", encoding="utf-8") as f:
            for p in sorted(out_overlays.glob("*_overlay.png")):
                f.write(f"file '{p.resolve()}'\n")
                f.write("duration 0.04\n")  # ~25 fps
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-vsync",
                "vfr",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ],
            check=True,
        )
        print(f"[semantic] Wrote {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--frames_dir", default="frames/")
    parser.add_argument("--output_dir", default="semantic/")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    parser.add_argument("--no_video", action="store_true")
    args = parser.parse_args()
    lift_semantics(
        args.checkpoint_dir,
        args.frames_dir,
        args.output_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        overlay_alpha=args.overlay_alpha,
        write_video=not args.no_video,
    )

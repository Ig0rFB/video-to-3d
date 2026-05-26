"""
Optional semantic lifting: Grounding DINO + SAM2 → per-frame 2D overlays.

Run after the core pipeline produces a valid splatfacto export. Requires
groundingdino-py, sam2, and checkpoints/sam2.1_hiera_large.pt.

``--checkpoint-dir`` selects the trained splatfacto run (for workflow consistency).
Per-Gaussian 3D labels are not implemented yet — output is 2D overlay PNGs/video only.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from checkpoint_paths import LATEST_CHECKPOINT, resolve_checkpoint_dir
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

# BGR colours for OpenCV overlays
CLASS_COLOURS: dict[str, tuple[int, int, int]] = {
    "table": (0, 153, 255),
    "chair": (0, 255, 153),
    "sofa": (255, 153, 0),
    "shelf": (153, 0, 255),
    "door": (0, 0, 255),
    "window": (255, 0, 0),
    "floor": (64, 64, 64),
    "wall": (128, 128, 128),
    "ceiling": (200, 200, 200),
    "lamp": (0, 255, 255),
}


def _class_colour(name: str) -> tuple[int, int, int]:
    return CLASS_COLOURS.get(name, (255, 255, 255))


def _download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-L", "-o", str(path), url], check=True)


def _iter_frames(frames_dir: Path, *, stride: int, max_frames: int | None) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    if stride > 1:
        files = files[::stride]
    if max_frames is not None:
        files = files[:max_frames]
    return files


def _boxes_cxcywh_to_xyxy_px(boxes: Any, w: int, h: int) -> Any:
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


def _assert_cuda() -> None:
    if get_device() != "cuda":
        raise SystemExit("[semantic] This script requires CUDA.")


def _resolve_splatfacto_run(checkpoint_dir: str) -> Path:
    """Validate the splatfacto training run (3D label export is not implemented yet)."""
    run_dir = resolve_checkpoint_dir(checkpoint_dir)
    print(
        f"[semantic] Splatfacto run: {run_dir}\n"
        "[semantic] Producing 2D frame overlays only (per-Gaussian 3D labels: not implemented)."
    )
    return run_dir


def _load_grounding_dino(repo_root: Path) -> Any:
    try:
        from patch_groundingdino import patch_groundingdino_bertwarper

        patch_groundingdino_bertwarper()
        from groundingdino.util.inference import load_model  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "[semantic] groundingdino-py is not installed (or patch failed). Install with:\n"
            "  uv add --prerelease=allow groundingdino-py\n"
            "If you see BertModel/get_head_mask errors, also run:\n"
            "  uv run --no-sync python patch_groundingdino.py\n"
            "  uv pip install --python .venv/bin/python 'transformers>=4.35,<5'\n"
        ) from exc

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
    return load_model(str(gdino_cfg), str(gdino_weights))


def _load_sam2_predictor(repo_root: Path) -> Any:
    try:
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "[semantic] SAM2 is not installed. Install it with:\n"
            '  uv add "git+https://github.com/facebookresearch/sam2.git"\n'
        ) from exc

    import sam2  # type: ignore

    sam2_pkg = Path(sam2.__file__).resolve().parent
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
    return SAM2ImagePredictor(sam2_model)


def _process_frame(
    path: Path,
    out_overlays: Path,
    *,
    dino: Any,
    sam2_pred: Any,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    overlay_alpha: float,
    cv2: Any,
    np: Any,
    torch: Any,
) -> None:
    from groundingdino.util.inference import predict  # type: ignore

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out_path = out_overlays / f"{path.stem}_overlay.png"

    image_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    boxes, _logits, phrases = predict(
        model=dino,
        image=image_t,
        caption=caption,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device="cuda",
    )
    if len(phrases) == 0:
        cv2.imwrite(str(out_path), bgr)
        return

    xyxy = _boxes_cxcywh_to_xyxy_px(boxes, w=w, h=h)
    sam2_pred.set_image(rgb)
    overlay = bgr.copy()

    for j, phrase in enumerate(phrases):
        cls = _match_class(phrase, TARGET_CLASSES)
        if cls is None:
            continue
        x0, y0, x1, y1 = xyxy[j]
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        masks, _ious, _ = sam2_pred.predict(
            box=box,
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )
        if masks is None or len(masks) == 0:
            continue
        mask = masks[0].astype(bool)
        colour = _class_colour(cls)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - overlay_alpha)
            + np.array(colour, dtype=np.float32) * overlay_alpha
        ).astype(np.uint8)
        cv2.rectangle(overlay, (int(x0), int(y0)), (int(x1), int(y1)), colour, 2)
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

    cv2.imwrite(str(out_path), overlay)


def _write_overlay_video(out_root: Path, out_overlays: Path) -> None:
    video_path = out_root / "overlay.mp4"
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
    3. Blend coloured masks on the RGB frame and save overlay PNGs
    """
    _assert_cuda()
    _resolve_splatfacto_run(checkpoint_dir)

    try:
        import cv2  # type: ignore
        import numpy as np
        import torch
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[semantic] Missing runtime deps: {exc}") from exc

    repo_root = Path(__file__).resolve().parent
    frames_path = Path(frames_dir)
    out_root = Path(output_dir)
    out_overlays = out_root / "overlays"
    out_overlays.mkdir(parents=True, exist_ok=True)

    dino = _load_grounding_dino(repo_root)
    sam2_pred = _load_sam2_predictor(repo_root)

    files = _iter_frames(frames_path, stride=stride, max_frames=max_frames)
    if not files:
        raise SystemExit(f"[semantic] No frames found under: {frames_path}")

    caption = " . ".join(TARGET_CLASSES) + " ."
    print(f"[semantic] Frames: {len(files)}  stride={stride}  output={out_overlays}")

    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), ctx:
        for idx, path in enumerate(files):
            _process_frame(
                path,
                out_overlays,
                dino=dino,
                sam2_pred=sam2_pred,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                overlay_alpha=overlay_alpha,
                cv2=cv2,
                np=np,
                torch=torch,
            )
            if (idx + 1) % 25 == 0:
                print(f"[semantic] {idx + 1}/{len(files)}")

    if write_video:
        _write_overlay_video(out_root, out_overlays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "2D semantic overlays (Grounding DINO + SAM2). "
            "Validates --checkpoint-dir; 3D Gaussian labelling is not implemented yet."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=LATEST_CHECKPOINT,
        help=(
            f"Splatfacto run with config.yml (default: {LATEST_CHECKPOINT}). "
            "Used to align with the trained scene; output is 2D overlays only."
        ),
    )
    parser.add_argument("--frames-dir", default="frames/")
    parser.add_argument("--output-dir", default="semantic/")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--no-video", action="store_true")
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

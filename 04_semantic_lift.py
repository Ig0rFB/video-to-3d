"""
Optional semantic lifting: Grounding DINO + SAM2 → per-frame 2D overlays.

Run after the core pipeline produces a valid splatfacto export. Requires
groundingdino-py, sam2, and checkpoints/sam2.1_hiera_large.pt.

``--checkpoint-dir`` selects the trained splatfacto run (for workflow consistency).
Per-Gaussian 3D labels are not implemented yet — output is 2D overlay PNGs/video only.
"""

from __future__ import annotations

import argparse
import inspect
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


def _prepare_grounding_dino_image(rgb: Any) -> Any:
    """Resize + ImageNet normalise — required by GroundingDINO predict()."""
    from PIL import Image

    import groundingdino.datasets.transforms as T  # type: ignore

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(rgb)
    image_transformed, _ = transform(image_pil, None)
    return image_transformed


def _boxes_cxcywh_to_xyxy_px(boxes: Any, w: int, h: int) -> Any:
    """Map normalised cxcywh boxes to pixel xyxy (same as groundingdino annotate())."""
    from torchvision.ops import box_convert

    scale = boxes.new_tensor([w, h, w, h])
    boxes_scaled = boxes * scale
    xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
    return xyxy.detach().cpu().numpy()


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


SAM2_HIERA_L_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


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

    # build_sam2 uses Hydra compose() — pass a config name on pkg://sam2, not a filesystem path.
    sam2_pkg = Path(sam2.__file__).resolve().parent
    cfg_on_disk = sam2_pkg / "configs" / "sam2.1" / "sam2.1_hiera_l.yaml"
    if not cfg_on_disk.exists():
        raise SystemExit(
            "[semantic] SAM2 config missing from the installed package:\n"
            f"  - {cfg_on_disk}\n"
            "Reinstall SAM2 from the official repo:\n"
            '  uv add "git+https://github.com/facebookresearch/sam2.git"\n'
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

    print(f"[semantic] Loading SAM2: {SAM2_HIERA_L_CONFIG}")
    sam2_model = build_sam2(SAM2_HIERA_L_CONFIG, str(sam2_ckpt), device="cuda")
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
) -> tuple[int, int]:
    """Returns (detection_count, labelled_count) for progress stats."""
    from groundingdino.util.inference import predict  # type: ignore

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return 0, 0
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out_path = out_overlays / f"{path.stem}_overlay.png"

    image_t = _prepare_grounding_dino_image(rgb)
    predict_kwargs: dict[str, Any] = {
        "model": dino,
        "image": image_t,
        "caption": caption,
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "device": "cuda",
    }
    if "remove_combined" in inspect.signature(predict).parameters:
        predict_kwargs["remove_combined"] = True
    boxes, _logits, phrases = predict(**predict_kwargs)
    phrases = [p.strip() for p in phrases if p.strip()]
    if len(phrases) == 0:
        cv2.imwrite(str(out_path), bgr)
        return 0, 0

    xyxy = _boxes_cxcywh_to_xyxy_px(boxes, w=w, h=h)
    sam2_pred.set_image(rgb)
    overlay = bgr.copy()
    labelled = 0

    for j, phrase in enumerate(phrases):
        cls = _match_class(phrase, TARGET_CLASSES)
        label = cls if cls is not None else phrase
        colour = _class_colour(cls) if cls is not None else (180, 180, 180)
        x0, y0, x1, y1 = xyxy[j]
        x0i, y0i, x1i, y1i = int(x0), int(y0), int(x1), int(y1)
        cv2.rectangle(overlay, (x0i, y0i), (x1i, y1i), colour, 2)
        cv2.putText(
            overlay,
            label,
            (x0i, max(0, y0i - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
        if cls is None:
            continue
        box = np.array([x0, y0, x1, y1], dtype=np.float32)
        masks, _ious, _ = sam2_pred.predict(
            box=box,
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )
        if masks is None or len(masks) == 0:
            labelled += 1
            continue
        mask = masks[0].astype(bool)
        overlay[mask] = (
            overlay[mask].astype(np.float32) * (1.0 - overlay_alpha)
            + np.array(colour, dtype=np.float32) * overlay_alpha
        ).astype(np.uint8)
        labelled += 1

    cv2.imwrite(str(out_path), overlay)
    return len(phrases), labelled


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
    box_threshold: float = 0.28,
    text_threshold: float = 0.22,
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

    total_detections = 0
    total_labelled = 0
    ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), ctx:
        for idx, path in enumerate(files):
            n_det, n_lbl = _process_frame(
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
            total_detections += n_det
            total_labelled += n_lbl
            if (idx + 1) % 25 == 0:
                print(f"[semantic] {idx + 1}/{len(files)}")

    print(
        f"[semantic] Done: {len(files)} frames, "
        f"{total_detections} detections, {total_labelled} with target-class masks"
    )
    if total_detections == 0:
        print(
            "[semantic] No detections — try lower thresholds, e.g. "
            "--box-threshold 0.25 --text-threshold 0.2"
        )

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
    parser.add_argument("--box-threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.22)
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

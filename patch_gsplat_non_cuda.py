"""
Fall back to gsplat's pure-PyTorch rasterizer when the CUDA extension is unavailable.

On Apple Silicon, gsplat prints "No CUDA toolkit found" and sets _C = None.
Splatfacto / ns-viewer then fail in fully_fused_projection. This patch routes
rasterization() to _rasterization() when CUDA is missing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATCH_DONE = "# video3d: gsplat non-cuda fallback applied"

FALLBACK_MARKER = "from gsplat.cuda._backend import _C as _gsplat_cuda"

FALLBACK_BODY = """\
    from gsplat.cuda._backend import _C as _gsplat_cuda

    if _gsplat_cuda is None:
        return _rasterization(
            means,
            quats,
            scales,
            opacities,
            colors,
            viewmats,
            Ks,
            width,
            height,
            near_plane=near_plane,
            far_plane=far_plane,
            eps2d=eps2d,
            sh_degree=sh_degree,
            tile_size=tile_size,
            backgrounds=backgrounds,
            render_mode=render_mode,
            rasterize_mode=rasterize_mode,
            channel_chunk=channel_chunk,
        )

"""

TORCH_RASTER_MARKER = "_isect_tiles,\n        _isect_offset_encode,\n        _spherical_harmonics,"
TORCH_RASTER_PATCH = f"""\
    {PATCH_DONE}
    from gsplat.cuda._backend import _C as _gsplat_cuda
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
        _rasterize_to_pixels,
        _isect_tiles,
        _isect_offset_encode,
        _spherical_harmonics,
    )
    if _gsplat_cuda is None:
        isect_tiles = _isect_tiles
        isect_offset_encode = _isect_offset_encode
        spherical_harmonics = _spherical_harmonics

"""
TORCH_RASTER_OLD = """\
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
        _rasterize_to_pixels,
    )

"""

ISECT_CALL_MARKER = "    if _gsplat_cuda is None:\n        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(\n            means2d,"
ISECT_CALL_NEW = """\
    if _gsplat_cuda is None:
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
        )
    else:
        tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=False,
            n_cameras=C,
            camera_ids=camera_ids,
            gaussian_ids=gaussian_ids,
        )
"""

ISECT_CALL_OLD = """\
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
"""

SH_BLOCK_CORRECT = """\
        if _gsplat_cuda is None:
            colors = spherical_harmonics(sh_degree, dirs, shs)
            colors = colors * masks[..., None]
        else:
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
"""

SH_BLOCK_OLD = """\
        colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
"""

# Repair duplicated / half-applied SH blocks inside _rasterization only.
_SH_REPAIR = re.compile(
    r"(        shs = colors\n)"
    r"(?:        if _gsplat_cuda is None:.*?)"
    r"(    # Rasterize to pixels\n)",
    re.DOTALL,
)


def _validate_syntax(rendering: Path) -> None:
    compile(rendering.read_text(), str(rendering), "exec")


def _repair_rasterization_sh(rast_body: str) -> tuple[str, bool]:
    if SH_BLOCK_OLD in rast_body and SH_BLOCK_CORRECT not in rast_body:
        return rast_body.replace(SH_BLOCK_OLD, SH_BLOCK_CORRECT, 1), True
    match = _SH_REPAIR.search(rast_body)
    if match and SH_BLOCK_CORRECT not in match.group(0):
        fixed = match.group(1) + SH_BLOCK_CORRECT
        return rast_body[: match.start()] + fixed + rast_body[match.end() :], True
    return rast_body, False


def _is_patched(text: str) -> bool:
    if PATCH_DONE in text:
        return True
    if "def _rasterization" not in text:
        return False
    tail = text.split("def _rasterization", 1)[1]
    return (
        FALLBACK_MARKER in text
        and TORCH_RASTER_MARKER in tail
        and ISECT_CALL_MARKER in tail
        and SH_BLOCK_CORRECT in tail
    )


def patch_gsplat_rendering(site_packages: Path | None = None) -> Path:
    if site_packages is None:
        import gsplat

        rendering = Path(gsplat.__file__).parent / "rendering.py"
    else:
        rendering = site_packages / "gsplat" / "rendering.py"

    if not rendering.exists():
        raise FileNotFoundError(f"gsplat rendering.py not found: {rendering}")

    text = rendering.read_text()

    if _is_patched(text):
        head, tail = text.split("def _rasterization", 1)
        rast_body, repaired = _repair_rasterization_sh(tail)
        if repaired:
            rendering.write_text(head + "def _rasterization" + rast_body)
            print(f"[patch] Repaired gsplat SH block: {rendering}")
        else:
            print(f"[patch] gsplat CPU/MPS fallback already applied: {rendering}")
        _validate_syntax(rendering)
        return rendering

    changed = False

    if FALLBACK_MARKER not in text:
        needle = '    """\n    meta = {}'
        if needle not in text:
            raise RuntimeError(
                f"Could not find rasterization insertion point in {rendering}."
            )
        text = text.replace(needle, f'    """\n{FALLBACK_BODY}\n    meta = {{}}', 1)
        changed = True

    if "def _rasterization" not in text:
        raise RuntimeError(f"def _rasterization not found in {rendering}")

    head, tail = text.split("def _rasterization", 1)
    rast_body = tail

    if TORCH_RASTER_MARKER not in rast_body:
        if TORCH_RASTER_OLD not in rast_body:
            raise RuntimeError(
                f"Could not find _rasterization import block in {rendering}."
            )
        rast_body = rast_body.replace(TORCH_RASTER_OLD, TORCH_RASTER_PATCH, 1)
        changed = True
    elif PATCH_DONE not in rast_body:
        rast_body = rast_body.replace(
            "    from gsplat.cuda._backend import _C as _gsplat_cuda\n",
            f"    {PATCH_DONE}\n    from gsplat.cuda._backend import _C as _gsplat_cuda\n",
            1,
        )
        changed = True

    if ISECT_CALL_MARKER not in rast_body:
        if ISECT_CALL_OLD not in rast_body:
            raise RuntimeError(
                f"Could not find isect_tiles call in _rasterization in {rendering}."
            )
        rast_body = rast_body.replace(ISECT_CALL_OLD, ISECT_CALL_NEW, 1)
        changed = True

    rast_body, sh_changed = _repair_rasterization_sh(rast_body)
    changed = changed or sh_changed

    text = head + "def _rasterization" + rast_body
    if changed:
        rendering.write_text(text)
        print(f"[patch] Applied gsplat CPU/MPS rasterizer fallback: {rendering}")
    else:
        print(f"[patch] No changes needed: {rendering}")

    try:
        _validate_syntax(rendering)
    except SyntaxError as exc:
        raise RuntimeError(
            f"gsplat rendering.py syntax error after patch: {exc}. "
            "Run: ./scripts/reset_gsplat.sh"
        ) from exc

    return rendering


if __name__ == "__main__":
    try:
        patch_gsplat_rendering()
    except Exception as exc:
        print(f"[patch] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

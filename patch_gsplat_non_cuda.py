"""
Fall back to gsplat's pure-PyTorch rasterizer when the CUDA extension is unavailable.

On Apple Silicon, gsplat prints "No CUDA toolkit found" and sets _C = None.
Splatfacto / ns-viewer then fail in fully_fused_projection. This patch routes
rasterization() to _rasterization() (PyTorch path) when CUDA is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
TORCH_RASTER_PATCH = """\
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

# Unique to _rasterization() — do not patch the main rasterization() SH block.
SH_PATCH_MARKER = "# gsplat non-cuda SH (_rasterization only)"
SH_CALL_OLD = """\
        colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
"""
SH_CALL_NEW = """\
        if _gsplat_cuda is None:
            colors = spherical_harmonics(sh_degree, dirs, shs)
            colors = colors * masks[..., None]
        else:
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
"""


def _validate_syntax(rendering: Path) -> None:
    compile(rendering.read_text(), str(rendering), "exec")


def patch_gsplat_rendering(site_packages: Path | None = None) -> Path:
    if site_packages is None:
        import gsplat

        rendering = Path(gsplat.__file__).parent / "rendering.py"
    else:
        rendering = site_packages / "gsplat" / "rendering.py"

    if not rendering.exists():
        raise FileNotFoundError(f"gsplat rendering.py not found: {rendering}")

    text = rendering.read_text()
    changed = False

    if FALLBACK_MARKER not in text:
        needle = '    """\n    meta = {}'
        if needle not in text:
            raise RuntimeError(
                f"Could not find rasterization insertion point in {rendering}."
            )
        text = text.replace(needle, f'    """\n{FALLBACK_BODY}\n    meta = {{}}', 1)
        changed = True

    # Only patch inside _rasterization (after its def line).
    if "def _rasterization" in text:
        head, tail = text.split("def _rasterization", 1)
        rast_body = tail
        rast_changed = False

        if TORCH_RASTER_MARKER not in rast_body:
            if TORCH_RASTER_OLD not in rast_body:
                raise RuntimeError(
                    f"Could not find _rasterization import block in {rendering}."
                )
            rast_body = rast_body.replace(TORCH_RASTER_OLD, TORCH_RASTER_PATCH, 1)
            rast_changed = True

        if ISECT_CALL_MARKER not in rast_body:
            if ISECT_CALL_OLD not in rast_body:
                raise RuntimeError(
                    f"Could not find isect_tiles call in _rasterization in {rendering}."
                )
            rast_body = rast_body.replace(ISECT_CALL_OLD, ISECT_CALL_NEW, 1)
            rast_changed = True

        if SH_PATCH_MARKER not in rast_body and SH_CALL_OLD in rast_body:
            rast_body = rast_body.replace(SH_CALL_OLD, SH_CALL_NEW, 1)
            rast_changed = True

        if rast_changed:
            text = head + "def _rasterization" + rast_body
            changed = True

    if not changed:
        print(f"[patch] gsplat CPU/MPS fallback already applied: {rendering}")
    else:
        rendering.write_text(text)
        print(f"[patch] Applied gsplat CPU/MPS rasterizer fallback: {rendering}")

    try:
        _validate_syntax(rendering)
    except SyntaxError as exc:
        raise RuntimeError(
            f"gsplat rendering.py has a syntax error after patching: {exc}. "
            "Reinstall gsplat: uv pip reinstall gsplat"
        ) from exc

    return rendering


if __name__ == "__main__":
    try:
        patch_gsplat_rendering()
    except Exception as exc:
        print(f"[patch] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

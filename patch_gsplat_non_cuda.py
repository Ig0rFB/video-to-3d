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

# gsplat 1.4 _rasterization(): use torch_impl isect helpers when CUDA ext is missing
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

SH_CALL_MARKER = "colors = colors * masks[..., None]"
SH_CALL_OLD = """\
        colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
"""
SH_CALL_NEW = """\
        if _gsplat_cuda is None:
            colors = spherical_harmonics(sh_degree, dirs, shs)
            colors = colors * masks[..., None]
        else:
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
"""


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

    if TORCH_RASTER_MARKER not in text:
        if TORCH_RASTER_OLD not in text:
            raise RuntimeError(
                f"Could not find _rasterization import block in {rendering}."
            )
        text = text.replace(TORCH_RASTER_OLD, TORCH_RASTER_PATCH, 1)
        changed = True

    if ISECT_CALL_MARKER not in text:
        if ISECT_CALL_OLD not in text:
            raise RuntimeError(
                f"Could not find isect_tiles call in _rasterization in {rendering}."
            )
        text = text.replace(ISECT_CALL_OLD, ISECT_CALL_NEW, 1)
        changed = True

    if SH_CALL_MARKER not in text:
        if SH_CALL_OLD not in text:
            raise RuntimeError(
                f"Could not find spherical_harmonics call in _rasterization in {rendering}."
            )
        text = text.replace(SH_CALL_OLD, SH_CALL_NEW, 1)
        changed = True

    if not changed:
        print(f"[patch] gsplat CPU/MPS fallback already applied: {rendering}")
        return rendering

    rendering.write_text(text)
    print(f"[patch] Applied gsplat CPU/MPS rasterizer fallback: {rendering}")
    return rendering


if __name__ == "__main__":
    try:
        patch_gsplat_rendering()
    except Exception as exc:
        print(f"[patch] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

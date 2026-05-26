# Design notes — video3d

Pipeline: **ffmpeg → COLMAP SfM → nerfstudio splatfacto (3DGS / gsplat) → PLY + spiral render**, with optional semantic lifting.

## 1. Approach overview

| Stage | Tool | Role |
|-------|------|------|
| Frames | ffmpeg | Extract JPEGs at 2 fps (3–4 fps for short clips) |
| Poses | COLMAP via `pycolmap` | Sparse reconstruction; single shared camera model |
| Training | nerfstudio `splatfacto` + gsplat | 3D Gaussian scene representation |
| Packaging | `ns-process-data` | Converts COLMAP + images to nerfstudio format |
| Export | `ns-export` / `ns-render interpolate` | PLY + video along training camera path (not `spiral` — incompatible with splatfacto datamanager) |

**MuSHRoom path:** pre-computed COLMAP and RGB from Zenodo skip steps 01–02; training uses `--mushroom` and `--skip-colmap` in `03_train_gaussian.py`.

## 2. Why 3D Gaussian Splatting over NeRF

- **Explicit geometry** — Gaussians are discrete, inspectable primitives (PLY export).
- **Training time** — ~20 minutes on a CUDA RTX 3090/4090 vs hours for high-quality NeRF.
- **Semantics** — per-Gaussian labels are a natural post-process (optional step 04).
- **Stack** — `nerfstudio` + `gsplat` support CUDA; gsplat’s fused CUDA rasteriser is required for practical splatfacto training (not available on Apple Silicon alone).

Trade-off: less smooth in unseen regions than implicit NeRF volumes.

## 3. Why COLMAP over DUSt3R / MASt3R

- COLMAP is the default for **textured indoor** phone video and for MuSHRoom baselines.
- `02_run_colmap.py` uses **SINGLE** camera mode, exhaustive matching for small sets, and keeps the largest registered model.
- DUSt3R remains a documented fallback if registration falls below ~30% of frames (see `AGENT_BRIEF.md`).

## 4. Semantic lifting (optional)

Grounding DINO + SAM2 → 2D masks → majority vote on projected Gaussian centres. Does not change trained geometry. See `04_semantic_lift.py` and README optional section.

## 5. Hardware and environment

| Host | Frames + COLMAP | splatfacto training |
|------|-----------------|---------------------|
| Apple Silicon (MPS) | Yes | No (needs CUDA Linux/cloud) |
| Linux + NVIDIA GPU | Yes | Yes |

- **`device.py`** — selects `cuda` → `mps` → `cpu`.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** — set in `pipeline.py` before torch import.
- **PyTorch on Linux CUDA** — pinned to **2.5.1 + cu124** in `pyproject.toml` (avoids CUDA 13 wheels that mismatch common cloud drivers).
- **`uv run --no-sync`** — pipeline commands avoid re-resolving torch after sync.
- **`scripts/setup_cloud.sh`** — one-shot cloud instance setup (apt deps, sync, patch, CUDA verify).
- **`patch_nerfstudio_mps.py`** — fixes splatfacto init on MPS (harmless on CUDA).
- **`env_utils.py`** — COLMAP binary check, `.venv/bin` CLI resolution, CUDA preflight.

## 6. MuSHRoom data sources

| Content | Zenodo | Notes |
|---------|--------|--------|
| COLMAP `sparse/` (all rooms) | [13986996](https://zenodo.org/records/13986996) | Phase 1 of `download_mushroom.py` |
| RGB `images/` per room | [10230733](https://zenodo.org/records/10230733) | `<room>_iphone.tar.gz` |
| SDF only (no `images/`) | [10151161](https://zenodo.org/records/10151161) | Use `prepare_mushroom_images.py` only if you already have `*_iphone_our` |

## 7. Future work

- LangSplat / Feature 3DGS for language queries.
- Mesh from Gaussian opacity field + marching cubes.
- DUSt3R poses → 3DGS hybrid benchmark vs COLMAP on the same room.
- Depth prior (e.g. Depth Anything v2) for sparse registrations.

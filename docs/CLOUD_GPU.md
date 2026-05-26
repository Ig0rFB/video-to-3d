# Cloud GPU setup (vast.ai, Lambda, etc.)

## One-shot setup

From the repository root on a **fresh Linux instance with an NVIDIA GPU**:

```bash
git clone https://github.com/Ig0rFB/video-to-3d.git
cd video-to-3d
chmod +x scripts/setup_cloud.sh
./scripts/setup_cloud.sh
```

This script:

1. Installs **ffmpeg** and **COLMAP** (`apt-get` when available).
2. Runs `uv sync --prerelease=allow` (torch **2.5.1+cu124** on Linux per `pyproject.toml`).
3. Runs `scripts/ensure_env.py --fix-cuda --require-cuda`.

## Run training

Always use **`uv run --no-sync`** so uv does not re-resolve PyTorch wheels:

```bash
uv run --no-sync python 03_train_gaussian.py --mushroom input/MuSHRoom/room_datasets/coffee_room
```

Equivalent: `.venv/bin/python 03_train_gaussian.py ...` (after setup).

## Verify environment

```bash
nvidia-smi
uv run --no-sync python scripts/ensure_env.py --require-cuda
uv run --no-sync python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.5.1+cu124` and `True`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Could not find COLMAP` | `sudo apt-get install -y colmap` |
| `No such file or directory: ns-process-data` | `uv sync --prerelease=allow`; use `uv run --no-sync` or `.venv/bin/python` |
| `undefined symbol: ncclCommResume` | Mixed torch wheels: `uv run --no-sync python scripts/ensure_env.py --fix-cuda` |
| `torch.cuda.is_available() False` after sync | Run `./scripts/setup_cloud.sh` or `--fix-cuda` above |
| Plain `uv run` re-breaks CUDA | Use **`--no-sync`** for all pipeline commands |

## Existing data on the instance

If `input/MuSHRoom/...` is already present, skip download and train directly after `setup_cloud.sh`.

Pull latest docs/scripts: `git pull` in the repo directory.

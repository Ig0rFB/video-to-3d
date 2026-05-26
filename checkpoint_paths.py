"""Locate trained nerfstudio splatfacto runs (directories containing config.yml)."""

from __future__ import annotations

from pathlib import Path

CHECKPOINT_CONFIG = "config.yml"
LATEST_CHECKPOINT = "latest"

# ns-train --output-dir outputs/ writes outputs/splatfacto/<timestamp>/.
# Some docs refer to outputs/nerfstudio_data/splatfacto/ — include for compatibility.
SPLATFACTO_SEARCH_ROOTS = (
    Path("outputs") / "splatfacto",
    Path("outputs") / "nerfstudio_data" / "splatfacto",
)


def is_splatfacto_run(path: Path) -> bool:
    """True when path is a splatfacto run directory (contains config.yml)."""
    return (path / CHECKPOINT_CONFIG).is_file()


def iter_splatfacto_runs(
    roots: tuple[Path, ...] = SPLATFACTO_SEARCH_ROOTS,
) -> list[Path]:
    """Return all splatfacto run directories under the standard search roots."""
    runs: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            if child.is_dir() and is_splatfacto_run(child):
                runs.append(child.resolve())
    return runs


def find_latest_splatfacto_run(
    roots: tuple[Path, ...] = SPLATFACTO_SEARCH_ROOTS,
) -> Path | None:
    """Return the most recently modified splatfacto run, or None if none exist."""
    runs = iter_splatfacto_runs(roots)
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def resolve_checkpoint_dir(
    checkpoint: str | None,
    *,
    latest_token: str = LATEST_CHECKPOINT,
) -> Path:
    """
    Resolve a checkpoint path for export.

    Accepts an explicit run directory or the token ``latest`` (default) to pick
    the newest run under outputs/splatfacto/ (and legacy nerfstudio_data paths).
    """
    if checkpoint is None or checkpoint == latest_token:
        found = find_latest_splatfacto_run()
        if found is None:
            searched = ", ".join(str(r) for r in SPLATFACTO_SEARCH_ROOTS)
            raise SystemExit(
                f"No splatfacto checkpoint found (no {CHECKPOINT_CONFIG} under {searched}).\n"
                "Train first with 03_train_gaussian.py, or pass --checkpoint-dir explicitly."
            )
        print(f"[checkpoint] Using latest run: {found}")
        return found

    path = Path(checkpoint).expanduser().resolve()
    if not is_splatfacto_run(path):
        raise SystemExit(
            f"Not a splatfacto run directory (missing {CHECKPOINT_CONFIG}): {path}\n"
            f"Pass --checkpoint-dir {LATEST_CHECKPOINT} to use the newest run under outputs/."
        )
    return path

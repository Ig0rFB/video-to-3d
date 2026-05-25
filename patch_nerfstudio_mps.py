"""
Apply upstream nerfstudio MPS fix to the installed splatfacto model.

Replaces hardcoded .cuda() with .to(self.device) so splatfacto initialises on
Apple Silicon (see nerfstudio commit ead21ba). Safe to run repeatedly.
"""

from __future__ import annotations

import sys
from pathlib import Path


# populate_modules() runs before device_indicator_param exists; use means.device there.
PATCHES = (
    (
        ".float().cuda()",
        ".float().to(means.device)",
    ),
    (
        "K = camera.get_intrinsics_matrices().cuda()",
        "K = camera.get_intrinsics_matrices().to(self.device)",
    ),
)
# If an older partial patch used self.device at init time, fix that too.
ROLLBACK = (
    (
        ".float().to(self.device)",
        ".float().to(means.device)",
    ),
)


def patch_splatfacto(site_packages: Path | None = None) -> Path:
    if site_packages is None:
        import nerfstudio

        splatfacto = Path(nerfstudio.__file__).parent / "models" / "splatfacto.py"
    else:
        splatfacto = site_packages / "nerfstudio" / "models" / "splatfacto.py"

    if not splatfacto.exists():
        raise FileNotFoundError(f"splatfacto.py not found: {splatfacto}")

    text = splatfacto.read_text()
    original = text
    for old, new in ROLLBACK:
        if old in text:
            text = text.replace(old, new)
    for old, new in PATCHES:
        if old in text:
            text = text.replace(old, new)

    if text == original:
        if all(new in text for _, new in PATCHES):
            print(f"[patch] Already applied: {splatfacto}")
            return splatfacto
        missing = [old for old, new in PATCHES if old not in original and new not in text]
        raise RuntimeError(
            f"Could not find expected patterns in {splatfacto}. "
            f"nerfstudio version may have changed. Missing: {missing}"
        )

    splatfacto.write_text(text)
    print(f"[patch] Applied MPS fix to {splatfacto}")
    return splatfacto


if __name__ == "__main__":
    try:
        patch_splatfacto()
    except Exception as exc:
        print(f"[patch] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

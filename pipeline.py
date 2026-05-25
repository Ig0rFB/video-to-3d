import os
import subprocess
import sys

# Must be set before any torch imports — no-op on CUDA/CPU, enables MPS op fallback
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

VIDEO = "input/room.mp4"

steps = [
    ["uv", "run", "python", "01_extract_frames.py", "--video", VIDEO, "--fps", "2"],
    ["uv", "run", "python", "02_run_colmap.py"],
    ["uv", "run", "python", "03_train_gaussian.py"],
    [
        "uv",
        "run",
        "python",
        "05_export.py",
        "--checkpoint_dir",
        "outputs/splatfacto/latest-run",
    ],
]

for step in steps:
    print(f"\n>>> {' '.join(step)}\n")
    result = subprocess.run(step, check=False)
    if result.returncode != 0:
        print(f"\nStep failed: {' '.join(step)}")
        print("Resolve the error above before continuing.")
        sys.exit(1)

print("\nPipeline complete. Check export/ for outputs.")

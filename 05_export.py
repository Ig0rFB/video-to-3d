import argparse
import subprocess

from env_utils import resolve_cli


def export(checkpoint_dir: str, skip_render: bool = False) -> None:
    config = f"{checkpoint_dir}/config.yml"
    ns_export = resolve_cli("ns-export")

    subprocess.run(
        [
            ns_export,
            "gaussian-splat",
            "--load-config",
            config,
            "--output-dir",
            "export/",
        ],
        check=True,
    )
    print("[export] Wrote Gaussian splat under export/")

    if skip_render:
        print("[export] Skipped video render (--skip-render).")
        return

    # spiral needs VanillaDataManager; render-nearest-camera needs depth (splatfacto has neither).
    ns_render = resolve_cli("ns-render")
    subprocess.run(
        [
            ns_render,
            "interpolate",
            "--load-config",
            config,
            "--output-path",
            "export/render.mp4",
            "--pose-source",
            "train",
            "--order-poses",
            "True",
            "--interpolation-steps",
            "2",
            "--frame-rate",
            "24",
            "--render-nearest-camera",
            "False",
            "--rendered-output-names",
            "rgb",
        ],
        check=True,
    )
    print("[export] Wrote export/render.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PLY + trajectory video (uv run --no-sync python 05_export.py …)"
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only export PLY (skip ns-render; useful if ffmpeg or render fails).",
    )
    args = parser.parse_args()
    export(args.checkpoint_dir, skip_render=args.skip_render)

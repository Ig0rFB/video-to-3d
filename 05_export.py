import argparse
import subprocess


def export(checkpoint_dir: str) -> None:
    subprocess.run(
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            f"{checkpoint_dir}/config.yml",
            "--output-dir",
            "export/",
        ],
        check=True,
    )

    subprocess.run(
        [
            "ns-render",
            "spiral",
            "--load-config",
            f"{checkpoint_dir}/config.yml",
            "--output-path",
            "export/render.mp4",
            "--render-nearest-camera",
            "True",
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    args = parser.parse_args()
    export(args.checkpoint_dir)

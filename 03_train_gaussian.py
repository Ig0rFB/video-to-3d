import argparse
import subprocess

from device import get_device


def train(image_dir: str, colmap_dir: str, output_dir: str) -> None:
    device = get_device()
    ns_data_dir = "nerfstudio_data"

    subprocess.run(
        [
            "ns-process-data",
            "images",
            "--data",
            image_dir,
            "--output-dir",
            ns_data_dir,
            "--colmap-model-path",
            f"{colmap_dir}/sparse/0",
            "--no-gpu",
        ],
        check=True,
    )

    subprocess.run(
        [
            "ns-train",
            "splatfacto",
            "--data",
            ns_data_dir,
            "--output-dir",
            output_dir,
            "--max-num-iterations",
            "30000",
            "--pipeline.model.cull-alpha-thresh",
            "0.005",
            "--machine.device",
            device,
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="frames/")
    parser.add_argument("--colmap_dir", default="colmap_workspace/")
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()
    train(args.image_dir, args.colmap_dir, args.output_dir)

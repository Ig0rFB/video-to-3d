import argparse
import pathlib
import subprocess


def extract_frames(video_path: str, output_dir: str, fps: float = 2.0) -> None:
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        f"fps={fps}",
        "-q:v",
        "2",
        f"{output_dir}/%04d.jpg",
    ]
    subprocess.run(cmd, check=True)
    count = len(list(pathlib.Path(output_dir).glob("*.jpg")))
    print(f"Extracted {count} frames → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", default="frames/")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()
    extract_frames(args.video, args.output_dir, args.fps)

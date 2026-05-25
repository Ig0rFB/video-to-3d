import argparse
import pathlib

import pycolmap


def run_colmap(image_dir: str, output_dir: str) -> None:
    image_path = pathlib.Path(image_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / "database.db"

    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_path),
        camera_mode=pycolmap.CameraMode.AUTO,
    )

    num_images = len(list(image_path.glob("*.jpg")))
    if num_images <= 100:
        pycolmap.match_exhaustive(database_path=str(db_path))
    else:
        pycolmap.match_sequential(database_path=str(db_path))

    sparse_path = output_path / "sparse"
    sparse_path.mkdir(exist_ok=True)
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_path),
        output_path=str(sparse_path),
    )

    print(f"COLMAP complete. Reconstructed {len(maps)} model(s).")
    for i, model in maps.items():
        print(f"  Model {i}: {model.summary()}")

    if len(maps) == 0:
        raise RuntimeError(
            "COLMAP produced 0 models. Increase fps in step 01 and retry. "
            "If it still fails, trigger the DUSt3R fallback."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="frames/")
    parser.add_argument("--output_dir", default="colmap_workspace/")
    args = parser.parse_args()
    run_colmap(args.image_dir, args.output_dir)

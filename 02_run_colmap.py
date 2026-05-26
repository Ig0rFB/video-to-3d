import argparse
import shutil
from pathlib import Path
from typing import Any

import pycolmap

# Downscale 4K frames before SIFT; speeds up matching and improves stability
MAX_IMAGE_SIZE = 1920
# Warn if fewer than this fraction of frames register in the sparse model
MIN_REGISTERED_FRACTION = 0.3

# pycolmap incremental_mapping returns model_id -> Reconstruction
ColmapModels = dict[int, Any]


def _select_best_model(maps: ColmapModels) -> tuple[Any | None, int]:
    """Log each reconstructed model and return the one with the most registered images."""
    best_registered = 0
    best_model = None
    for model_id, model in maps.items():
        print(f"  Model {model_id}: {model.summary()}")
        registered = model.num_reg_images()
        if registered > best_registered:
            best_registered = registered
            best_model = model
    return best_model, best_registered


def _export_best_model_to_sparse_zero(
    sparse_path: Path,
    best_model: Any,
    best_registered: int,
    num_models: int,
) -> None:
    """
    nerfstudio expects colmap_workspace/sparse/0.

    Multiple models: clear sparse/ and write only the largest to sparse/0.
    Single model: rename sparse/1 (etc.) to sparse/0 when needed.
    """
    if num_models > 1:
        for sub in sparse_path.iterdir():
            if sub.is_dir():
                shutil.rmtree(sub)
        export_dir = sparse_path / "0"
        export_dir.mkdir(parents=True, exist_ok=True)
        best_model.write(str(export_dir))
        print(f"Exported largest model ({best_registered} images) to sparse/0")
        return

    only_dir = next(sparse_path.iterdir(), None)
    if only_dir is not None and only_dir.name != "0":
        target = sparse_path / "0"
        if target.exists():
            shutil.rmtree(target)
        only_dir.rename(target)


def _assert_registration_quality(
    maps: ColmapModels,
    best_registered: int,
    num_images: int,
) -> None:
    if len(maps) == 0:
        raise RuntimeError(
            "COLMAP produced 0 models. Increase fps in step 01 and retry. "
            "If it still fails, trigger the DUSt3R fallback."
        )

    print(f"Registered {best_registered} / {num_images} extracted frames in the largest model.")
    if best_registered < num_images * MIN_REGISTERED_FRACTION:
        raise RuntimeError(
            f"COLMAP registered only {best_registered}/{num_images} frames "
            f"(<{MIN_REGISTERED_FRACTION:.0%}). Try:\n"
            "  - Re-run after deleting colmap_workspace/ and frames/ (02 now uses a single shared camera)\n"
            "  - Higher overlap: slower walk, --fps 3 or 4 in step 01\n"
            "  - More texture/lighting in the scene"
        )


def run_colmap(image_dir: str, output_dir: str) -> None:
    image_path = Path(image_dir)
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / "database.db"

    num_images = len(list(image_path.glob("*.jpg")))
    if num_images == 0:
        raise RuntimeError(f"No .jpg files found in {image_path}")

    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.max_image_size = MAX_IMAGE_SIZE

    # SINGLE: one shared camera for all frames (required for phone/video walkthroughs).
    # AUTO wrongly assigns a separate camera per frame and SfM usually stalls at ~2 views.
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(image_path),
        camera_mode=pycolmap.CameraMode.SINGLE,
        extraction_options=extraction_options,
    )

    # Exhaustive matching for <=100 frames: needed to find a strong initial pair.
    # Sequential-only matching often yields 0 models on walkthrough video.
    if num_images <= 100:
        pycolmap.match_exhaustive(database_path=str(db_path))
    else:
        pycolmap.match_sequential(database_path=str(db_path))
        pycolmap.match_exhaustive(database_path=str(db_path))

    pipeline_options = pycolmap.IncrementalPipelineOptions()
    pipeline_options.mapper.init_min_num_inliers = 50
    pipeline_options.mapper.abs_pose_min_num_inliers = 15

    sparse_path = output_path / "sparse"
    if sparse_path.exists():
        shutil.rmtree(sparse_path)
    sparse_path.mkdir(parents=True, exist_ok=True)

    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(image_path),
        output_path=str(sparse_path),
        options=pipeline_options,
    )

    print(f"COLMAP complete. Reconstructed {len(maps)} model(s).")
    best_model, best_registered = _select_best_model(maps)

    if best_model is not None:
        _export_best_model_to_sparse_zero(
            sparse_path, best_model, best_registered, len(maps)
        )

    _assert_registration_quality(maps, best_registered, num_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="frames/")
    parser.add_argument("--output-dir", default="colmap_workspace/")
    args = parser.parse_args()
    run_colmap(args.image_dir, args.output_dir)

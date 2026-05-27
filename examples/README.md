# Example artefacts

This folder is intended to be **committed to git** and shared as a lightweight bundle of example outputs.

Large runtime outputs (videos, checkpoints, datasets) are gitignored. Instead, we extract a small, consistent
set of PNG frames which are easy to review.

## Generate frames

Run from the repository root *after* you have produced:

- `export/render.mp4`
- `semantic/overlay.mp4`

```bash
uv run --no-sync python scripts/prepare_examples.py --progress 0.5
```

Outputs are written to `examples/frames/`:

- `input_frame.png` (optional)
- `render_frame.png`
- `semantic_overlay_frame.png`

If you want a specific frame index instead of a timestamp:

```bash
uv run --no-sync python scripts/prepare_examples.py --frame 120 --fps 30
```

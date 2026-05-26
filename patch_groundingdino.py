"""
Patch installed GroundingDINO for transformers >= 5.x (missing BertModel.get_head_mask).

Run once after `uv add groundingdino-py`, or let 04_semantic_lift.py call it automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

PATCH_MARKER = "# video3d: transformers 5.x bertwarper fix"

OLD_INIT = """\
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.invert_attention_mask = bert_model.invert_attention_mask
        self.get_head_mask = bert_model.get_head_mask
"""

NEW_INIT = f"""\
        self.get_extended_attention_mask = bert_model.get_extended_attention_mask
        self.invert_attention_mask = bert_model.invert_attention_mask
        {PATCH_MARKER}
        self.get_head_mask = getattr(bert_model, "get_head_mask", None)
"""

OLD_MASK_CALL = """\
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
"""

NEW_MASK_CALL = """\
        try:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape, device
            )
        except TypeError:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
                attention_mask, input_shape
            )
"""

OLD_HEAD = """\
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
"""

NEW_HEAD = """\
        if self.get_head_mask is not None:
            head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        else:
            head_mask = [None] * self.config.num_hidden_layers
"""


def patch_groundingdino_bertwarper(site_packages: Path | None = None) -> Path:
    if site_packages is None:
        import groundingdino

        bertwarper = Path(groundingdino.__file__).resolve().parent / "models" / "GroundingDINO" / "bertwarper.py"
    else:
        bertwarper = site_packages / "groundingdino" / "models" / "GroundingDINO" / "bertwarper.py"

    if not bertwarper.exists():
        raise FileNotFoundError(f"bertwarper.py not found: {bertwarper}")

    text = bertwarper.read_text()
    if PATCH_MARKER in text:
        print(f"[patch] GroundingDINO bertwarper already patched: {bertwarper}")
        return bertwarper

    changed = False
    if OLD_INIT in text:
        text = text.replace(OLD_INIT, NEW_INIT, 1)
        changed = True
    if OLD_MASK_CALL in text:
        text = text.replace(OLD_MASK_CALL, NEW_MASK_CALL, 1)
        changed = True
    if OLD_HEAD in text:
        text = text.replace(OLD_HEAD, NEW_HEAD, 1)
        changed = True

    if not changed:
        raise RuntimeError(
            f"Could not patch {bertwarper} — file layout may have changed. "
            "Try: uv pip install 'transformers>=4.35,<5'"
        )

    bertwarper.write_text(text)
    print(f"[patch] Patched GroundingDINO bertwarper for transformers 5.x: {bertwarper}")
    return bertwarper


if __name__ == "__main__":
    try:
        patch_groundingdino_bertwarper()
    except Exception as exc:
        print(f"[patch] Failed: {exc}", file=sys.stderr)
        sys.exit(1)

"""Generate/check an example patch without editing the pinned Villa checkout."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
COMMIT = "f07d33be6a00d12ace7d6a9465efe17c78ed7b47"
PREFIX = "vesuvius/src/vesuvius/models/build/pretrained_backbones/"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Expected one source anchor: " + old)
    return source.replace(old, new, 1)


def sources(repo):
    paths = [PREFIX + "dinovol_2_eva.py", PREFIX + "dinovol_2_builder.py"]
    original = {}
    for path in paths:
        original[path] = subprocess.check_output(
            ["git", "-C", str(repo), "show", COMMIT + ":" + path], text=True)
    eva = original[paths[0]]
    eva = replace_once(eva, "from .patch_encode_decode import PatchEmbed, PatchEmbedDeeper\n",
                      "from .patch_encode_decode import PatchEmbed, PatchEmbedDeeper\nfrom .sdpa_padding import inference_sdpa\n")
    eva = replace_once(eva, "            norm_layer: Optional[Callable] = None,\n    ):",
                      "            norm_layer: Optional[Callable] = None,\n            sdpa_head_padding: bool = False,\n    ):")
    eva = replace_once(eva, "        self.fused_attn = use_fused_attn()\n",
                      "        self.fused_attn = use_fused_attn()\n        self.sdpa_head_padding = sdpa_head_padding\n")
    old = """            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )"""
    new = """            if self.sdpa_head_padding and not self.training:
                x = inference_sdpa(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.,
                    enabled=True,
                )
            else:
                x = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.,
                )"""
    eva = replace_once(eva, old, new)
    eva = replace_once(eva, "            ndim: Optional[int] = None,\n    ):",
                      "            ndim: Optional[int] = None,\n            sdpa_head_padding: bool = False,\n    ):")
    eva = replace_once(eva, "            norm_layer=norm_layer if scale_attn_inner else None,\n",
                      "            norm_layer=norm_layer if scale_attn_inner else None,\n            sdpa_head_padding=sdpa_head_padding,\n")
    eva = replace_once(eva, "            deeper_embed_batch_chunk_size: Optional[int] = None,\n    ):",
                      "            deeper_embed_batch_chunk_size: Optional[int] = None,\n            sdpa_head_padding: bool = False,\n    ):")
    eva = replace_once(eva, "                rope_kwargs=rope_kwargs if self.use_per_block_rope else None,\n                ndim=self.ndim,\n",
                      "                rope_kwargs=rope_kwargs if self.use_per_block_rope else None,\n                ndim=self.ndim,\n                sdpa_head_padding=sdpa_head_padding,\n")
    builder = original[paths[1]]
    anchor = '    "qkv_fused": '
    for value in ("False", "True"):
        builder = replace_once(builder, anchor + value + ",\n",
                               anchor + value + ',\n    "sdpa_head_padding": False,\n')
    updated = {paths[0]: eva, paths[1]: builder,
               PREFIX + "sdpa_padding.py": (ROOT / "sdpa_padding.py").read_text()}
    for path, source in updated.items():
        ast.parse(source, filename=path)
    return original, updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--villa", type=Path, required=True)
    args = parser.parse_args()
    original, updated = sources(args.villa)
    pieces = []
    for path, source in updated.items():
        pieces.append("diff --git a/" + path + " b/" + path + "\n")
        if path not in original:
            pieces.append("new file mode 100644\n")
        pieces.extend(difflib.unified_diff(
            original.get(path, "").splitlines(keepends=True), source.splitlines(keepends=True),
            fromfile="a/" + path if path in original else "/dev/null", tofile="b/" + path))
    output = ROOT / "example_villa_integration.patch"
    output.write_text("".join(pieces), newline="\n")
    subprocess.run(["git", "-C", str(args.villa), "apply", "--check", str(output)], check=True)
    receipt = {"status": "PATCH_CHECK_AND_AST_PASS", "base_commit": COMMIT,
               "modified_checkout": False,
               "patch_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "base_files_sha256": {path: hashlib.sha256(source.encode()).hexdigest()
                                      for path, source in original.items()},
               "helper_sha256": hashlib.sha256((ROOT / "sdpa_padding.py").read_bytes()).hexdigest()}
    (ROOT / "INTEGRATION_PATCH_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

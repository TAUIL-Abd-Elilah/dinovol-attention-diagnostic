"""Small CPU factory smoke using edited source in memory, not a modified checkout."""
import argparse
import json
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch

from make_integration_example import COMMIT, PREFIX, ROOT, sources


def load_package(name, texts, dependency_dir):
    package = types.ModuleType(name)
    package.__path__ = [str(dependency_dir)]
    sys.modules[name] = package
    for leaf in ("sdpa_padding", "dinovol_2_eva", "dinovol_2_builder"):
        key = PREFIX + leaf + ".py"
        if key not in texts:
            continue
        fullname = name + "." + leaf
        module = types.ModuleType(fullname)
        module.__package__ = name
        module.__file__ = str(ROOT / (name + "_in_memory_" + leaf + ".py"))
        sys.modules[fullname] = module
        exec(compile(texts[key], module.__file__, "exec"), module.__dict__)
    return sys.modules[name + ".dinovol_2_eva"], sys.modules[name + ".dinovol_2_builder"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--villa", type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    original, edited = sources(args.villa)
    native_eva, native_builder = load_package("native_example", original, args.villa / PREFIX)
    edited_eva, edited_builder = load_package("edited_example", edited, args.villa / PREFIX)
    cases = []
    for model_type in ("v1", "v2"):
        for chunks in (0, 1):
            config = dict(model_type=model_type, global_crops_size=8, local_crops_size=8,
                          patch_size=4, embed_dim=12, num_heads=2, depth=1,
                          drop_path_rate=0.0, block_chunks=chunks)
            torch.manual_seed(7)
            native = native_builder.build_dinovol_2_backbone(config)
            torch.manual_seed(7)
            default = edited_builder.build_dinovol_2_backbone(config)
            torch.manual_seed(7)
            enabled = edited_builder.build_dinovol_2_backbone({**config, "sdpa_head_padding": True})
            states = [model.state_dict() for model in (native, default, enabled)]
            assert states[0].keys() == states[1].keys() == states[2].keys()
            assert all(torch.equal(states[0][k], state[k]) for state in states[1:] for k in state)
            get_attention = lambda model: next(m for m in model.modules() if type(m).__name__ == "EvaAttention")
            n, d, e = map(get_attention, (native, default, enabled))
            assert d.sdpa_head_padding is False and e.sdpa_head_padding is True
            x = torch.randn(1, 5, 12)
            for attention in (n, d, e):
                attention.eval()
            with torch.inference_mode():
                expected = n(x)
                with patch.object(edited_eva, "inference_sdpa", wraps=edited_eva.inference_sdpa) as helper:
                    assert torch.equal(d(x), expected)
                    helper.assert_not_called()
                    assert torch.equal(e(x), expected)
                    helper.assert_called_once()
                e.train()
                with patch.object(edited_eva, "inference_sdpa", side_effect=AssertionError("training must stay native")):
                    assert torch.equal(e(x), expected)
            cases.append({"model_type": model_type, "block_chunks": chunks,
                          "factory_flag_reached_attention": True, "state_dict_identical": True,
                          "default_and_cpu_opt_in_match_native": True,
                          "training_bypasses_helper": True})
    assert not torch.cuda.is_initialized()
    receipt = {"status": "PASS", "base_commit": COMMIT, "cases": cases,
               "cuda_initialized": False, "modified_checkout": False,
               "torch": torch.__version__, "cpu_threads": torch.get_num_threads()}
    (ROOT / "INTEGRATION_CPU_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

"""Extract exact SDPA operator counts from a profiler trace."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.trace.read_bytes()
    events = json.loads(raw)['traceEvents']
    names = ('aten::scaled_dot_product_attention', 'aten::_scaled_dot_product_attention_math',
             'aten::_scaled_dot_product_flash_attention', 'aten::_scaled_dot_product_efficient_attention',
             'aten::_scaled_dot_product_cudnn_attention')
    counts = collections.Counter(e.get('name') for e in events
                                 if e.get('ph') == 'X' and e.get('cat') == 'cpu_op')
    result = {'trace_sha256': hashlib.sha256(raw).hexdigest(),
              'count_rule': 'Complete-duration (ph=X) CPU operator events, exact names',
              'operator_counts': {name: counts[name] for name in names}}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

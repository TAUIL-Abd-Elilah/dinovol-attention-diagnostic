"""Build a compact reading index from the published complete comparison records."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
result = {'scope': 'Frozen DINO similarity only; one Windows/RTX3090 environment',
          'label_equivalence': 'Untested; similarity >0.5 eligibility is not the production pseudo-label',
          'replicas': {}}
for cube in 'ABC':
    path = root / 'evidence' / f'comparison_replica_{cube}.json'
    raw = path.read_bytes()
    report = json.loads(raw)
    comparison = report['comparisons'][0]
    result['replicas'][cube] = {
        'comparison_file': str(path.relative_to(root)).replace('\\', '/'),
        'comparison_sha256': hashlib.sha256(raw).hexdigest(),
        'input_sha256': report['input_provenance']['data']['array_sha256'],
        'native_timing': report['baseline']['timing_summary'],
        'helper_timing': comparison['candidate']['timing_summary'],
        'performance': comparison['timing'],
        'native_memory': report['baseline']['gpu_memory_timed'],
        'helper_memory': comparison['candidate']['gpu_memory_timed'],
        'native_backend': report['baseline']['backend']['operator_counts'],
        'helper_backend': comparison['candidate']['backend']['operator_counts'],
        'final_map_error': comparison['stages']['similarity_first'],
        'eligibility': comparison['eligibility'],
    }
(root / 'evidence/SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print('Wrote compact evidence/SUMMARY.json')

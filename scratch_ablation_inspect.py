import os, json
from studies.analysis.common.config import namespace_root
from studies.analysis.common import io as cio

ablation_root = namespace_root('ablation')
eval_dir = ablation_root / 'eval'
for root, dirs, files in os.walk(str(cio.to_long_path(eval_dir))):
    if 'eval_results.json' in files:
        ep = os.path.join(root, 'eval_results.json')
        data = json.loads(open(ep, 'r', encoding='utf-8').read())
        rel = os.path.relpath(ep, str(cio.to_long_path(eval_dir)))
        has_trace = 'trace.jsonl' in files
        print(f"{data.get('task')} | {data.get('method_name')} | {data.get('seed')} | dir={rel} | trace={has_trace}")

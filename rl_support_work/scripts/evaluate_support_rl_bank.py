"""Evaluate an exact saved training bank with canonical sensor-only inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import train_support_rl as experiment

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--bank-from', required=True)
p.add_argument('--output', required=True)
p.add_argument('--mode', required=True)
a, _ = p.parse_known_args()
assert a.mode in ('zero', 'evaluate')
index = sys.argv.index('--bank-from')
del sys.argv[index:index+2]
bank_path = Path(a.bank_from)
bank = json.loads(bank_path.read_text())
original_cfg = experiment.DeploymentEnvCfg


def saved_cfg(**kwargs):
    out = Path(a.output)
    shutil.copy2(__file__, out/'source'/Path(__file__).name)
    manifest = json.loads((out/'manifest.json').read_text())
    manifest['saved_geometry_bank'] = dict(path=str(bank_path),
        sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    assert kwargs['geometry_count'] == len(bank)
    return original_cfg(**kwargs, geometry_bank=bank)


experiment.DeploymentEnvCfg = saved_cfg
experiment.main()

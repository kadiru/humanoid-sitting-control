"""Counterfactual evaluation: zero only the new live-depth policy channels.

Initial depth geometry, proprioception, frozen scaffold, learned weights and all
physical evaluation criteria remain identical. Live depth is still measured for
diagnostics. This is an input ablation, not a retrained blind-policy comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import train_support_rl as experiment

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--output',required=True); p.add_argument('--mode',required=True); p.add_argument('--checkpoint',required=True)
a,_=p.parse_known_args()
assert a.mode=='evaluate'
original=experiment.SupportVecEnv._observe


def without_live_depth(self,update_depth):
    original(self,update_depth)
    self.obs['policy'][:,132:142]=0.


experiment.SupportVecEnv._observe=without_live_depth
experiment.main()
out=Path(a.output)
shutil.copy2(__file__,out/'source'/Path(__file__).name)
manifest=json.loads((out/'manifest.json').read_text())
manifest['ablation']={'kind':'zero live-depth observation channels 132:142',
    'initial_depth_geometry_retained':True,'live_measurement_recording_retained':True,
    'ablation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

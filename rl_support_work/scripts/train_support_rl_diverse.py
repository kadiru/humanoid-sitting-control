"""Second PPO pilot: balanced geometry coverage; same reward and action contract.

The first four random geometries covered almost no negative seat yaw and no
deep bench. This run deliberately covers each allowed parameter range. No
failed held-out geometry is copied into the training bank.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import train_support_rl as experiment

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--output',required=True); p.add_argument('--mode',required=True)
p.add_argument('--geometry-count',type=int,required=True); p.add_argument('--geometry-seed',type=int,required=True)
a,_=p.parse_known_args()
assert a.mode=='train' and a.geometry_count==16
original_cfg=experiment.DeploymentEnvCfg


def diverse_cfg(**kwargs):
    out=Path(a.output)
    shutil.copy2(__file__,out/'source'/Path(__file__).name)
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['training_bank_design']={'kind':'16 balanced rigid geometries; independent permutations of four levels per dimension',
        'heldout_examples_copied':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rng=np.random.default_rng(a.geometry_seed)
    bank=experiment.furniture.bank(2,a.geometry_seed,16,True)
    ranges=dict(depth=[.41,.47,.53,.59],width=[.77,.86,.95,1.04],
                height=[.278,.294,.310,.324],yaw=np.deg2rad([-5.,-1.7,1.7,5.]).tolist())
    for key,levels in ranges.items():
        values=rng.permutation(np.tile(levels,4))
        for g,value in zip(bank,values):g[key]=float(value)
    return original_cfg(**kwargs,geometry_bank=bank)


experiment.DeploymentEnvCfg=diverse_cfg
experiment.main()
out=Path(a.output)
shutil.copy2(__file__,out/'source'/Path(__file__).name)
manifest=json.loads((out/'manifest.json').read_text())
manifest['training_bank_design']={'kind':'16 balanced rigid geometries; independent permutations of four levels per dimension',
    'heldout_examples_copied':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

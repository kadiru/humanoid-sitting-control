"""Diagnostic intervention: same final PPO, original progression fixed at one."""
import hashlib
import json
from pathlib import Path
import shutil
import torch
import train_support_rl_pace_validated as checked
pace=checked.pace;experiment=checked.experiment
assert pace.independent.args.mode=='evaluate','Diagnostic only; no training through this entry point'

def neutral(previous,action,ready):
    return torch.ones_like(previous)

pace.update_pace=neutral
old_cfg=experiment.DeploymentEnvCfg

def cfg(**kwargs):
    result=old_cfg(**kwargs);out=Path(pace.independent.args.output)
    manifest=json.loads((out/'manifest.json').read_text())
    source=Path(__file__);shutil.copy2(source,out/'source'/source.name)
    manifest['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['intervention']='Final candidate unchanged; pace action ignored, applied pace and observed filter fixed at 1. No training.'
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return result

experiment.DeploymentEnvCfg=cfg
if __name__=='__main__':experiment.main()

"""Use previously exercised wider training placements; audit fresh reset starts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
p=argparse.ArgumentParser(add_help=False);p.add_argument('--placement-probe',action='store_true')
placement_options,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
import train_support_rl_pace as pace
experiment=pace.experiment
BANK=Path('/data/humanoid-sitting-perception/support_rl_independent_train_04/geometry_bank.json')


def validated_bank(count,seed,balanced):
    if pace.pace_options.pace_curriculum=='mixed':
        assert balanced and count==16
        # Training prototypes only; never take shapes from an evaluation bank.
        # Deterministic first half, not ranked by success or reward.
        return experiment.furniture.bank(2,seed+101,8,True)+json.loads(BANK.read_text())[:8]
    return pace.original_bank(count,seed,balanced)


class PlacementCheckedEnv(pace.PaceSupportVecEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if placement_options.placement_probe:
            rows=[]
            with torch.inference_mode():
                for repeat in range(12):
                    self.reset();self.step(torch.zeros(self.num_envs,24,device=self.device))
                    m=experiment.physical_metrics(self.raw)
                    rows.append(dict(repeat=repeat,ready=self.ready.tolist(),
                        failures={k:v.tolist() for k,v in m['flags'].items()},
                        peak_force_N=m['peak_force'].tolist()))
                    assert not m['failure'].any(),rows[-1]
                assert all(any(r['ready'][i] for r in rows) for i in range(self.num_envs))
                (self.out/'placement_probe.json').write_text(json.dumps(dict(passed=True,
                    reset_starts=self.num_envs*12,no_first_tick_physical_failure=True,rows=rows),indent=2)+'\n')
                self.episodes.clear();self.reset()


previous_cfg=experiment.DeploymentEnvCfg

def checked_cfg(**kwargs):
    cfg=previous_cfg(**kwargs);out=Path(pace.independent.args.output)
    manifest=json.loads((out/'manifest.json').read_text())
    source=Path(__file__);shutil.copy2(source,out/'source'/source.name)
    manifest['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest['placement_validation']=dict(probe=placement_options.placement_probe,
        mixed_hard_source=str(BANK),source_sha256=hashlib.sha256(BANK.read_bytes()).hexdigest(),
        indices=list(range(8)),selection='First eight previously used TRAINING prototypes, no success ranking; holdouts unchanged.',
        excludes_interrupted_run='support_rl_pace_mixed_train_01')
    shutil.copy2(BANK,out/'source/prior_training_geometry_bank.json')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return cfg

pace.independent.curriculum_bank=validated_bank
experiment.DeploymentEnvCfg=checked_cfg
experiment.SupportVecEnv=PlacementCheckedEnv
if __name__=='__main__':experiment.main()

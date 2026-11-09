"""Isolated cap-only PPO comparison; frozen prior experiment sources preserved."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from humanoid_sitting.support_capped_impact import CappedImpactStudy

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--cap-arm',choices=['legacy','capped'],required=True)
options,rest=p.parse_known_args()
sys.argv=[sys.argv[0],*rest,'--reward-arm','legacy']
import train_support_reward_study as previous
experiment=previous.experiment
previous.objective=CappedImpactStudy(options.cap_arm)


class CappedImpactEnv(previous.RewardStudyEnv):
    def finish_episode(self,ids=None):
        rows=super().finish_episode(ids)
        for row in rows:
            i=row['env']; ticks=previous.objective.hold.count(i)
            # Compare independent counters before replacing the reported seconds.
            assert abs(ticks*.02-row['hold_s']) < .001
            row['legacy_float_hold_s']=row['hold_s']
            row['legacy_float_success']=row['success']
            row['hold_ticks']=ticks; row['hold_s']=ticks*.02
            row['success']=bool(self.initial_standing[i] and self.alive[i] and
                self.ready[i] and self.standing_checked[i] and ticks>=150)
        return rows


def capped_cfg(**kwargs):
    cfg=previous.study_cfg(**kwargs)
    out=Path(previous.split.independent.args.output)
    manifest=json.loads((out/'manifest.json').read_text())
    for path in (Path(__file__).resolve(),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_capped_impact.py',
                 experiment.ROOT/'scripts/test_support_capped_impact.py'):
        shutil.copy2(path,out/'source'/path.name)
        manifest['hashes'][str(path.relative_to(experiment.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['reward_study'].update(arm=options.cap_arm,
        peak_cost='capped only: -5 * increase in max_so_far(min(max(F/W-1,0)^2,1)); entire episode cost <=5',
        leg_economy='unchanged original reward in both arms',
        evaluation_reward='legacy objective for all evaluation; exact integer hold scoring')
    manifest['hold_timer']=dict(dtype='int64',control_dt_s=.02,required_ticks=150,
        legacy_float_score_retained=True,independent_reset=True,runtime_observations_unchanged=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return cfg


experiment.support_objective=previous.objective
experiment.SupportVecEnv=CappedImpactEnv
experiment.DeploymentEnvCfg=capped_cfg
if __name__=='__main__':experiment.main()

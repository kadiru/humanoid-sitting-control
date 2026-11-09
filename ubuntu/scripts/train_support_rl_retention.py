"""Matched joint-only PPO study: fixed progression, with/without parent KL."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from rsl_rl.algorithms import PPO
from humanoid_sitting.support_retention import RetentionPPO

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--retention-coefficient',type=float,default=0.)
p.add_argument('--parent',required=True)
p.add_argument('--training-bank')
options,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
import train_support_rl_independent as independent
experiment=independent.experiment


class RetentionEnv(independent.IndependentSupportVecEnv):
    def __init__(self,*args,**kwargs):
        self.sensor_replay=[];self.action_samples=0;self.saturated_actions=0
        self.noise_sum=np.zeros(23);self.noise_square_sum=np.zeros(23)
        super().__init__(*args,**kwargs)

    def step(self,actions):
        # Save only actual sensor observations. This will also permit function
        # comparisons without privileged pose/force channels in future studies.
        if self.t%5==0 and (self.args.mode!='train' or self.t%100==0):
            self.sensor_replay.append(self.obs['policy'].cpu().numpy().copy())
        if self.args.mode=='train' and hasattr(self,'training_actor'):
            mean=self.training_actor.output_mean
            diff=(actions-mean).detach().cpu().numpy()
            self.noise_sum+=diff.sum(0);self.noise_square_sum+=(diff**2).sum(0)
            self.action_samples+=len(actions)
            self.saturated_actions+=int((actions.abs()>=1).sum())
        result=super().step(actions)
        assert self.num_actions==23 and self.obs['policy'].shape[1]==144
        # Original actions-mode scaffold: stochastic and deterministic runs
        # have exactly the same recurrent progression, including during resets.
        rate=self.controller.last_rate
        assert torch.all((rate==0.)|(rate==1.)),rate
        assert torch.all(rate.reshape(-1)[self.ready & ~self.warming]==1.),rate
        return result

    def close(self):
        if self.sensor_replay:
            np.savez_compressed(self.out/'sensor_replay.npz',policy=np.stack(self.sensor_replay))
        n=max(self.action_samples,1)
        (self.out/'action_audit.json').write_text(json.dumps(dict(
            pace_action=False,actor_channels=144,actions=23,admitted_progression_exactly_one=True,
            sample_rows=self.action_samples,saturated_action_coordinates=self.saturated_actions,
            noise_mean=(self.noise_sum/n).tolist(),noise_rms=np.sqrt(self.noise_square_sum/n).tolist(),
            note='Gaussian joint exploration remains; only the one-sided pace actuator is removed. Shared physical clipping and slew limits remain.'),indent=2)+'\n')
        super().close()


original_cfg=experiment.DeploymentEnvCfg
def retention_cfg(**kwargs):
    cfg=original_cfg(**kwargs);out=Path(independent.args.output)
    m=json.loads((out/'manifest.json').read_text())
    for source in [Path(__file__).resolve(),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_retention.py',
                   experiment.ROOT/'scripts/test_support_retention.py']:
        shutil.copy2(source,out/'source'/source.name)
        m['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    upstream=Path(inspect.getfile(PPO));shutil.copy2(upstream,out/'source/rsl_rl_ppo_original.py')
    m['retention']=dict(coefficient=options.retention_coefficient,parent=options.parent,
        parent_sha256=hashlib.sha256(Path(options.parent).read_bytes()).hexdigest(),
        KL='KL(frozen parent || current), sum over 23 action coordinates, mean over minibatch',
        normalizers='parent actor and critic statistics frozen in both matched arms',
        optimizer='fresh in both arms; identical weights and Gaussian std from parent',
        progression='original actions-mode scaffold at 1; no pace action',
        upstream_ppo_sha256=hashlib.sha256(upstream.read_bytes()).hexdigest(),
        training_bank=options.training_bank)
    if options.training_bank:
        shutil.copy2(options.training_bank,out/'source/training_geometry_bank.json')
        m['retention']['training_bank_sha256']=hashlib.sha256(Path(options.training_bank).read_bytes()).hexdigest()
    m['method']='Sensor-only joint-residual PPO; matched fixed-normalizer continuation with optional frozen-parent KL'
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    return cfg

original_bank=independent.curriculum_bank
def training_bank(count,seed,balanced):
    if options.training_bank:
        assert balanced,'Archived bank is training-only'
        bank=json.loads(Path(options.training_bank).read_text());assert len(bank)==count
        return bank
    return original_bank(count,seed,balanced)

original_runner=experiment.create_rsl_rl_runner
def retention_runner(env,agent,log_dir):
    runner=original_runner(env,agent,log_dir=log_dir)
    assert type(runner.alg) is PPO
    runner.alg.__class__=RetentionPPO
    parent=torch.load(options.parent,map_location=env.device,weights_only=False)
    runner.alg.configure_retention(parent['actor_state_dict'],options.retention_coefficient)
    if env.args.mode=='train':
        assert not env.args.checkpoint,'Matched training always starts fresh from --parent'
        runner.alg.actor.load_state_dict(parent['actor_state_dict'])
        runner.alg.critic.load_state_dict(parent['critic_state_dict'])
        prepared=Path(log_dir)/'prepared.pt';runner.save(str(prepared))
        env.args.checkpoint=str(prepared)
    else:
        assert env.args.checkpoint,'Evaluation requires an explicit checkpoint'
    env.training_actor=runner.alg.actor
    old_update=runner.alg.update
    def audited_update():
        result=old_update()
        with (Path(log_dir)/'retention_updates.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        return result
    runner.alg.update=audited_update
    return runner

independent.curriculum_bank=training_bank
experiment.DeploymentEnvCfg=retention_cfg
experiment.SupportVecEnv=RetentionEnv
experiment.create_rsl_rl_runner=retention_runner
if __name__=='__main__':experiment.main()

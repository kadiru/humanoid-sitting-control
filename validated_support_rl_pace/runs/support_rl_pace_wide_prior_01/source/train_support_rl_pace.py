"""Pilot 4: 23 joint residuals plus one bounded, filtered progression request."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
from tensordict import TensorDict
from humanoid_sitting.support_pace import PaceController, update_pace, migrate_state

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--warm-start-23')
p.add_argument('--pace-curriculum',choices=['standard','mixed'],default='standard')
pace_options,remaining=p.parse_known_args();sys.argv=[sys.argv[0],*remaining]
import train_support_rl_independent as independent
experiment=independent.experiment


class PaceSupportVecEnv(independent.IndependentSupportVecEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.num_actions=24

    def reset(self):
        self.pace=torch.ones(self.num_envs,device=self.device)
        return super().reset()

    def _reset_rows(self,ids):
        self.pace[ids]=1.
        super()._reset_rows(ids)

    def _observe(self,update_depth):
        self.controller.progress_rate=self.pace
        super()._observe(update_depth)
        self.obs=TensorDict({'policy':torch.cat((self.obs['policy'],self.pace[:,None]),-1)},batch_size=[self.num_envs])
        assert self.obs['policy'].shape==(self.num_envs,145)

    def step(self,actions):
        # The rate request changes the NEXT nominal target, generated after the
        # ordinary physics/sensor tick. Current residuals act on the current
        # nominal. The realized filter state accompanies that next observation.
        before=self.pace.clone()
        pace_action=actions[:,23] if actions.shape[1]==24 else torch.zeros_like(self.pace)
        self.pace=update_pace(self.pace,pace_action,self.ready)
        trace_len=len(self.trace)
        result=super().step(actions[:,:23])
        if len(self.trace)>trace_len:
            self.trace[-1].update(command_pace=before.cpu().numpy(),
                next_pace=self.pace.cpu().numpy().copy(),pace_request=pace_action.cpu().numpy().copy())
        return result

    def reset_probe(self):
        super().reset_probe()
        with torch.inference_mode():
            for _ in range(7):self.step(torch.zeros(self.num_envs,24,device=self.device))
            for _ in range(20):
                action=torch.zeros(self.num_envs,24,device=self.device);action[:,-1]=-1.
                self.step(action)
            admitted=self.ready.clone()
            assert (self.pace[admitted]<.5).all(),self.pace
            before=self.pace.clone();ids=torch.tensor([0],device=self.device)
            self._reset_rows(ids)
            assert self.pace[0]==1. and torch.equal(self.pace[1:],before[1:])
            self._observe(False)
            assert torch.equal(self.obs['policy'][:,-1],self.pace)
            assert self.warming[0] and not self.ready[0]
            check=dict(passed=True,nonneutral_pace=before.tolist(),selected_reset_to_neutral=True,
                other_pace_unchanged=True,filter_state_observed=True,one_tick_action_delay=True)
            (self.out/'pace_probe.json').write_text(json.dumps(check,indent=2)+'\n')
            self.episodes.clear();self.reset()


original_bank=independent.curriculum_bank

def staged_bank(count,seed,balanced):
    bank=original_bank(count,seed,balanced)
    if pace_options.pace_curriculum=='mixed':
        assert balanced and count==16
        easy=experiment.furniture.bank(2,seed+101,8,True)
        hard=original_bank(8,seed+103,True)
        bank=easy+hard
    return bank

original_cfg=experiment.DeploymentEnvCfg

def pace_cfg(**kwargs):
    cfg=original_cfg(**kwargs)
    out=Path(independent.args.output)
    manifest=json.loads((out/'manifest.json').read_text())
    for source in [Path(__file__),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_pace.py',
                   experiment.ROOT/'scripts/test_support_pace.py']:
        shutil.copy2(source,out/'source'/source.name)
        manifest['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    manifest.update(method='PPO: frozen sensor scaffold + 23 residuals + filtered recurrent progression request',
        actor_inputs='Original 144 sensor-derived channels + realized progression filter state = 145',
        critic_inputs='same 145 sensor-only channels',
        pace=dict(action_index=23,range=[.25,1.],neutral=1.,filter_gain=.1,slew_per_tick=.04,
            command_delay_ticks=1,new_action_initial_std=.5,latent_progression_not_physical_velocity=True),
        pace_curriculum=pace_options.pace_curriculum,warm_start_23=pace_options.warm_start_23,
        optimizer_migration='Fresh optimizer for 23-to-24 expansion; ordinary resume for subsequent 24-action stage')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return cfg

original_runner=experiment.create_rsl_rl_runner

def make_runner(env,agent,log_dir):
    runner=original_runner(env,agent,log_dir=log_dir)
    if pace_options.warm_start_23:
        assert not env.args.checkpoint,'Use either migration or a 24-action resume'
        path=Path(pace_options.warm_start_23)
        parent=torch.load(path,map_location=env.device,weights_only=False)
        for group,model,actor in [('actor_state_dict',runner.alg.actor,True),('critic_state_dict',runner.alg.critic,False)]:
            model.load_state_dict(migrate_state(parent[group],model.state_dict(),actor))
        # The base entry point zero-initializes actors without --checkpoint.
        # Have it load our fully initialized, architecture-compatible snapshot.
        prepared=Path(log_dir)/'migrated.pt';runner.save(str(prepared))
        env.args.checkpoint=str(prepared)
        (Path(log_dir)/'migration.json').write_text(json.dumps(dict(parent=str(path),
            parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),optimizer_reset=True,
            old_action_mean_preserved=True,new_action_mean=0.,new_action_std=.5),indent=2)+'\n')
    return runner

independent.curriculum_bank=staged_bank
experiment.SoftLandingController=PaceController
experiment.SupportVecEnv=PaceSupportVecEnv
experiment.DeploymentEnvCfg=pace_cfg
experiment.create_rsl_rl_runner=make_runner
if __name__=='__main__':experiment.main()

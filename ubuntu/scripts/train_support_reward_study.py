"""Continue the validated 151-input PPO policy, keeping perception frozen."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import torch
from humanoid_sitting.support_reward_study import SittingRewardStudy
from humanoid_sitting.support_split_observation import install_split_linear
from humanoid_sitting.support_retention import RetentionPPO

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--resume151', required=True)
p.add_argument('--reward-arm', choices=['legacy', 'gentle'], required=True)
study, rest = p.parse_known_args(); sys.argv = [sys.argv[0], *rest]
import train_support_split_observation as split
experiment = split.experiment
objective = SittingRewardStudy(study.reward_arm)


class RewardStudyEnv(split.SplitObservationEnv):
    def reset(self):
        objective.reset()
        return super().reset()

    def _reset_rows(self, ids):
        objective.reset(ids)
        return super()._reset_rows(ids)


def study_cfg(**kwargs):
    cfg = split.split_cfg(**kwargs)
    out = Path(split.independent.args.output)
    manifest = json.loads((out / 'manifest.json').read_text())
    paths = [Path(__file__).resolve(), experiment.ROOT / 'sitting_task/src/humanoid_sitting/support_reward_study.py',
             experiment.ROOT / 'scripts/test_support_reward_study.py']
    for path in paths:
        shutil.copy2(path, out / 'source' / path.name)
        manifest['hashes'][str(path.relative_to(experiment.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['reward_study'] = dict(arm=study.reward_arm, resume151=study.resume151,
        resume_sha256=hashlib.sha256(Path(study.resume151).read_bytes()).hexdigest(),
        initialization='exact actor/critic continuation from151 checkpoint; fresh optimizer in both arms',
        peak_cost='gentle: -12 * increase in episode maximum squared excess force/bodyweight above1',
        leg_economy='gentle: 4*exp(-(leg_RMS_Nm/10)^2) per supported, stable, upright second',
        evaluation_reward='legacy arm for every evaluation; unchanged physical success criteria',
        perception='unchanged validated split interface and tracking')
    manifest['split_interface']['optimizer'] = 'fresh PPO; zero KL penalty; reward specified by reward_study'
    manifest['split_interface']['initialized_new_weights'] = 'continued unchanged from resume151 checkpoint'
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return cfg


def study_runner(env, agent, log_dir):
    assert split.options.interface == 'split'
    runner = split.original_runner(env, agent, log_dir=log_dir)
    install_split_linear(runner.alg.actor); install_split_linear(runner.alg.critic)
    parent = torch.load(study.resume151, map_location=env.device, weights_only=False)
    assert parent['actor_state_dict']['mlp.0.weight'].shape[1] == 151
    runner.alg.__class__ = RetentionPPO
    runner.alg.configure_retention(parent['actor_state_dict'], 0., freeze_normalization=True)
    if env.args.mode == 'train' or not env.args.checkpoint:
        assert not env.args.checkpoint, 'Training begins from --resume151 with a fresh optimizer'
        for role in ('actor', 'critic'):
            model = getattr(runner.alg, role); state = parent[f'{role}_state_dict']
            model.load_state_dict(state)
            assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
        prepared = Path(log_dir) / 'prepared.pt'; runner.save(str(prepared)); env.args.checkpoint = str(prepared)
    normalization = {role: {k: v.clone() for k, v in getattr(runner.alg, role).state_dict().items()
        if k.startswith('obs_normalizer.')} for role in ('actor', 'critic')}
    old_update = runner.alg.update; updates = 0; started = time.monotonic()
    def audited_update():
        nonlocal updates
        result = old_update(); updates += 1
        assert all(torch.isfinite(v).all() for role in ('actor', 'critic') for v in getattr(runner.alg, role).parameters())
        for role in ('actor', 'critic'):
            assert all(torch.equal(v, getattr(runner.alg, role).state_dict()[k]) for k, v in normalization[role].items())
        row = dict(update=updates, transitions=updates * agent.num_steps_per_env * env.num_envs,
            elapsed_training_s=time.monotonic()-started, losses=result,
            reward_contributions=objective.pop_totals(),
            completed_attempts=len(env.episodes),
            completed_successes=sum(x['success'] for x in env.episodes),
            new_distance_weight_max=float(runner.alg.actor.mlp[0].weight[:,144:].abs().max()))
        with (Path(log_dir)/'study_updates.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        temp=Path(log_dir)/'progress.tmp'; temp.write_text(json.dumps(row,indent=2)+'\n')
        temp.replace(Path(log_dir)/'progress.json')
        return result
    runner.alg.update = audited_update
    return runner


experiment.support_objective = objective
experiment.DeploymentEnvCfg = study_cfg
experiment.SupportVecEnv = RewardStudyEnv
experiment.create_rsl_rl_runner = study_runner
if __name__ == '__main__':
    experiment.main()

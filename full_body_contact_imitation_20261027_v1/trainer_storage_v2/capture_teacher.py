"""Observe frozen teachers at50Hz, without altering their commands or rewards.

Runs only after the comparison completes. Records PRE-action observations,
applied commands and POST-action physical labels for independent validation.
"""
import argparse
import runpy
import sys
from pathlib import Path
import numpy as np
import torch
from common import ROOT, OBS, read, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    args = p.parse_args()
    job = read(args.job)
    sys.path.insert(1, str(ROOT/'scripts'))
    sys.argv = [str(ROOT/'scripts'/job['entry'])]+job['arguments']
    import run_supported_transition as transition
    import run_supported_postures as support
    registry = []
    original_init = transition.TransitionEnv.__init__

    def initialize(env, *a, **kw):
        original_init(env, *a, **kw)
        raw = env.raw
        original_step = raw.step
        data = env.robot.data
        model = read(ROOT/'runs/g1_kinematics_geometry_04/model.json')
        assert env.robot.body_names == model['body_names'] and env.robot.joint_names == model['joint_names']
        np.testing.assert_array_equal(data.body_mass.torch[0].cpu().numpy(),np.asarray(model['body_mass'],np.float32))
        n = env.num_envs
        ids = env.ids
        previous = env.previous.detach().clone()
        history = None
        records = []
        original_reset = {k: env.resets[k][np.asarray(env.index), 0].copy() for k in ['q', 'qd', 'command', 'root_pose', 'root_velocity']}
        meta = dict(joint_names=env.robot.joint_names, body_names=env.robot.body_names,
                    geometry_indices=np.asarray(env.index).tolist(),
                    physics_steps_at_capture_install=raw.sim.get_physics_step_count(),
                    joint_ids=ids, task=job['task'], bank=job['bank'],
                    lower=data.joint_pos_limits.torch[0, ids, 0].cpu().tolist(),
                    upper=data.joint_pos_limits.torch[0, ids, 1].cpu().tolist())

        def step(command, *sa, **sk):
            nonlocal previous, history
            active = (env.warming == 0)
            # Existing teacher evaluations reset all rows together.
            assert bool(active.all()) or bool((~active).all())
            frame = torch.cat((data.joint_pos.torch[:, ids], data.joint_vel.torch[:, ids]/10,
                               data.applied_torque.torch[:, ids]/data.joint_effort_limits.torch[:, ids],
                               data.projected_gravity_b.torch, data.root_ang_vel_b.torch/3,
                               previous[:, ids], torch.as_tensor(env.geometry, device=env.device)), -1)
            assert frame.shape == (n, 111)
            history = frame[:, None].repeat(1, 3, 1) if history is None else torch.cat((history[:, 1:], frame[:, None]), 1)
            task = torch.tensor([[1., 0.] if job['task'] == 'sitting' else [0., 1.]], device=env.device).expand(n, -1)
            obs = torch.cat((history.flatten(1), task, env.episode_length_buf[:, None]/1000.), -1)
            assert obs.shape == (n, OBS) and torch.isfinite(obs).all()
            row = None
            if bool(active.all()):
                root = data.root_pose_w.torch.clone(); root[:, :3] -= raw.scene.env_origins
                row = dict(obs=obs.cpu().numpy().copy(), q_pre=data.joint_pos.torch.cpu().numpy().copy(),
                           qd=data.joint_vel.torch.cpu().numpy().copy(), root=root.cpu().numpy(),
                           root_velocity=data.root_vel_w.torch.cpu().numpy().copy(),
                           previous=previous.cpu().numpy().copy(), command=command.cpu().numpy().copy(),
                           tick=env.episode_length_buf.cpu().numpy().copy())
            # The exact caller-provided tensor is passed through unchanged.
            result = original_step(command, *sa, **sk)
            meta['physics_steps_at_last_control'] = raw.sim.get_physics_step_count()
            previous = command.detach().clone()
            if row is not None:
                m, d, flags = support.metrics(raw)
                loaded = m['force_w'].norm(dim=-1) > .01*m['bodyweight_N'][:, None]
                row.update(q_post=data.joint_pos.torch.cpu().numpy().copy(),
                           loaded_slip=torch.where(loaded, m['slip_m_s'], 0).max(1).values.cpu().numpy(),
                           force=m['force_w'].cpu().numpy(), slip=m['slip_m_s'].cpu().numpy(),
                           weight=m['bodyweight_N'].cpu().numpy(), legal=d['contacts_legal'].cpu().numpy(),
                           no_slip=d['no_slip'].cpu().numpy(), good=d['good'].cpu().numpy(),
                           failure=m['failure'].cpu().numpy(), speed=m['speed_m_s'].cpu().numpy(),
                           gyro=m['angular_speed_rad_s'].cpu().numpy(), support_bw=d['rest_normal_load_bw'].cpu().numpy(),
                           wrong_force=d['wrong_force_N'].cpu().numpy(), peak_bw=(d['peak_force_N']/m['bodyweight_N']).cpu().numpy(),
                           torque_squared=data.applied_torque.torch.square().sum(1).cpu().numpy(),
                           flags=torch.stack([flags[k] for k in ['fall', 'forbidden', 'torque', 'impact', 'joint_limit']], -1).cpu().numpy())
                records.append(row)
            return result

        raw.step = step
        registry.append((records, meta, original_reset))

    transition.TransitionEnv.__init__ = initialize
    runpy.run_path(str(ROOT/'scripts'/job['entry']), run_name='__main__')
    assert len(registry) == 1
    records, meta, reset = registry[0]
    assert len(records) == 1000, len(records)
    assert meta['physics_steps_at_capture_install'] <= 1000
    assert meta['physics_steps_at_last_control'] <= meta['physics_steps_at_capture_install']+4000
    out = Path(job['arguments'][job['arguments'].index('--output')+1])
    arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
    assert np.array_equal(arrays['tick'], np.arange(1000)[:, None]+np.zeros_like(arrays['tick']))
    for k, v in arrays.items():
        assert np.isfinite(v).all(), k
    np.savez_compressed(out/'demonstrations_50hz.npz', **arrays)
    np.savez_compressed(out/'original_reset.npz', **reset)
    write(out/'demonstration_capture.json', dict(**meta, samples=1000, pre_action_alignment=True,
          teacher_commands_modified=False, evaluation_labels_recomputed_at_every_control_sample=True,
          source_job=job, result='Recorded, not yet admitted; independent audit required'))


if __name__ == '__main__':
    main()

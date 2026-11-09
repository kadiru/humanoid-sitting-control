"""Original physical evaluator, with latched sensor input and potential reward.

Demonstrations form an explicit frozen motion prior in the policy. No case ID,
contact truth, simulator root pose or geometry-bank truth enters that selection.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from common import ROOT, DT, STEP, OBS, write, read
sys.path.insert(1, str(ROOT/'scripts'))
import run_resting_posture as base
import run_supported_postures as support
from run_supported_transition import TransitionEnv
from humanoid_sitting.supported_surface import SupportTracker
from policy import command as apply_action
from rewards import staged_reward, phase_from_clock


class FullBodyEnv:
    _sensor_frame = TransitionEnv._sensor_frame
    _observe = TransitionEnv._observe

    def __init__(self, raw, args, out, bank, metadata, reset_valid=None):
        self.raw = raw; self.cfg = raw.cfg; self.device = raw.device; self.num_envs = raw.num_envs
        self.args = args; self.out = out; self.bank = bank; self.metadata = metadata
        self.robot = raw.scene['robot']; self.ids = metadata['joint_ids']
        assert [self.robot.joint_names[i] for i in self.ids] == base.BODY_JOINT_NAMES
        self.index = np.arange(self.num_envs)
        assert np.array_equal(base.furniture.indices(raw).cpu().numpy(), self.index)
        self.model = base.BodyModel(base.MODEL)
        frozen_model = read(base.MODEL)
        assert self.robot.body_names == frozen_model['body_names'] and self.robot.joint_names == frozen_model['joint_names']
        np.testing.assert_array_equal(self.robot.data.body_mass.torch[0].cpu().numpy(),np.asarray(frozen_model['body_mass'],np.float32))
        self.trackers = [SupportTracker(self.model) for _ in self.index]
        self.geometry = np.zeros((self.num_envs, 13), np.float32); self.geometry[:, 12] = 1
        self.t = 0; self.camera_count = 0; self.audit = []; self.hist = None; self.obs = None
        self.episodes = []; self.episode_records = []; self.saved_episodes = []
        self.active_transitions = self.warmup_transitions = self.simulator_control_steps = 0
        self.initial_settling_transitions = 0; self.attempt = 0
        self.phase_start = torch.zeros(self.num_envs, device=self.device)
        self.reset_valid = np.zeros((self.num_envs, 4), bool) if reset_valid is None else reset_valid
        self.reset_valid[:, 0] = True
        self.reference = torch.as_tensor(bank['reference'], device=self.device)
        self.q_reference = torch.as_tensor(bank['q_post'][:, :, self.ids], device=self.device)
        self.phase_times = torch.as_tensor(bank['phase_times'], device=self.device)
        self.admitted = np.asarray(bank['admitted'], bool)
        self.limit_lower = self.robot.data.joint_pos_limits.torch[:, self.ids, 0]
        self.limit_upper = self.robot.data.joint_pos_limits.torch[:, self.ids, 1]
        np.testing.assert_allclose(self.limit_lower[0].cpu(), metadata['lower'], rtol=0, atol=0)
        np.testing.assert_allclose(self.limit_upper[0].cpu(), metadata['upper'], rtol=0, atol=0)
        self.raw.reset()
        self.initial_physics_steps = self.raw.sim.get_physics_step_count()
        assert self.initial_physics_steps <= 1000, 'INITIALIZATION_PHYSICS_BUDGET'
        self.previous = self.robot.data.joint_pos.torch.clone()
        self.startup = self.reset(0)

    def reset(self, slot):
        """Synchronous reset. All outcomes are measured after explicit warmup."""
        n = self.num_envs
        chosen = np.full(n, slot, int)
        chosen[~self.admitted | ~self.reset_valid[:, slot]] = 0
        # Twenty percent standing-start exposure throughout the curriculum.
        if self.args.mode == 'train' and slot:
            chosen[(self.index+self.attempt) % 5 == 0] = 0
        times = np.array([0., 6., 10., 15.])[chosen]
        states = {}
        source = {'q':'q_pre', 'qd':'qd', 'command':'previous', 'root_pose':'root', 'root_velocity':'root_velocity'}
        for k, field in source.items():
            a = self.bank['reset_'+k].copy()
            for i in np.flatnonzero(chosen):
                a[i] = self.bank[field][round(times[i]/DT), i]
            states[k] = a
        support.restore(self.raw, states)
        self.previous = torch.as_tensor(states['command'], dtype=torch.float32, device=self.device).clone()
        self.fixed_fingers = self.previous[:, 23:].clone()
        self.phase_start = torch.as_tensor(times, device=self.device, dtype=torch.float32)
        self.episode_length_buf = torch.zeros(n, device=self.device, dtype=torch.long)
        self.alive = torch.ones(n, device=self.device, dtype=torch.bool)
        self.held = torch.zeros(n, device=self.device, dtype=torch.long)
        self.slip_ticks = self.held.clone(); self.illegal_ticks = self.held.clone()
        self.flags = torch.zeros((n, 5), device=self.device, dtype=torch.bool)
        self.hist = None
        self.geometry[:] = 0; self.geometry[:, 12] = 1
        for tracker in self.trackers: tracker.reset()
        warm_failure = torch.zeros(n, device=self.device, dtype=torch.bool)
        for _ in range(25):
            self.raw.step(self.previous); self.simulator_control_steps += 1
            self.warmup_transitions += n
            m, _, _ = support.metrics(self.raw); warm_failure |= m['failure']
        self._observe(force=True)
        self.initial_sensor=self.hist[:,-1].clone()
        self.previous_potential=torch.zeros(n,device=self.device)
        self.attempt += 1
        self.episode_records = []
        # A bad warmup is not silently erased at the start of an episode.
        self.alive &= ~warm_failure
        return dict(chosen_slots=chosen.tolist(), warm_failure=warm_failure.cpu().tolist(),
                    sensor_valid=(self.geometry[:, 11] > 0).tolist())

    def observation(self):
        task = torch.tensor([[1., 0.] if self.metadata['task'] == 'sitting' else [0., 1.]], device=self.device).expand(self.num_envs, -1)
        clock = ((self.phase_start+self.episode_length_buf*DT)/20).clamp_max(1)[:, None]
        obs = torch.cat((self.hist.flatten(1), task, clock, self.initial_sensor), -1)
        assert obs.shape == (self.num_envs, OBS) and torch.isfinite(obs).all()
        return obs

    def step(self, latent, teacher_command=None, record=False, imitation_weight=1.):
        before = self.alive.clone()
        prior = self.previous[:, self.ids].clone()
        if teacher_command is None:
            applied = apply_action(prior, latent, self.limit_lower, self.limit_upper)
        else:
            applied = teacher_command[:, self.ids]
        applied = torch.where(before[:, None], applied, prior)
        self.previous[:, self.ids] = applied
        assert torch.isfinite(self.previous).all()
        if teacher_command is None:
            # Addition rounding at native float32 precision is recorded, not a
            # changed physical success threshold.
            assert ((applied-prior).abs() <= STEP+2e-7).all()
            assert (applied >= self.limit_lower+.03-2e-7).all()
            assert (applied <= self.limit_upper-.03+2e-7).all()
        assert torch.equal(self.previous[:, 23:], self.fixed_fingers)
        self.raw.step(self.previous); self.simulator_control_steps += 1
        assert not self.raw.reset_buf.any()
        self.t += 1; self.episode_length_buf += 1; self.active_transitions += self.num_envs
        m, d, flags = support.metrics(self.raw)
        first = m['failure'] & before
        self.alive &= ~m['failure']
        self.flags |= torch.stack([flags[k] for k in ['fall', 'forbidden', 'torque', 'impact', 'joint_limit']], -1)
        good = d['good'] & self.alive
        self.held = torch.where(good, self.held+1, 0)
        self.slip_ticks += (~d['no_slip']).long()
        self.illegal_ticks += (~d['contacts_legal']).long()
        loaded = m['force_w'].norm(dim=-1) > .01*m['bodyweight_N'][:, None]
        slip = torch.where(loaded, m['slip_m_s'], 0).max(1).values
        horizon = self.episode_length_buf >= 1000
        stable = horizon & self.alive & (self.held >= 250)
        clean = stable & (self.slip_ticks == 0) & (self.illegal_ticks == 0)
        seconds = (self.phase_start+self.episode_length_buf*DT).clamp_max(20)
        ref_tick = (seconds/DT).round().long().clamp(1, 1000)-1
        q_target = self.q_reference[ref_tick, self.reference]
        q_error = ((self.robot.data.joint_pos.torch[:, self.ids]-q_target)/.25).square().mean(-1)
        phase = phase_from_clock(seconds, self.phase_times[:, 0], self.phase_times[:, 1])
        values = dict(alive=self.alive, legal=d['contacts_legal'], no_slip=d['no_slip'], good=good,
                      phase=phase, q_error=q_error, speed=m['speed_m_s'], gyro=m['angular_speed_rad_s'],
                      loaded_slip=slip, wrong_force=d['wrong_force_N'], peak_bw=d['peak_force_N']/m['bodyweight_N'],
                      normalized_command_delta=(applied-prior)/STEP, support_bw=d['rest_normal_load_bw'],
                      held_ticks=self.held, torque_squared=self.robot.data.applied_torque.torch.square().sum(1),
                      clean_completion=clean, stable_completion=stable, first_failure=first)
        values.update(horizon=horizon,previous_potential=self.previous_potential.clone())
        reward, terms = staged_reward(values, imitation_weight)
        self.previous_potential=terms['potential'].detach().clone()
        reward = torch.where(before, reward, 0.)
        # Successful-only completion/effort rewards never revive a failed row.
        self._observe()
        if record:
            self.episode_records.append({**{k:v.detach().cpu().numpy().copy() for k,v in values.items()},
                                         'reward':reward.cpu().numpy(), 'learn':before.cpu().numpy(),
                                         'force':m['force_w'].cpu().numpy(), 'slip':m['slip_m_s'].cpu().numpy(),
                                         'weight':m['bodyweight_N'].cpu().numpy(), 'failure':m['failure'].cpu().numpy(),
                                         'flags':self.flags.cpu().numpy().copy(), 'q':self.robot.data.joint_pos.torch.cpu().numpy(),
                                         'body':self.robot.data.body_pose_w.torch.cpu().numpy().copy(),
                                         'command':self.previous.cpu().numpy().copy()})
            self.episode_records[-1].update({'physical_'+k:v.detach().cpu().numpy().copy() for k,v in m.items()})
            self.episode_records[-1]['root_height'] = d['root_height_m'].cpu().numpy().copy()
        if horizon.all():
            rows = []
            for i in range(self.num_envs):
                rows.append(dict(case=i, id=self.metadata['ids'][i], attempt=self.attempt,
                    start_time_s=float(self.phase_start[i]), steps=1000, duration_s=20.,
                    stable=bool(stable[i]), clean=bool(clean[i]), final_hold_s=float(self.held[i])*DT,
                    slip_ticks=int(self.slip_ticks[i]), illegal_ticks=int(self.illegal_ticks[i]),
                    physical_failures=self.flags[i].cpu().tolist(),
                    full_standing_start=bool(self.phase_start[i] == 0)))
            self.episodes += rows
            with (self.out/'episodes.jsonl').open('a') as f:
                import json
                for row in rows: f.write(json.dumps(row)+'\n')
        return self.observation(), reward, first | horizon, before, bool(horizon.all())

    def save_records(self, path):
        assert len(self.episode_records) == 1000
        np.savez_compressed(path, **{k:np.stack([r[k] for r in self.episode_records]) for k in self.episode_records[0]})

    def close(self):
        physical_steps = self.raw.sim.get_physics_step_count()
        assert physical_steps <= self.initial_physics_steps+4*self.simulator_control_steps
        write(self.out/'results.json', dict(task=self.metadata['task'], episodes=self.episodes,
             active_transitions=self.active_transitions, warmup_transitions=self.warmup_transitions,
             simulator_control_steps=self.simulator_control_steps, physics_substeps_per_control=4,
             initialization_physics_steps=self.initial_physics_steps, actual_total_physics_steps=physical_steps,
             body_names=self.robot.body_names, joint_names=self.robot.joint_names, origins=self.raw.scene.env_origins.cpu().tolist(),
             physical_criteria_source=str(Path(support.__file__).resolve()), success_criteria_unchanged=True))
        write(self.out/'sensor_audit.json', dict(actor_channels=OBS, body_joints=23, camera_frames=self.camera_count,
             actor_has_simulator_truth=False, actor_uses_frozen_demonstration_prior=True, prior_selected_by_initial_sensors=True, camera_extra_physics_steps=0,
             truth_validation=self.audit))
        self.raw.close()

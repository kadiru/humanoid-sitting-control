"""Development probe: keep ankle balance until sensor-estimated support.

The old bridge moves balance authority to the hips by 11.5 seconds even when
the pelvis is still airborne. This probe changes that blend only on the
measured-plane-selected normal-descent rows. It does not change physical labels.
"""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from low_seat_normal_controller import install as install_normal


def install(harness, config):
    install_normal(harness, config)
    Parent = harness.FeedbackEnv

    class BalanceEnv(Parent):
        def __init__(self, *a, **kw):
            self.balance_ready = False
            self.balance_records = []
            super().__init__(*a, **kw)
            self.balance_fraction = torch.zeros(self.num_envs, device=self.device)
            self.balance_tick = -1
            self.balance_ready = True

        def _reset_rows(self, ids):
            if self.balance_ready:
                self.balance_fraction[ids] = 0
                self.balance_tick = -1
            return super()._reset_rows(ids)

        def reference(self):
            ref = super().reference()
            if not self.balance_ready or not self.planning_enabled:
                return ref
            elapsed = (self.episode_length_buf-self.near_start).float()*.02
            u = (elapsed/5*24).clamp(0, 24)
            lo = u.floor().long().clamp_max(23)
            f = u-lo
            ids = torch.arange(self.num_envs, device=self.device)
            target = (1-f[:, None])*self.target_paths[ids, lo]+f[:, None]*self.target_paths[ids, lo+1]
            velocity = (self.target_paths[ids, lo+1, 1]-self.target_paths[ids, lo, 1])*24/5*(u<24)
            gravity = self.robot.data.projected_gravity_b.torch
            pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
            correction = .6*(pitch-target[:, 1])+.15*(self.robot.data.root_ang_vel_b.torch[:, 1]-velocity)
            if self.estimated_root_x is not None:
                correction += self.estimated_root_x-target[:, 0]
            correction = correction.clamp(-.25, .25)*(elapsed/.2).clamp(0, 1)*self.target_ready*(self.near_start>=0)
            old = ((elapsed-.5)/1.).clamp(0, 1)
            old = old.square()*(3-2*old)
            selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for i in self.landing_states:
                selected[i] = True
            selected &= (self.near_start>=0) & (self.warming==0)
            if self.t % 5 == 0 and self.balance_tick != self.t:
                self.balance_tick = self.t
                observed = 1-self.filtered_wrench[:, :, 1].sum(1)/self.wrench_observer.weight
                condition = np.max(self.shear_records[-1]['condition'], axis=1)<200
                desired = np.clip((observed-.05)/.30, 0, 1)*condition
                if config['balance_mode'] == 'ankle_only':
                    desired[:] = 0
                desired = torch.tensor(desired, device=self.device, dtype=torch.float32)
                self.balance_fraction += (desired-self.balance_fraction).clamp(-.04, .04)
                self.balance_fraction[~selected] = 0
                self.balance_records.append(dict(tick=self.t, observed_share=observed.tolist(),
                    condition_valid=condition.tolist(), selected=selected.cpu().tolist(),
                    fraction=self.balance_fraction.cpu().tolist()))
            delta = (self.balance_fraction-old)*selected
            ref[:, self.hip_pitch_ids] += correction[:, None]*delta[:, None]
            ref[:, self.ankle_ids] -= correction[:, None]*delta[:, None]
            ref[:, self.ankle_ids] += (-3*self.foot_pitch).clamp(-.15, .15)*delta[:, None]
            return ref

        def close(self):
            super().close()
            (self.out/'balance_feedback_audit.json').write_text(json.dumps(self.balance_records))
            shutil.copy2(__file__, self.out/'source'/Path(__file__).name)

    harness.FeedbackEnv = BalanceEnv

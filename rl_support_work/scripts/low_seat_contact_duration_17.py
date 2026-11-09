"""Encoder/IMU landing damping and a separately tested clearance candidate.

No contact labels or live simulator poses enter this controller. Foot anchors
are kinematic predictions established at the measured-plane planning instant.
The fixed-foot assumption must be checked against physical foot-slip labels.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
from positive_slope_planner_v2 import plan as previous
from positive_slope_planner import plan as changed

def selected(a, cfg):
    n = np.asarray(a['normal'])
    height = (n[0] * .4 - a['offset']) / n[2]
    slope = np.rad2deg(np.arctan2(n[0], n[2]))
    return bool((height < cfg['max_height'] and slope > cfg['min_slope']) or
                (height < .415 and slope > 18.))

def plan(*args, inertial, config):
    c, q, a = previous(*args, inertial=inertial, target_x=-.18,
        target_pitch=.25, contact_gap=.001, land_fraction=.6, travel_fraction=1.)
    a['landing_repair_applied'] = selected(a, config)
    if not a['landing_repair_applied']:
        return c, q, a
    if config['path'] == 'forward_clearance':
        c, q, a = changed(*args, inertial=inertial, target_x=config['target_x'],
            target_pitch=config['target_pitch'], contact_gap=config['gap'],
            land_fraction=config['land_fraction'], travel_fraction=config['travel_fraction'])
        a['landing_repair_applied'] = True
    a['landing_config'] = config
    a['landing_scope'] = 'Kinematic screen and sensor-only damping; physical validation pending'
    return c, q, a

def install(harness, config):
    bridge = harness.parent_harness.traction.bridge
    bridge.progressive.rolling_plan = lambda *a, **kw: plan(
        *a, **kw, inertial=bridge.inertial, config=config)
    Parent = harness.FeedbackEnv

    class LandingEnv(Parent):
        def __init__(self, *args, **kwargs):
            self.landing_ready = False
            self.landing_records = []
            super().__init__(*args, **kwargs)
            self.landing_cursor = 0
            self.landing_states = {}
            self.landing_correction = torch.zeros(self.num_envs, 37, device=self.device)
            self.landing_tick = -1
            self.landing_ready = True

        def _reset_rows(self, ids):
            if self.landing_ready:
                for i in ids.cpu().tolist():
                    self.landing_states.pop(i, None)
                self.landing_correction[ids] = 0
                self.landing_tick = -1
            return super()._reset_rows(ids)

        def reference(self):
            ref = super().reference()
            if not self.landing_ready or not self.planning_enabled:
                return ref
            if self.t % 5 == 0 and self.landing_tick != self.t:
                self.landing_tick = self.t
                q = self.robot.data.joint_pos.torch.cpu().numpy()
                gravity = self.robot.data.projected_gravity_b.torch.cpu().numpy()
                for a in self.plan_audit[self.landing_cursor:]:
                    i = a['case']
                    if not a.get('landing_repair_applied', False):
                        continue
                    R = level_rotation(gravity[i]); fk = self.model.fk(q[i])
                    origin = np.asarray(a['origin'])
                    feet = np.array([R @ fk[s+'_ankle_roll_link'][:3,3] + origin
                                     for s in ['left','right']])
                    self.landing_states[i] = dict(anchor=feet, roots=np.asarray(a['root_path']),
                        previous=origin.copy(), velocity=np.zeros(3), tick=self.t,
                        start=a['tick'], config=config)
                self.landing_cursor = len(self.plan_audit)
                correction = np.zeros((self.num_envs,37))
                records = []
                for i, state in self.landing_states.items():
                    if self.warming[i] or self.near_start[i] < 0:
                        continue
                    R = level_rotation(gravity[i]); fk = self.model.fk(q[i])
                    relative = np.array([R @ fk[s+'_ankle_roll_link'][:3,3]
                                         for s in ['left','right']])
                    position = (state['anchor'] - relative).mean(0)
                    dt = (self.t-state['tick']) * .02
                    if dt > 0:
                        state['velocity'] = .5*state['velocity'] + .5*(position-state['previous'])/dt
                    state['previous'] = position.copy(); state['tick'] = self.t
                    elapsed = (int(self.episode_length_buf[i])-state['start'])*.02
                    u = np.clip(elapsed / float(getattr(self, "repair_duration", [5.]*self.num_envs)[i]) * 24, 0, 24)
                    lo = min(int(u),23); f = u-lo
                    target = (1-f)*state['roots'][lo,:3] + f*state['roots'][lo+1,:3]
                    desired_velocity = (state['roots'][lo+1,:3]-state['roots'][lo,:3])*24/float(getattr(self, "repair_duration", [5.]*self.num_envs)[i])*(u<24)
                    delta = np.array([config['kp_x'],0,config['kp_z']])*(target-position)
                    delta += np.array([config['kd_x'],0,config['kd_z']])*(desired_velocity-state['velocity'])
                    delta = np.clip(delta,-config['translation_cap'],config['translation_cap'])
                    ramp = np.clip(elapsed/.5,0,1); ramp = ramp*ramp*(3-2*ramp)
                    for side in ['left','right']:
                        ids = [self.model.names.index(side+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']]
                        J=[]; eps=1.e-5
                        for j in ids:
                            values=[]
                            for sign in [-1,1]:
                                trial=q[i].astype(float).copy();trial[j]+=sign*eps
                                foot=self.model.fk(trial)[side+'_ankle_roll_link']
                                values.append(np.r_[(R@foot[:3,3])[[0,2]],
                                    Rotation.from_matrix(R@foot[:3,:3]).as_euler('xyz')[1]])
                            J.append((values[1]-values[0])/(2*eps))
                        J=np.asarray(J).T
                        shift=np.linalg.solve(J,np.array([-delta[0],-delta[2],0.]))
                        correction[i,ids]=np.clip(shift,-config['joint_cap'],config['joint_cap'])*ramp
                    records.append(dict(case=i,position=position.tolist(),target=target.tolist(),
                        velocity=state['velocity'].tolist(),delta=delta.tolist()))
                self.landing_correction=torch.tensor(correction,device=self.device,dtype=torch.float32)
                self.landing_records.append(dict(tick=self.t,states=records,correction=correction.tolist()))
            return ref+self.landing_correction

        def _physics_step(self, actions):
            result=super()._physics_step(actions)
            # Diagnostic truth ONLY: neither field is consumed above.
            contacts=self.raw.scene['all_contacts']
            self.contact_records[-1].update(
                unexpected_body_normal=contacts.data.net_normal_forces_w_history.torch.norm(dim=-1).max(1).values.cpu().numpy(),
                contact_position=np.stack([self.raw.scene[s].data.contact_pos_w.torch[:,0,0].cpu().numpy()
                    for s,b,r in harness.old.support.CONTACTS],1)-self.raw.scene.env_origins.cpu().numpy()[:,None,:])
            return result

        def close(self):
            import json, shutil
            from pathlib import Path
            names=list(self.raw.scene['all_contacts'].body_names)
            super().close()
            (self.out/'landing_controller_audit.json').write_text(json.dumps(self.landing_records))
            (self.out/'contact_body_names.json').write_text(json.dumps(names))
            (self.out/'landing_config.json').write_text(json.dumps(config,indent=2))
            shutil.copy2(__file__,self.out/'source'/Path(__file__).name)

    harness.FeedbackEnv=LandingEnv

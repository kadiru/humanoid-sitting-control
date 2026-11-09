"""Second resting interface: bounded reference offsets, preserved physical labels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import run_resting_posture as base
from humanoid_sitting.resting_bounded_control import bounded_command, MeasuredRestHold

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--hold-goal',action='store_true')
p.add_argument('--constant-action',type=float)
p.add_argument('--reset-cycles',type=int,default=10)
options,remaining=p.parse_known_args()
sys.argv=[sys.argv[0],*remaining]
OriginalEnv=base.RestEnv


class BoundedEnv(OriginalEnv):
    def __init__(self,*args,**kwargs):
        self.reference=None;self.hold_gate=None;self.needs_calibration=None
        self.hold_trace=[]
        super().__init__(*args,**kwargs)
        if options.hold_goal and self.args.mode=='train':
            raise ValueError('The optional hold rule is an evaluation ablation, not a training modification')
        self.span=torch.tensor([.025 if 'roll_joint' in name and ('hip' in name or 'ankle' in name)
                                else .05 for name in base.BODY_JOINT_NAMES],device=self.device)
        paths=[Path(__file__).resolve(),base.ROOT/'sitting_task/src/humanoid_sitting/resting_bounded_control.py']
        for source in paths:shutil.copy2(source,self.out/'source'/source.name)
        manifest=json.loads((self.out/'manifest.json').read_text())
        manifest.update(action='Bounded offsets from retained starting command: .025rad hip/ankle roll, .05rad others; alpha .2, slew .002rad/tick; physical margins .03rad',
            actor='393 sensor-only channels: original324 plus starting-command reference23 per frame',
            bounded_options=vars(options),bounded_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths})
        base.write_json(self.out/'manifest.json',manifest)

    def _augment(self):
        if self.hist is not None:
            refs=self.reference[:,self.ids,None].transpose(1,2).expand(-1,3,-1)
            self.obs['policy']=torch.cat((self.hist,refs),dim=-1).flatten(1)
            assert self.obs['policy'].shape==(self.num_envs,393)

    def _observe(self,force_depth=False):
        super()._observe(force_depth)
        self._augment()

    def _reset_rows(self,ids,initial=False):
        if self.reference is None:
            self.reference=torch.as_tensor(self.bank['command'],device=self.device).clone()
            self.needs_calibration=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.reference[ids]=torch.as_tensor(self.bank['command'],device=self.device)[ids]
        self.needs_calibration[ids]=True
        if self.hold_gate is not None:
            self.hold_gate.held[ids]=False;self.hold_gate.ticks[ids]=0
        super()._reset_rows(ids,initial)
        self._augment()

    def _calibrate_hold(self,ids):
        d=self.robot.data
        effort=(d.applied_torque.torch/d.joint_effort_limits.torch).square().mean(1)
        self.hold_gate.reset(ids,effort[ids],d.projected_gravity_b.torch[ids])
        self.needs_calibration[ids]=False

    def reset(self):
        result=super().reset()
        self.hold_gate=MeasuredRestHold(self.num_envs,self.device)
        self._calibrate_hold(torch.arange(self.num_envs,device=self.device))
        return result

    def step(self,actions):
        with torch.inference_mode():
            if options.constant_action is not None:
                actions=torch.full_like(actions,options.constant_action)
            d=self.robot.data;lim=d.joint_pos_limits.torch[:,self.ids]
            desired=bounded_command(self.reference[:,self.ids],self.previous[:,self.ids],actions,
                                    lim[...,0],lim[...,1],self.span)
            if options.hold_goal:
                effort=(d.applied_torque.torch/d.joint_effort_limits.torch).square().mean(1)
                geometry=torch.as_tensor(self.geometry,device=self.device)
                held=self.hold_gate.update(effort,d.joint_vel.torch[:,self.ids],d.root_ang_vel_b.torch,
                    d.projected_gravity_b.torch,geometry[:,0],geometry[:,8]>0,geometry[:,9])
                desired=torch.where(held[:,None],self.previous[:,self.ids],desired)
            inactive=(self.warming>0)|~self.alive
            desired[inactive]=self.previous[inactive][:,self.ids]
            equivalent_increment=(desired-self.previous[:,self.ids])/.01
            result=super().step(equivalent_increment)
            ids=(self.needs_calibration&(self.warming==0)).nonzero().flatten()
            if len(ids):self._calibrate_hold(ids)
            assert bool(((self.previous[:,self.ids]-self.reference[:,self.ids]).abs()<=self.span+1.e-6).all())
            if self.args.mode!='train' and self.t%5==0:
                self.hold_trace.append(dict(t=self.t*.02,held=self.hold_gate.held.cpu().tolist(),
                    effort=self.hold_gate.filtered_effort.cpu().tolist(),reference_effort=self.hold_gate.reference_effort.cpu().tolist()))
            return result

    @torch.inference_mode()
    def reset_probe(self):
        super().reset_probe()
        rows=[]
        for cycle in range(options.reset_cycles):
            ids=torch.tensor([cycle%self.num_envs],device=self.device)
            self._reset_rows(ids)
            for _ in range(25):self.step(torch.zeros(self.num_envs,23,device=self.device))
            m,d,f=base.metrics(self.raw)
            row=dict(cycle=cycle,case=int(ids[0]),good=bool(d['good'][ids].all()),
                failure=bool(m['failure'][ids].any()),wrong_force_N=float(d['wrong_force_N'][ids].max()),
                command_error=float((self.previous[ids]-self.reference[ids]).abs().max()))
            rows.append(row)
            assert row['good'] and not row['failure'] and row['command_error']==0.,row
        probe=json.loads((self.out/'reset_probe.json').read_text())
        probe.update(repeated_settled_resets=rows,passed=True)
        base.write_json(self.out/'reset_probe.json',probe)

    def close(self):
        base.write_json(self.out/'hold_goal.json',dict(enabled=options.hold_goal,target_effort_ratio=.9,
            dwell_s=2.,source='measured joint torque, encoders, gyro, gravity, depth clearance/validity/age; no simulator contact input',
            entries=self.hold_gate.entries.cpu().tolist(),releases=self.hold_gate.releases.cpu().tolist(),trace=self.hold_trace))
        super().close()
        audit=json.loads((self.out/'sensor_audit.json').read_text())
        audit.update(actor_channels=393,starting_reference='remembered command, not ground-truth body or surface pose')
        base.write_json(self.out/'sensor_audit.json',audit)


base.RestEnv=BoundedEnv
if __name__=='__main__':base.main()

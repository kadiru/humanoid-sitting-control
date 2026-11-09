"""Static intermediate-pose validation, including unsupported upright poses.

No physical failure label is relaxed. Supported endpoint success is reported
separately from stationary intermediate-pose success.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
import run_resting_posture as base
import run_supported_postures as support

p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True)
p.add_argument('--settle',type=float,default=4.);p.add_argument('--seconds',type=float,default=20.)
base.add_launcher_args(p);a=p.parse_args();a.seed=54801;a.mode='fixed';a.depth_check=False
out=Path(a.output);out.mkdir(exist_ok=False)
bank=dict(np.load(Path(a.bank)/'candidates.npz'));geometry=json.loads((Path(a.bank)/'geometry.json').read_text());a.num_envs=len(geometry)
base.write_json(out/'geometry.json',geometry);shutil.copy2(__file__,out/'source.py')
cfg=support.make_cfg(a,geometry);cfg.scene.rest_camera=None
with base.launch_simulation(cfg,a):
 raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;raw.reset();support.restore(raw,bank)
 r=raw.scene['robot'];cmd=torch.tensor(bank['command'],device=raw.device,dtype=torch.float32)
 flags={k:torch.zeros(raw.num_envs,device=raw.device,dtype=torch.bool) for k in ['fall','forbidden','torque','impact','joint_limit']}
 stable_ticks=torch.zeros(raw.num_envs,device=raw.device);good_ticks=stable_ticks.clone();held=stable_ticks.clone();sums={};traces=[];captured=None
 first=[];latched=torch.zeros_like(flags['fall']);steps=round(a.seconds/.02);settle=round(a.settle/.02)
 with torch.inference_mode():
  for step in range(settle+steps):
   raw.step(cmd);assert not raw.reset_buf.any()
   m,d,f=support.metrics(raw)
   for k,v in f.items():flags[k]|=v
   new=m['failure']&~latched
   for i in new.nonzero().flatten().tolist():first.append(dict(case=i,t=(step+1)*.02,flags={k:bool(v[i]) for k,v in f.items()},wrong_force=float(d['wrong_force_N'][i])))
   latched|=m['failure']
   stationary=(~latched)&d['contacts_legal']&d['traction_ok']&d['no_slip']&(m['speed_m_s']<=.10)&(m['angular_speed_rad_s']<=.25)&(m['trunk_tilt_rad']<=np.deg2rad(40))
   if step==settle-1:
    pose=r.data.root_pose_w.torch.clone();pose[:,:3]-=raw.scene.env_origins
    captured=dict(q=r.data.joint_pos.torch.cpu().numpy().copy(),qd=r.data.joint_vel.torch.cpu().numpy().copy(),command=cmd.cpu().numpy().copy(),root_pose=pose.cpu().numpy().copy(),root_velocity=r.data.root_vel_w.torch.cpu().numpy().copy())
   if step>=settle:
    stable_ticks+=stationary;good_ticks+=d['good']&~latched;held=torch.where(stationary,held+1,torch.zeros_like(held))
    for k,v in dict(total_squared_torque_Nm2=r.data.applied_torque.torch.square().sum(1),joint_rms_Nm=r.data.applied_torque.torch.square().mean(1).sqrt(),support_load_bw=d['rest_normal_load_bw'],foot_load_bw=d['foot_load_bw'],speed=m['speed_m_s'],slip=d['max_slip_m_s']).items():
     if k not in sums:sums[k]=torch.zeros(raw.num_envs,device=raw.device)
     sums[k]+=v
   if step%10==9:traces.append(dict(t=(step+1)*.02,root=r.data.root_pose_w.torch.cpu().numpy().copy(),q=r.data.joint_pos.torch.cpu().numpy().copy(),body=r.data.body_pose_w.torch.cpu().numpy().copy(),stable=stationary.cpu().numpy(),good=d['good'].cpu().numpy(),support=d['rest_normal_load_bw'].cpu().numpy()))
   if step%250==249:print('SECONDS',(step+1)*.02,'physical_safe',int((~latched).sum()),flush=True)
  stationary_ok=(stable_ticks>=.95*steps)&(held>=100)&~latched
  rows=[dict(case=i,geometry=g,valid=bool(stationary_ok[i]),supported=bool(stationary_ok[i] and good_ticks[i]>=.95*steps),stable_fraction=float(stable_ticks[i]/steps),support_fraction=float(good_ticks[i]/steps),final_hold_s=float(held[i]*.02),failures={k:bool(v[i]) for k,v in flags.items()},metrics={k:float(v[i]/steps) for k,v in sums.items()}) for i,g in enumerate(geometry)]
  np.savez_compressed(out/'candidates.npz',**captured)
  np.savez_compressed(out/'trace.npz',**{k:np.stack([x[k] for x in traces]) for k in traces[0]})
  base.write_json(out/'results.json',dict(rows=rows,stationary_valid=int(stationary_ok.sum()),total=len(rows),seconds=a.seconds,settle=a.settle,body_names=r.body_names,joint_names=r.joint_names,origins=raw.scene.env_origins.cpu().tolist()))
  base.write_json(out/'first_failures.json',first)
  print('VALID',int(stationary_ok.sum()),len(rows),flush=True)
 raw.close()

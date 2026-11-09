"""Read-only instrumentation of the preserved reference controller.

All extra contact/COM quantities are offline diagnostics, not policy inputs.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
import run_two_surface_path as original

original_metrics=original.support.metrics
records=[]
def metrics(raw):
    m,d,f=original_metrics(raw)
    normal=(m['force_w']*m['normal_w']).sum(-1)
    tangent=(m['force_w']-normal[...,None]*m['normal_w']).norm(dim=-1)
    utilization=tangent/(.6*normal.clamp_min(1e-7))
    active=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None]
    robot=raw.scene['robot'];data=robot.data;weight=data.body_mass.torch
    com=(data.body_com_pos_w.torch*weight[...,None]).sum(1)/weight.sum(1)[:,None]
    points=torch.stack([raw.scene[s].data.contact_pos_w.torch[:,0,0] for s,b,r in original.support.CONTACTS],1)
    contact=raw.scene['all_contacts'].data.net_normal_forces_w_history.torch.norm(dim=-1).max(1).values
    values=dict(force=m['force_w'],normal=normal,tangent=tangent,utilization=utilization,active=active,inside=m['inside_patch'],slip=m['slip_m_s'],point=points,com=com-raw.scene.env_origins,root=data.root_pose_w.torch,body_forces=contact,peak=d['peak_force_N']/m['bodyweight_N'],wrong=d['wrong_force_N'],foot=d['foot_load_bw'],support=d['rest_normal_load_bw'],vertical_support=d['rest_vertical_load_bw'],good=d['good'])
    records.append({k:v.cpu().numpy().copy() for k,v in values.items()})
    return m,d,f

close=original.PathEnv.close
def close_with_diagnostics(self):
    np.savez_compressed(self.out/'contact_diagnostics.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
    p=Path(__file__);shutil.copy2(p,self.out/'contact_probe_source.py')
    original.base.write_json(self.out/'contact_probe_manifest.json',dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source='Unmodified original support.metrics return values; extra quantities are offline only.',sample_period_s=.02,patches=original.support.CONTACTS,body_names=self.raw.scene['all_contacts'].body_names))
    close(self)

original.support.metrics=metrics
original.PathEnv.close=close_with_diagnostics
if __name__=='__main__':original.main()

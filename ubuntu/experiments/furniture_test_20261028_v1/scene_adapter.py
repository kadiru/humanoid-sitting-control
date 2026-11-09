"""Separate furniture environment; never imported by the running trainer."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
def load_case(name):
    cases=json.loads((HERE/'cases.json').read_text())
    return next(c for c in cases if c['id']==name)

def make_cfg(args,case):
    import run_supported_postures as support
    import run_resting_posture as base
    # Geometry is a proposed inspection target only. It cannot justify a clean
    # score: a couch has multiple surfaces; the old slab-plane scorer cannot
    # simply be reused as a validated furniture scorer.
    n=np.array(case['target_normal']);u=np.cross([0,1,0],n);u/=np.linalg.norm(u);v=np.cross(n,u)
    q=Rotation.from_matrix(np.stack([u,v,n],axis=1)).as_quat().tolist()
    center=case['target_center']
    geometry=[dict(case=0,label=case['id'],kind='lean' if case['task']=='leaning' else 'sit',family='furniture_probe',
        height=center[2],x=center[0],y=center[1],yaw=0.,pitch_deg=float(np.rad2deg(np.arccos(n[2]))),
        depth=.3,width=.4,top_center=center,normal=n.tolist(),rotation_xyzw=q,start_stage=0,split='separate_furniture_test')]
    cfg=support.make_cfg(args,geometry)
    cfg.scene.num_envs=1;cfg.scene.rest_camera=None
    cfg.scene.rest_floor.spawn.size=(5.,5.,.1)
    cfg.scene.seat=base.RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Seat',
        spawn=base.sim.UsdFileCfg(usd_path=str(HERE/'scenes'/case['id']/'furniture.usda'),activate_contact_sensors=True),
        init_state=base.RigidObjectCfg.InitialStateCfg(pos=(0.,0.,0.)))
    cfg.events.reset_seats.func=reset_mesh
    return cfg

def reset_mesh(raw,env_ids):
    import torch
    ids=torch.arange(raw.num_envs,device=raw.device) if env_ids is None else env_ids
    if isinstance(ids,slice):ids=torch.arange(raw.num_envs,device=raw.device)[ids]
    # Vertices already include their behind-robot placement. Do not apply the
    # old slab reset offset or rotation a second time.
    pose=torch.zeros((len(ids),7),device=raw.device);pose[:,:3]=raw.scene.env_origins[ids];pose[:,6]=1.
    raw.scene['seat'].write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
    raw.scene['seat'].write_root_velocity_to_sim_index(root_velocity=torch.zeros((len(ids),6),device=raw.device),env_ids=ids)

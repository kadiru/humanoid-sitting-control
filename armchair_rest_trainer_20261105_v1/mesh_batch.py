"""Original physical labels with per-environment exact furniture contact mapping."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pathlib import Path
from surface_map import SurfaceMap
from contact_adapter import array

class MeshMetrics:
    def __init__(self,case,original,support,record=False):
        self.surface=SurfaceMap(case);self.original=original;self.support=support;self.record=record
        self.validated=None;self.frames=[];self.calls=0
    def reset_recording(self,n):self.validated=np.ones(n,bool);self.frames=[];self.calls=0
    def __call__(self,raw):
        m,old,flags=self.original(raw);n=raw.num_envs
        if self.validated is None:self.reset_recording(n)
        self.calls+=1;robot=raw.scene['robot'];data=robot.data
        origins=raw.scene.env_origins.cpu().numpy();mapping=np.ones(n,bool);rows=[[] for _ in range(n)]
        for j,(sensor_name,body,rest) in enumerate(self.support.CONTACTS):
            if not rest:continue
            sensor=raw.scene[sensor_name];values=sensor.contact_view.get_contact_data(dt=.005)
            points,counts,starts=array(values[1]).reshape(-1,3),array(values[4]).reshape(-1),array(values[5]).reshape(-1)
            assert len(counts)==len(starts)==n
            cp=sensor.data.contact_pos_w.torch[:,0,0]
            b=robot.body_names.index(body);Rs=Rotation.from_quat(data.body_quat_w.torch[:,b].cpu().numpy()).as_matrix()
            positions=data.body_pos_w.torch[:,b].cpu().numpy()
            normals=[];inside=[]
            for i,(count,start) in enumerate(zip(counts,starts)):
                assert count<sensor.cfg.max_contact_data_count_per_prim
                ps=points[int(start):int(start+count)];labels=self.surface.map_points(ps-origins[i])
                active=bool(m['force_w'][i,j].norm()>.01*m['bodyweight_N'][i])
                same=bool(labels and np.all(np.linalg.norm(np.array([x['normal'] for x in labels])-labels[0]['normal'],axis=1)<1e-5))
                if active and (not labels or not same):mapping[i]=False
                normal=np.array(labels[0]['normal'] if same else self.surface.case['target_normal'])
                valid=bool(labels and all(x['valid'] for x in labels) and same)
                if j in [3,4] and len(ps) and np.isfinite(ps).all():valid&=bool(np.all(((ps-positions[i])@Rs[i])[:,0]<-.035))
                normals.append(normal);inside.append(valid)
                if self.record:rows[i].append(dict(channel=j,points=ps.tolist(),mapped=labels))
            m['normal_w'][:,j]=torch.as_tensor(np.array(normals),device=raw.device,dtype=m['normal_w'].dtype)
            m['inside_patch'][:,j]=torch.tensor(inside,device=raw.device)
            cp=torch.where(torch.isfinite(cp),cp,data.body_com_pos_w.torch[:,b]);vel=data.body_com_vel_w.torch[:,b]
            v=vel[:,:3]+torch.cross(vel[:,3:],cp-data.body_com_pos_w.torch[:,b],dim=-1);normal=m['normal_w'][:,j]
            m['slip_m_s'][:,j]=(v-(v*normal).sum(-1,keepdim=True)*normal).norm(dim=-1)
        active=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None]
        flags['forbidden']=(old['wrong_force_N']>10)|(active&~m['inside_patch']).any(1)
        m['failure']=torch.stack(list(flags.values())).any(0)
        _,good,d=self.support.resting_objective(m,torch.ones(n,dtype=torch.bool,device=raw.device))
        d.update(good=good,wrong_force_N=old['wrong_force_N'],peak_force_N=old['peak_force_N'],root_height_m=old['root_height_m'],tilt_deg=old['tilt_deg'])
        self.validated&=mapping
        if self.record:self.frames.append(dict(mapping_valid=mapping.tolist(),contacts=rows))
        return m,d,flags

def make_cfg(args,case):
    import run_supported_postures as support
    import run_resting_posture as base
    from scene_adapter import reset_mesh
    n=np.array(case['target_normal']);u=np.cross([0,1,0],n);u/=np.linalg.norm(u);v=np.cross(n,u)
    q=Rotation.from_matrix(np.stack([u,v,n],axis=1)).as_quat().tolist();center=case['target_center']
    geometry=[dict(case=i,label=case['id'],kind='lean',family='armchair_rest',height=center[2],x=center[0],y=center[1],yaw=0.,pitch_deg=float(np.rad2deg(np.arccos(n[2]))),depth=.3,width=.4,top_center=center,normal=n.tolist(),rotation_xyzw=q,start_stage=0,split='development') for i in range(args.num_envs)]
    cfg=support.make_cfg(args,geometry);cfg.scene.rest_camera=None;cfg.scene.env_spacing=5.
    cfg.scene.rest_floor.spawn.size=(5.,5.,.1)
    cfg.scene.seat=base.RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Seat',spawn=base.sim.UsdFileCfg(usd_path=str(Path(__file__).resolve().parents[1]/'furniture_test_20261028_v1/scenes'/case['id']/'furniture.usda'),activate_contact_sensors=True),init_state=base.RigidObjectCfg.InitialStateCfg(pos=(0.,0.,0.)))
    cfg.events.reset_seats.func=reset_mesh
    return cfg

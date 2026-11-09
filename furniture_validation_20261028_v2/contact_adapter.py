"""Evaluation-only geometry adapter; all physical constants remain upstream.

Preserves the seven aggregate body-contact channels and their load thresholds.
Every actual contact point must belong to the intended mesh patch. A channel
touching differently oriented faces cannot receive a validated clean score.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from surface_map import SurfaceMap

def array(value):
    import warp as wp
    if isinstance(value, torch.Tensor): return value.detach().cpu().numpy().copy()
    return wp.to_torch(value).detach().cpu().numpy().copy()

def raw_points(sensor, dt=.005):
    values = sensor.contact_view.get_contact_data(dt=dt)
    points, counts, starts = array(values[1]), array(values[4]), array(values[5])
    counts = counts.reshape(-1); starts = starts.reshape(-1)
    assert len(counts)==1, ('EXPECTED_ONE_BODY_ONE_FILTER',counts.shape)
    n, start = int(counts[0]), int(starts[0])
    assert n < sensor.cfg.max_contact_data_count_per_prim, 'CONTACT_BUFFER_SATURATED'
    return points.reshape(-1,3)[start:start+n].copy()

class MeshMetrics:
    def __init__(self, case, original, support):
        self.surface=SurfaceMap(case); self.original=original; self.support=support
        self.frames=[]; self.validated=True

    def __call__(self, raw):
        m, old_detail, flags = self.original(raw)
        assert raw.num_envs==1
        robot=raw.scene['robot']; data=robot.data
        origin=raw.scene.env_origins[0].cpu().numpy()
        points=[]; mapped=[]; mapping_ok=True
        # Compute the original body velocity at its original mean contact point.
        # Only the normal and geometric membership change for the new surface.
        for j,(sensor_name,body,rest) in enumerate(self.support.CONTACTS):
            sensor=raw.scene[sensor_name]
            cp=sensor.data.contact_pos_w.torch[0,0,0]
            if not rest:
                points.append(cp.detach().cpu().numpy().tolist() if torch.isfinite(cp).all() else None); mapped.append([]); continue
            ps=raw_points(sensor); labels=self.surface.map_points(ps-origin)
            active=bool(m['force_w'][0,j].norm()>.01*m['bodyweight_N'][0])
            same_normal=bool(labels and np.all(np.linalg.norm(np.array([x['normal'] for x in labels])-labels[0]['normal'],axis=1)<1e-5))
            if active and (not labels or not same_normal): mapping_ok=False
            normal=np.array(labels[0]['normal'] if same_normal else self.surface.case['target_normal'])
            m['normal_w'][0,j]=torch.as_tensor(normal,device=raw.device,dtype=m['normal_w'].dtype)
            inside=bool(labels and all(x['valid'] for x in labels) and same_normal)
            b=robot.body_names.index(body)
            if rest and j in (3,4) and np.isfinite(ps).all() and len(ps):
                rot=Rotation.from_quat(data.body_quat_w.torch[0,b].cpu().numpy()).as_matrix()
                local=(ps-data.body_pos_w.torch[0,b].cpu().numpy())@rot
                inside &= bool(np.all(local[:,0]<-.035))
            m['inside_patch'][0,j]=inside
            if not torch.isfinite(cp).all(): cp=data.body_com_pos_w.torch[0,b]
            velocity=data.body_com_vel_w.torch[0,b]
            v=velocity[:3]+torch.cross(velocity[3:],cp-data.body_com_pos_w.torch[0,b],dim=-1)
            n=m['normal_w'][0,j]
            m['slip_m_s'][0,j]=(v-(v*n).sum()*n).norm()
            points.append(cp.cpu().numpy().tolist()); mapped.append(labels)
        active=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None]
        flags['forbidden']=(old_detail['wrong_force_N']>10)|(active&~m['inside_patch']).any(1)
        m['failure']=torch.stack(list(flags.values())).any(0)
        _,good,detail=self.support.resting_objective(m,torch.ones(1,dtype=torch.bool,device=raw.device))
        detail.update(good=good,wrong_force_N=old_detail['wrong_force_N'],peak_force_N=old_detail['peak_force_N'],
            root_height_m=old_detail['root_height_m'],tilt_deg=old_detail['tilt_deg'])
        self.validated &= mapping_ok
        self.frames.append(dict(points=points,mapped=mapped,mapping_valid=mapping_ok))
        return m,detail,flags

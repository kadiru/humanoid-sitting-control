"""Clean-completion reward; bounded shaping and causal first-event costs.

All geometry/contact data in this module are reward labels, never observations.
The collision-vertex distances are shaping proxies, not physical certificates.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

# The static collision model has these six leg-housing meshes; ankle-pitch
# links have no collision geometry. Feet remain governed by physical labels.
HOUSINGS = [s+'_'+j+'_link' for s in ('left','right')
            for j in ('hip_pitch','hip_roll','knee')]
MODEL = Path('/home/kadir/humanoid-sitting/runs/g1_kinematics_geometry_04/model.json')
SCALE = 50.0

class LandingLabels:
    def __init__(self, body_names, geometry, origins, model_path=MODEL):
        data=json.loads(Path(model_path).read_text())
        self.vertices={}
        for body in ['pelvis']+HOUSINGS:
            points=[np.asarray(c['points'],dtype=np.float64) for c in data['collisions'] if c['body']==body]
            assert points,body
            self.vertices[body]=np.unique(np.concatenate(points),axis=0)
        self.ids={b:body_names.index(b) for b in self.vertices}
        self.rotation=Rotation.from_quat(np.array([g['rotation_xyzw'] for g in geometry])).as_matrix()
        self.top=np.array([g['top_center'] for g in geometry])+np.asarray(origins)
        self.extent=np.array([[g['depth']/2,g['width']/2] for g in geometry])

    def __call__(self, poses):
        # poses [N,B,7], quaternions XYZW, matching the saved physical trace.
        assert poses.ndim==3 and poses.shape[-1]==7
        clearance=np.full(len(poses),np.inf)
        gap=None
        for body,vertices in self.vertices.items():
            p=poses[:,self.ids[body]]
            R=Rotation.from_quat(p[:,3:]).as_matrix()
            world=np.einsum('nij,vj->nvi',R,vertices)+p[:,None,:3]
            local=np.einsum('nvi,nij->nvj',world-self.top[:,None,:],self.rotation)
            if body=='pelvis':
                outside=np.maximum(np.abs(local[...,:2])-(self.extent[:,None,:]-.01),0)
                gap=np.sqrt((outside**2).sum(-1)+local[...,2]**2).min(-1)
            else:
                centered=local.copy();centered[...,2]+=.05
                extent=np.c_[self.extent,np.full(len(poses),.05)]
                q=np.abs(centered)-extent[:,None,:]
                sdf=np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
                clearance=np.minimum(clearance,sdf.min(-1))
        assert np.isfinite(gap).all() and np.isfinite(clearance).all()
        return dict(pelvis_target_distance_m=gap,housing_vertex_clearance_m=clearance)

def first_events(mask, cost):
    mask=np.asarray(mask,dtype=bool)
    before=np.vstack([np.zeros((1,mask.shape[1]),bool),np.maximum.accumulate(mask,axis=0)[:-1]])
    return -float(cost)*(mask & ~before)

def episode_reward(records, geometry_labels, rows, baseline_margin, decision_steps):
    r=records
    n=len(rows)
    assert r['force'].shape==(1000,n,7,3)
    assert np.asarray(geometry_labels['pelvis_target_distance_m']).shape==(200,n)
    weight=r['weight']
    allowed=r['permitted'] & r['inside'] & r['rest_patch']
    normal=np.maximum((r['force']*r['normal']).sum(-1),0)
    load=np.clip((normal*allowed).sum(-1)/weight/.2,0,1)
    legal=r['legal'] & (r['wrong']<=10) & ~r['failure'] & r['alive']
    still=np.exp(-(r['root_speed']/.1)**2-(r['gyro']/.25)**2)
    loaded=np.linalg.norm(r['force'],axis=-1)>.01*weight[...,None]
    speed=np.where(loaded,r['slip'],0).max(-1)
    t=np.arange(1,1001)*.02
    active=(t>6)[:,None]
    support_gate=legal*(t>=8)[:,None]
    # Full horizon support reward cannot exceed .02*4/8 per physical tick.
    components={
        'legal_stationary_support':.02*2*load*still*support_gate/8,
        'good_rest':.02*2*r['good']*support_gate/8,
        'loaded_tangent_speed':-.02*2*np.clip(speed/.02,0,1)/8,
        'wrong_contact_force':-.02*np.clip(np.maximum(r['wrong']-10,0)/(.1*weight),0,1)/8,
        'illegal_patch':-.02*(~r['legal'])/8,
        'effort':-.02*.25*np.clip(r['total_squared_torque_Nm2']/1350,0,1)/8,
    }
    # Geometry is sampled after each .1s control interval, as in trace.npz.
    # Its reward is assigned at that endpoint, never to an earlier decision.
    gt=np.arange(1,201)*.1
    approach=.1*.5*np.exp(-(geometry_labels['pelvis_target_distance_m']/.05)**2)/8
    approach*=((gt>6)&(gt<=13))[:,None]*legal.reshape(200,5,n)[:,-1]
    housing=-.1*np.clip((.015-geometry_labels['housing_vertex_clearance_m'])/.015,0,1)/8
    housing*=(gt>6)[:,None]
    for k in components:components[k]=components[k]*active
    for name,values in [('approach',approach),('housing_clearance',housing)]:
        x=np.zeros((1000,n),dtype=np.float64);x[4::5]=values;components[name]=x
    events={'first_slip':first_events(~r['no_slip'],5),
            'first_illegal_patch':first_events(~r['legal'],5),
            'first_physical_failure':first_events(r['failure'],20)}
    for j,row in enumerate(rows):
        assert int((~r['no_slip'][:,j]).sum())==row['slip_ticks']
        assert int((~r['legal'][:,j]).sum())==row['illegal_ticks']
        assert bool(r['failure'][:,j].any())==any(row['failures'].values())
    clean=np.array([x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks'] for x in rows])
    terminal=50*clean.astype(float)-10*np.array([not x['success'] for x in rows])-2*np.array([x['traction_bad_ticks']>baseline_margin[j] for j,x in enumerate(rows)])
    step=sum(components.values())+sum(events.values());step[-1]+=terminal
    assert np.isfinite(step).all()
    control=step.reshape(200,5,n).sum(1)/SCALE
    reward=np.cumsum(control[::-1],axis=0)[::-1][decision_steps]
    shape=sum(components.values()).sum(0)
    # Conservative analytic bounds, including approach limited to 7 seconds:
    # support<=7; approach<=.4375; negative shaping>=-9.1875.
    assert (shape>=-9.1875-1e-10).all() and (shape<=7.4375+1e-10).all()
    full=control.sum(0)
    assert (full[clean]>=.77625-1e-10).all()
    assert (full[~clean]<=.14875+1e-10).all()
    audit=dict(clean=clean.tolist(),full_episode_return=full.tolist(),
        shaping_total=shape.tolist(),terminal_raw=terminal.tolist(),
        component_totals_raw={k:v.sum(0).tolist() for k,v in components.items()},
        event_totals_raw={k:v.sum(0).tolist() for k,v in events.items()},
        clean_return_lower_bound=.77625,nonclean_return_upper_bound=.14875,
        total_return_scaled_by=SCALE,geometry_is_proxy=True)
    return reward,control,audit

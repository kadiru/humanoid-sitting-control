"""Inspect no-drag pelvis rotation endpoints using the measured plane/model."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from low_positive_common import DATA,ROOT

p=DATA/'feedback_zero_01/old_parent';model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
audits=json.loads((p/'rolling_plan_audit.json').read_text());trace=np.load(p/'trace.npz');rows=[]
for case in range(0,12,2):
    a=next(a for a in audits if a['case']==case);q=trace['q'][99,case];origin=np.array(a['origin']);n=np.array(a['normal']);d=a['offset'];R=Rotation.from_euler('y',a['pitch']).as_matrix();f=model.fk(q)
    feet={s:(R@f[s+'_ankle_roll_link'][:3,3]+origin,R@f[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    vertices=model.vertices['pelvis'];w=vertices@R.T+origin;inside=(w[:,0]<-.16)&(w[:,0]>-.64)
    pivot=vertices[np.flatnonzero(inside)[np.argmin(w[inside]@n)]]
    seatR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0,(n[0]*.4-d)/n[2]])
    for angle in [-.5,-.35,-.2,0.,.1,.25]:
        x=origin[0]+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=.001,y=origin[1]);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=model.fk(qi);clear=[]
        for side in ['left','right']:
            for joint in ['hip_pitch','hip_roll']:
                v=(apply(T@fk[side+'_'+joint+'_link'],model.vertices[side+'_'+joint+'_link'])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):clear.append(float(v[:,2].min()))
        pelvis=(apply(T,vertices)-center)@seatR;patch=pelvis[pelvis[:,2]<=pelvis[:,2].min()+.002];margin=float(min(.25-abs(patch[:,0]).max(),.45-abs(patch[:,1]).max()))
        row=dict(case=case,height=float(center[2]),slope=float(np.rad2deg(np.arctan2(n[0],n[2]))),pitch=angle,x=float(x),ik_error=float(error),housing_clearance=min(clear),patch_margin=margin);rows.append(row);print(row,flush=True)
(DATA/'rolling_endpoint_screen.json').write_text(json.dumps(rows,indent=2))

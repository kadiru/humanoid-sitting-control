"""Offline finite-seat collision screen; its candidates require physics validation."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform

def clipped_vertices(points):
    """Intersect hull edges with the wide seat's front/back vertical boundaries."""
    hull=ConvexHull(points)
    edges=np.unique(np.sort(np.concatenate([hull.simplices[:,[0,1]],hull.simplices[:,[1,2]],hull.simplices[:,[0,2]]]),1),axis=0)
    kept=[points[(points[:,0]>=-.65)&(points[:,0]<=-.15)&(abs(points[:,1])<=.45)]]
    for x in [-.65,-.15]:
        a,b=points[edges[:,0]],points[edges[:,1]];dx=b[:,0]-a[:,0]
        valid=(abs(dx)>1e-10)&((a[:,0]-x)*(b[:,0]-x)<=0)
        hits=a[valid]+((x-a[valid,0])/dx[valid])[:,None]*(b[valid]-a[valid]);kept.append(hits[abs(hits[:,1])<=.45])
    return np.concatenate(kept)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False)
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json')
    old=Path('/data/humanoid-sitting-perception/contact_effort_repair_01')
    d=np.load(old/'banks/development/reset_candidates.npz');q0=d['q'][4,0];root0=d['root_pose'][4,0]
    qseed=d['q'][4,6];cseed=d['command'][4,6]
    T0=transform(Rotation.from_quat(root0[3:]).as_matrix(),root0[:3]);f0=m.fk(q0)
    feet={s:T0@f0[s+'_ankle_roll_link'] for s in ['left','right']}
    inertial=json.loads((old/'sitting_preflight_01/reference/static_inertial_model.json').read_text())
    rows=[];arrays=[]
    forbidden=[n for n in m.vertices if n not in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']]
    for x in [-.15,-.18,-.21,-.24,-.27,-.30]:
      for pitch in [-.2,0.,.2,.4,.6]:
        R=Rotation.from_euler('y',pitch).as_matrix();xyz=np.array([x,root0[1],0.])
        pv=m.vertices['pelvis']@R.T+xyz
        pv=pv[(pv[:,0]<-.16)&(pv[:,0]>-.64)]
        if len(pv)==0:continue
        xyz[2]=.4-pv[:,2].min()+.001
        q=qseed.astype(float).copy();error=[]
        for side in ['left','right']:
            ids=[m.names.index(side+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']]
            target=feet[side]
            def residual(v):
                trial=q.copy();trial[ids]=v;f=m.fk(trial)[side+'_ankle_roll_link']
                return np.r_[(R@f[:3,3]+xyz-target[:3,3])[[0,2]],.2*Rotation.from_matrix(target[:3,:3].T@R@f[:3,:3]).as_rotvec()[1]]
            fit=least_squares(residual,q[ids],bounds=([-2.4,.05,-.85],[.6,2.5,.5]),max_nfev=40)
            q[ids]=fit.x;error.append(float(np.linalg.norm(fit.fun)))
        fk=m.fk(q);T=transform(R,xyz);clear=[]
        for body in forbidden:
            pts=clipped_vertices(apply(T@fk[body],m.vertices[body]))
            if len(pts):clear.append((float(pts[:,2].min()-.4),body))
        clear.sort();com=np.average([apply(T@fk[n],c) for n,c in zip(inertial['body_names'],inertial['local_com'])],axis=0,weights=inertial['mass'])
        pv=apply(T,m.vertices['pelvis']);patch=pv[(pv[:,2]<pv[:,2].min()+.004)&(pv[:,0]<-.16)]
        sx=float(patch[:,0].mean()) if len(patch) else x
        footx=float(np.mean([t[0,3] for t in feet.values()]))
        row=dict(case=len(rows),x=x,pitch=pitch,ik_error=max(error),clearance=clear[:5],com=com.tolist(),seat_patch_x=sx,vertical_load_moment_estimate=(footx-com[0])/(footx-sx),scope='Static gravity moment estimate only; not a measured contact force')
        rows.append(row);arrays.append((q,np.r_[xyz,Rotation.from_matrix(R).as_quat()]))
        print(row,flush=True)
    (out/'screen.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(out/'poses.npz',q=np.array([v[0] for v in arrays]),root_pose=np.array([v[1] for v in arrays]),source_command=cseed,source_q=qseed)

if __name__=='__main__':main()

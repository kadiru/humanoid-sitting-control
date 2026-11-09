"""Offline near-seat reset candidates. Physics validation is mandatory."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices

OLD=Path('/data/humanoid-sitting-perception/seat_load_transfer_01')
ROOT=Path('/home/kadir/humanoid-sitting')

def solve(model,qseed,feet,normal,offset,x,pitch,gap=.001,y=0.):
    R=Rotation.from_euler('y',pitch).as_matrix();xyz=np.array([x,y,0.]);v=model.vertices['pelvis']@R.T
    selected=v[(v[:,0]+x<-.16)&(v[:,0]+x>-.64)];assert len(selected)
    xyz[2]=(gap-(selected@normal).min()-normal[:2]@xyz[:2]-offset)/normal[2]
    q=qseed.astype(float).copy();errors=[]
    for side in ['left','right']:
        ids=[model.names.index(side+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']];pos,rot=feet[side]
        def residual(values):
            trial=q.copy();trial[ids]=values;f=model.fk(trial)[side+'_ankle_roll_link']
            return np.r_[(R@f[:3,3]+xyz-pos)[[0,2]],.2*Rotation.from_matrix(rot.T@R@f[:3,:3]).as_rotvec()[1]]
        fit=least_squares(residual,q[ids],bounds=([-2.4,.05,-.85],[.6,2.5,.5]),max_nfev=60,xtol=1e-9,gtol=1e-9,ftol=1e-9);q[ids]=fit.x;errors.append(float(np.linalg.norm(fit.fun)))
    return q,np.r_[xyz,Rotation.from_matrix(R).as_quat()],max(errors)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');bank=OLD/'banks/development';gs=json.loads((bank/'geometry.json').read_text());res=dict(np.load(bank/'reset_candidates.npz'))
    # Only development surfaces; a future reserved bank is separate from probes.
    ids=[i for i,g in enumerate(gs) if g['kind']=='sit'];rows=[];arrays=[];audit=[]
    for i in ids:
        g=gs[i];q0=res['q'][i,0];r0=res['root_pose'][i,0];R0=Rotation.from_quat(r0[3:]).as_matrix();fk=model.fk(q0)
        feet={s:(R0@fk[s+'_ankle_roll_link'][:3,3]+r0[:3],R0@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        n=np.array(g['normal']);d=-n@np.array(g['top_center'])
        for pitch in [0.,.10,.20,.35]:
            q,r,error=solve(model,res['q'][i,6],feet,n,d,-.18,pitch,y=r0[1]);assert error<.003
            # Preserve non-leg command preload exactly as in the validated target.
            c=q+(res['command'][i,6]-res['q'][i,6]);leg=[j for j,name in enumerate(model.names) if any(s in name for s in ['hip_','knee','ankle'])];c[leg]=q[leg]
            clear=[];T=transform(Rotation.from_quat(r[3:]).as_matrix(),r[:3]);f=model.fk(q)
            for b in model.vertices:
                if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                pts=clipped_vertices(apply(T@f[b],model.vertices[b]))
                if len(pts):clear.append((float((pts@n+d).min()),b))
            row=dict(g,case=len(rows),label=g['label']+f'_pitch{pitch}',source_surface=i,near_pitch=pitch)
            rows.append(row);arrays.append((q,r,c));audit.append(dict(case=row['case'],ik_error=error,clearance=sorted(clear)[:4]))
    (a.output/'geometry.json').write_text(json.dumps(rows,indent=2));(a.output/'screen.json').write_text(json.dumps(audit,indent=2))
    np.savez_compressed(a.output/'candidates.npz',q=np.array([x[0] for x in arrays]),root_pose=np.array([x[1] for x in arrays]),command=np.array([x[2] for x in arrays]),qd=np.zeros((len(rows),37)),root_velocity=np.zeros((len(rows),6)),valid=np.zeros(len(rows),bool))
    print('CANDIDATES',len(rows),'min clearance',min(x['clearance'][0][0] for x in audit))
if __name__=='__main__':main()

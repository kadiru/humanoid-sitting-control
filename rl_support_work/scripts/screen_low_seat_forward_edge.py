"""Offline finite-seat clearance and support-patch screen; no new physics."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')

def main():
    p=D/'damped_original_01/baseline';audits=json.loads((p/'rolling_plan_audit.json').read_text());qs=np.load(p/'last_rolling_paths.npz')['q']
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');inertial=json.loads((p/'static_inertial_model.json').read_text());output=[]
    for i in [0,2,4,8,12]:
        a=audits[i];q=qs[i,0];origin=np.array(a['origin']);R=Rotation.from_euler('y',a['pitch']).as_matrix();fk=m.fk(q)
        feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        n=np.array(a['normal']);d=a['offset'];SR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0.,(n[0]*.4-d)/n[2]]);rows=[]
        for x in [-.06,-.08,-.10,-.12,-.14,-.16]:
            for pitch in [.4,.6,.8]:
                try:qi,root,error=solve(m,q,feet,n,d,x,pitch,gap=.001,y=origin[1])
                except AssertionError:
                    rows.append(dict(x=x,pitch=pitch,rejected='No pelvis model vertex over the finite seat'));continue
                T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(qi);v=[]
                for s in ['left','right']:
                    for j in ['hip_pitch','hip_roll','knee','ankle_pitch']:
                        name=s+'_'+j+'_link'
                        if name not in m.vertices:continue
                        points=(apply(T@f[name],m.vertices[name])-center)@SR;points[:,0]-=.4;points=clipped_vertices(points)
                        if len(points):v.append((float(points[:,2].min()),name))
                pelvis=apply(T,m.vertices['pelvis']);local=(pelvis-center)@SR;patch=local[:,2]<=local[:,2].min()+.002
                margin=min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max());sx=float(pelvis[patch,0].mean())
                b,st=static_bias(m,inertial,qi,root,sx,0.)
                rows.append(dict(x=x,pitch=pitch,error=error,clearance=min(v) if v else [1.,'outside seat'],patch_margin=float(margin),no_seat_foot_cop=st['foot_cop_relative_to_ankle']))
        valid=[r for r in rows if 'rejected' not in r and r['error']<.003 and r['patch_margin']>.015]
        print('CASE',i,sorted(valid,key=lambda r:r['clearance'][0],reverse=True)[:5],flush=True);output.append(dict(case=i,rows=rows))
    out=D/'forward_edge_screen_03.json';assert not out.exists();out.write_text(json.dumps(dict(scope='Offline model screen only, not a clearance certificate; known development data only',rows=output),indent=2))

if __name__=='__main__':main()

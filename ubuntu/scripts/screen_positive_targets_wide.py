"""Offline endpoint diagnosis from recorded measured planes and robot kinematics."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from positive_slope_common import *

def main():
    p=DATA/'assessment/reference';out=DATA/'target_screen_02';out.mkdir(exist_ok=False)
    model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    trace=np.load(p/'trace.npz');audits=read(p/'rolling_plan_audit.json');geoms=read(p/'geometry.json');rows=read(p/'results.json')['episodes']
    result=[]
    for case in range(0,24,2):
        a=next(x for x in audits if x['case']==case);q=trace['q'][99,case];R=Rotation.from_euler('y',a['pitch']).as_matrix();origin=np.array(a['origin']);fk=model.fk(q)
        n=np.array(a['normal']);d=a['offset'];feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        seatR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0.,(n[0]*.4-d)/n[2]])
        options=[]
        for x in [-.09,-.12,-.18,-.24,-.30,-.36]:
            for pitch in [.1,.35,.55,.66]:
                qi,root,error=solve(model,q,feet,n,d,x,pitch,gap=-.001,y=origin[1])
                T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=model.fk(qi);clear=[]
                for s in ['left','right']:
                    for j in ['hip_pitch','hip_roll']:
                        b=s+'_'+j+'_link';points=apply(T@f[b],model.vertices[b]);local=(points-center)@seatR
                        # Reuse exact convex-edge clipping in seat coordinates.
                        shifted=local.copy();shifted[:,0]-=.4;v=clipped_vertices(shifted)
                        if len(v):clear.append(float(v[:,2].min()))
                pelvis=apply(T,model.vertices['pelvis']);local=(pelvis-center)@seatR
                patch=local[local[:,2]<=local[:,2].min()+.002]
                margin=float(min(.25-abs(patch[:,0]).max(),.45-abs(patch[:,1]).max()))
                options.append(dict(x=x,pitch=pitch,error=error,clearance=min(clear) if clear else 1.,patch_margin=margin,root=root.tolist(),q=qi.tolist()))
        g=geoms[rows[case]['geometry_index']];valid=[x for x in options if x['error']<.003 and x['clearance']>.006 and x['patch_margin']>.02]
        record=dict(case=case,height=g['height'],slope=g['pitch_deg'],source='Recorded measured plane/origin; pitch from encoder/IMU planner audit; no live truth input',options=options,valid=valid)
        result.append(record);write(out/'screen.json',result)
        print(case,g['height'],g['pitch_deg'],'valid',[(x['x'],x['pitch'],round(x['clearance'],3),round(x['patch_margin'],3)) for x in valid],flush=True)
    write(out/'sources.json',{str(p/f):sha(p/f) for f in ['trace.npz','rolling_plan_audit.json','geometry.json','results.json']})
if __name__=='__main__':main()

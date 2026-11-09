"""Offline development-only feasibility screen, not physical admission.

Check coupled orientation/descent paths against the existing five-second
transfer and applied command-rate cap. No physical criteria change or simulation.
"""
import json, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, transform
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
from crouch_rest_motion import smooth

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')

def main():
    p=D/'normal_landing3s_06/candidate'
    audits=json.loads((p/'rolling_plan_audit.json').read_text())
    paths=np.load(p/'last_rolling_paths.npz')
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json')
    inertia=json.loads((p/'static_inertial_model.json').read_text())
    ids=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    rows=[]
    for case in [0,2,6,10]:
        a=next(a for a in audits if a['case']==case)
        q=paths['q'][case,0];cmd=paths['command'][case,0]
        origin=np.array(a['origin']);R=Rotation.from_euler('y',a['pitch']).as_matrix();fk=m.fk(q)
        feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        n=np.array(a['normal']);d=a['offset'];seatR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix()
        center=np.array([-.4,0.,(n[0]*.4-d)/n[2]])
        root0=np.r_[origin,Rotation.from_matrix(R).as_quat()]
        b0,_=static_bias(m,inertia,q,root0,-.2,0.)
        preload=cmd[ids]-q[ids]-b0
        for pitch in [.9,.95]:
            # Two geometrical phases; evaluate their separate necessary rates.
            commands=[];clearances=[];margins=[];cop_margins=[];errors=[]
            for k in range(49):
                rotate=smooth(k/48.);land=rotate
                angle=a['pitch']+(pitch-a['pitch'])*rotate
                gap=a['gap_initial']+(.001-a['gap_initial'])*land
                x=origin[0]-(a['gap_initial']-.001)*n[0]*land
                qi,root,err=solve(m,q,feet,n,d,x,angle,gap=gap,y=origin[1]);errors.append(err)
                T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(qi)
                pelvis=apply(T,m.vertices['pelvis']);local=(pelvis-center)@seatR
                patch=local[:,2]<local[:,2].min()+.002
                margins.append(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()))
                sx=float(pelvis[patch,0].mean());bias,stat=static_bias(m,inertia,qi,root,sx,0.)
                feet_points=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']])
                # Model sole bounds only; not dynamic support certification.
                sole=feet_points[feet_points[:,2]<feet_points[:,2].min()+.003]
                cop_margins.append(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x']))
                values=[]
                for side in ['left','right']:
                    for joint in ['hip_pitch','hip_roll','knee','ankle_pitch']:
                        body=side+'_'+joint+'_link'
                        if body not in m.vertices:continue
                        v=(apply(T@f[body],m.vertices[body])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                        if len(v):values.append(float(v[:,2].min()))
                clearances.append(min(values) if values else 1.)
                commands.append(qi[ids]+bias+(1-rotate)*preload)
            commands[0]=cmd[ids]
            rate=float(abs(np.diff(commands,axis=0)).max()*48/5)
            durations=[dict(rotation_seconds=5.,landing_seconds=5.,concurrent=True,finish_seconds=15.,
                peak_no_seat_target_rate_rad_s=rate,necessary_rate_pass=rate<=.15)]
            row=dict(case=case,pitch_rad=pitch,max_ik_error=max(errors),minimum_housing_clearance_m=min(clearances),
                endpoint_housing_clearance_m=clearances[-1],minimum_finite_patch_margin_m=min(margins),endpoint_patch_margin_m=margins[-1],
                minimum_no_seat_model_cop_margin_m=min(cop_margins),durations=durations)
            row['screen_pass']=bool(max(errors)<.003 and min(clearances)>=.003 and min(margins)>0. and margins[-1]>=.015 and min(cop_margins)>=.005 and any(x['necessary_rate_pass'] for x in durations))
            rows.append(row);print(json.dumps(row),flush=True)
    out=D/'coupled_path_screen_11.json';assert not out.exists()
    out.write_text(json.dumps(dict(scope='Offline recorded development state/model only. Clearance3mm, endpoint patch margin15mm (air-path patch must remain over seat), and CoP5mm are research-screen targets, not changes to physical pass/fail criteria. No physical admission implied; static preload rate ignores changing feedback and is only a necessary check.',
        physical_simulations=0,optimizer_updates=0,fresh_holdout_used=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2))

if __name__=='__main__':main()

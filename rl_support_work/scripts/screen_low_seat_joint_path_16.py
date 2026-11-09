"""Offline joint-path and command-arc timing check on failed development rows."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from humanoid_sitting.contact_aware_reference import measured_level_plane
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')

def main():
    p=D/'normal_landing3s_06/candidate';t=np.load(p/'trace.npz')
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');inertia=json.loads((p/'static_inertial_model.json').read_text())
    cal=json.loads(Path('/data/humanoid-sitting-perception/contact_effort_repair_01/plane_calibration.json').read_text())
    ids=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    rows=[]
    for case in [0,2,6,10]:
        ti=int(np.argmin(abs(t['t']-8.)));q=t['q'][ti,case].astype(float);cmd=t['command'][ti,case]
        gravity=Rotation.from_quat(t['body'][ti,case,0,3:]).inv().apply([0,0,-1.]);R=level_rotation(gravity);geom=t['geometry'][ti,case]
        raw_n,raw_d,origin=measured_level_plane(m,q,gravity,geom,[t['estimated_root_x'][ti,case],0,0])
        rn=np.array(cal['reference_normal']);rd=cal['reference_offset']
        height=.4-(-.4*raw_n[0]+raw_d)/raw_n[2]+(-.4*rn[0]+rd)/rn[2]
        slope=np.arctan2(raw_n[0],raw_n[2])-np.arctan2(rn[0],rn[2])
        n=Rotation.from_euler('y',slope).as_matrix()[:,2];d=-n@np.array([-.4,0.,height]);seatR=Rotation.from_euler('y',slope).as_matrix();center=np.array([-.4,0.,height])
        fk=m.fk(q);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        end,endroot,err=solve(m,q,feet,n,d,-.17,.9,gap=.001,y=origin[1])
        root0=np.r_[origin,Rotation.from_matrix(R).as_quat()];bias0,_=static_bias(m,inertia,q,root0,-.2,0.);preload=cmd[ids]-q[ids]-bias0
        commands=[];path=[]
        for fraction in np.linspace(0,1,97):
            qi=q+fraction*(end-q);fk=m.fk(qi)
            rots=[feet[s][1]@fk[s+'_ankle_roll_link'][:3,:3].T for s in ['left','right']]
            rr=rots[0];positions=[feet[s][0]-rr@fk[s+'_ankle_roll_link'][:3,3] for s in ['left','right']]
            xyz=np.mean(positions,axis=0);T=transform(rr,xyz);root=np.r_[xyz,Rotation.from_matrix(rr).as_quat()]
            pelvis=apply(T,m.vertices['pelvis']);local=(pelvis-center)@seatR;depth=pelvis@n+d
            patch=depth<depth.min()+.002;gap=float(depth.min());pm=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()))
            sx=float(pelvis[patch,0].mean());bias,stat=static_bias(m,inertia,qi,root,sx,0.)
            fp=np.concatenate([apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=fp[fp[:,2]<fp[:,2].min()+.003]
            cm=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x']))
            clear=[]
            for name,vertices in m.vertices.items():
                if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                v=(apply(T@fk[name],vertices)-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):clear.append(float(v[:,2].min()))
            commands.append(qi[ids]+bias+(1-fraction)*preload)
            path.append(dict(fraction=float(fraction),root=root.tolist(),gap_m=gap,patch_margin_m=pm,cop_margin_m=cm,
                forbidden_clearance_m=min(clear) if clear else 1.,foot_position_mismatch_m=float(np.linalg.norm(positions[0]-positions[1])),
                foot_rotation_mismatch_rad=float(np.linalg.norm(Rotation.from_matrix(rots[0].T@rots[1]).as_rotvec()))))
        commands[0]=cmd[ids];delta=np.diff(commands,axis=0);arc=float(np.max(abs(delta),axis=1).sum())
        # Reparameterization by command arc, with0.25s linear velocity ramps.
        # This is only a nominal rate check; feedback/actuator tracking remain untested.
        peak=arc/(7.-.25)
        near=[x for x in path if x['gap_m']<.02]
        row=dict(case=case,start_s=8,end_s=15,goal_x=-.17,goal_pitch_rad=.9,goal_ik_error=err,
            command_arc_rad=arc,retimed_peak_nominal_rate_rad_s=peak,
            min_forbidden_clearance_m=min(x['forbidden_clearance_m'] for x in path),
            minimum_pelvis_gap_m=min(x['gap_m'] for x in path),
            min_near_contact_patch_margin_m=min(x['patch_margin_m'] for x in near),endpoint_patch_margin_m=path[-1]['patch_margin_m'],
            min_no_seat_cop_margin_m=min(x['cop_margin_m'] for x in path),
            max_foot_position_mismatch_m=max(x['foot_position_mismatch_m'] for x in path),
            max_foot_rotation_mismatch_rad=max(x['foot_rotation_mismatch_rad'] for x in path),path=path)
        row['screen_pass']=bool(err<.003 and peak<=.15 and row['min_forbidden_clearance_m']>=.003 and row['minimum_pelvis_gap_m']>=0 and row['min_near_contact_patch_margin_m']>0 and row['endpoint_patch_margin_m']>=.015 and row['min_no_seat_cop_margin_m']>=.005 and row['max_foot_position_mismatch_m']<.003 and row['max_foot_rotation_mismatch_rad']<.003)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='path'}),flush=True)
    out=D/'joint_path_screen_16.json';assert not out.exists()
    out.write_text(json.dumps(dict(scope='Offline recorded development state/model screen only; no physics, learning or admission. Joint interpolation and command-arc timing test the identified rate conflict without increasing the actuator limit. Proposed selected-case prefix would be8s, not10s. Plane inputs are recorded sensor geometry with offline IMU-equivalent orientation; no simulator truth may enter any future runtime.',
        research_screen_revision='Finite pelvis patch is required when gap<20mm and at touchdown; an airborne approach may start outside the seat. This screen rule does not change physical support-patch criteria. Housing3mm and no-seat CoP5mm are unchanged model-screen targets.',
        physical_trials=0,optimizer_updates=0,fresh_holdout_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows),indent=2))

if __name__=='__main__':main()

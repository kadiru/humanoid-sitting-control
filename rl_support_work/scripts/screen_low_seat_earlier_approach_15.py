"""Development-recording feasibility check; no simulation or controller change."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from humanoid_sitting.contact_aware_reference import measured_level_plane
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
from crouch_rest_motion import smooth
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')

def main():
    p=D/'normal_landing3s_06/candidate';t=np.load(p/'trace.npz')
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json')
    inertia=json.loads((p/'static_inertial_model.json').read_text())
    cal=json.loads(Path('/data/humanoid-sitting-perception/contact_effort_repair_01/plane_calibration.json').read_text())
    ids=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    rows=[]
    for case in [0,2,6,10]:
        ti=int(np.argmin(abs(t['t']-8.)));q=t['q'][ti,case];cmd=t['command'][ti,case]
        # Recorded body orientation is used only for this offline IMU-equivalent
        # model screen; runtime, if admitted later, must consume actual IMU.
        gravity=Rotation.from_quat(t['body'][ti,case,0,3:]).inv().apply([0,0,-1.])
        R=level_rotation(gravity);geom=t['geometry'][ti,case]
        raw_n,raw_d,origin=measured_level_plane(m,q,gravity,geom,np.array([t['estimated_root_x'][ti,case],0.,0.]))
        rn=np.array(cal['reference_normal']);rd=cal['reference_offset']
        height=.4-(-.4*raw_n[0]+raw_d)/raw_n[2]+(-.4*rn[0]+rd)/rn[2]
        slope=np.arctan2(raw_n[0],raw_n[2])-np.arctan2(rn[0],rn[2])
        n=Rotation.from_euler('y',slope).as_matrix()[:,2];d=-n@np.array([-.4,0.,height])
        fk=m.fk(q);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        root0=np.r_[origin,Rotation.from_matrix(R).as_quat()];pitch0=Rotation.from_matrix(R).as_euler('xyz')[1]
        pelvis=apply(transform(R,origin),m.vertices['pelvis']);inside=pelvis[:,0]<-.16
        gap0=(pelvis[inside]@n+d).min()
        bias0,_=static_bias(m,inertia,q,root0,-.2,0.);preload=cmd[ids]-q[ids]-bias0
        seatR=Rotation.from_euler('y',slope).as_matrix();center=np.array([-.4,0.,height])
        commands=[];clear=[];patch_margin=[];cop_margin=[];errors=[]
        for k in range(49):
            blend=smooth(k/48);angle=pitch0+(.9-pitch0)*blend
            gap=gap0+(.001-gap0)*blend;x=origin[0]+(-.17-origin[0])*blend
            qi,root,err=solve(m,q,feet,n,d,x,angle,gap=gap,y=origin[1]);errors.append(err)
            T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(qi)
            pelvis=apply(T,m.vertices['pelvis']);local=(pelvis-center)@seatR
            patch=local[:,2]<local[:,2].min()+.002
            patch_margin.append(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()))
            sx=float(pelvis[patch,0].mean());bias,stat=static_bias(m,inertia,qi,root,sx,0.)
            fp=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']])
            sole=fp[fp[:,2]<fp[:,2].min()+.003]
            cop_margin.append(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x']))
            values=[]
            for side in ['left','right']:
                for joint in ['hip_pitch','hip_roll','knee','ankle_pitch']:
                    body=side+'_'+joint+'_link'
                    v=(apply(T@f[body],m.vertices[body])-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                    if len(v):values.append(float(v[:,2].min()))
            clear.append(min(values) if values else 1.)
            commands.append(qi[ids]+bias+(1-blend)*preload)
        commands[0]=cmd[ids]
        commands=np.array(commands);rate=float(abs(np.diff(commands,axis=0)).max()*48/7)
        row=dict(case=case,start_seconds=8.,end_seconds=15.,source_sample_seconds=float(t['t'][ti]),
            measured_geometry_valid=bool(geom[11] and geom[12]<=.5),measured_height_m=height,measured_slope_deg=float(np.rad2deg(slope)),
            initial_pitch_rad=pitch0,initial_gap_m=gap0,initial_root=origin.tolist(),goal_x=-.17,goal_pitch_rad=.9,
            max_ik_error=max(errors),minimum_housing_clearance_m=min(clear),endpoint_housing_clearance_m=clear[-1],
            minimum_patch_margin_m=min(patch_margin),endpoint_patch_margin_m=patch_margin[-1],
            minimum_no_seat_cop_margin_m=min(cop_margin),peak_nominal_target_rate_rad_s=rate,
            command_arc_lower_bound_seconds=float(np.max(abs(np.diff(commands,axis=0)),axis=1).sum()/.15))
        row['screen_pass']=bool(row['measured_geometry_valid'] and max(errors)<.003 and min(clear)>=.003 and min(patch_margin)>0 and patch_margin[-1]>=.015 and min(cop_margin)>=.005 and rate<=.15)
        rows.append(row);print(json.dumps(row),flush=True)
    out=D/'earlier_approach_screen_15.json';assert not out.exists()
    out.write_text(json.dumps(dict(scope='Offline model screen on four already-failed development geometries. Explicitly proposes changing selected-case approach after8s, instead of the previous10s lock; no runtime implementation or physical admission yet. Larger clearance and slower7s path must pass feasibility first. Physical criteria and materials unchanged.',
        physical_trials=0,optimizer_updates=0,fresh_holdout_used=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256={str(p/n):hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['trace.npz','static_inertial_model.json']},rows=rows),indent=2))

if __name__=='__main__':main()

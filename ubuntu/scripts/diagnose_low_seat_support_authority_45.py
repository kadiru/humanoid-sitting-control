"""Offline upper-body first-moment bound, not a controller or motion proposal.

The prior fixed-arm endpoint is limited by foot support. Quantify whether
bounded symmetric arm motion can change that limitation before designing any
new approach. Fixed feet, 38-degree pelvis and original physical contracts.
No simulator, learning, contact-force oracle or holdout is used.
"""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, transform, level_rotation
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OUT=D/'support_authority_45'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    mp=Path('runs/g1_kinematics_geometry_04/model.json')
    ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'
    cp=D.parent/'contact_effort_repair_01/plane_calibration.json'
    m=BodyModel(mp); inertia=read(ip); cal=read(cp); meta=read(mp)
    src=D/'normal_landing3s_06/candidate'; tr=np.load(src/'trace.npz'); sensors=np.load(src/'sensor_inputs.npz')['sensors']
    limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'}
    ids={s:{j:m.names.index(s+'_'+j+'_joint') for j in ['shoulder_pitch','shoulder_roll','elbow_pitch']} for s in ['left','right']}
    legs=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    mass=np.asarray(inertia['mass']); rows=[]
    at=int(np.argmin(abs(tr['t']-1.)))
    for case in [0,2,6,10]:
        q=tr['q'][at,case].astype(float); command=tr['command'][at,case].astype(float)
        frame=sensors[0,10,case,270:405];g=frame[69:72];geom=tr['geometry'][at,case]
        assert np.array_equal(frame[:23],q[:23].astype(np.float32)) and geom[11] and geom[12]<=.5
        n,d,origin,h,slope,_,_=measured(m,q,g,geom,[tr['estimated_root_x'][at,case],0,0],cal)
        R0=level_rotation(g);f=m.fk(q);feet={s:(R0@f[s+'_ankle_roll_link'][:3,3]+origin,R0@f[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        R=Rotation.from_euler('y',38.,degrees=True).as_matrix()
        def arm_q(v):
            out=q.copy()
            for s,sign in [('left',1),('right',-1)]:
                out[ids[s]['shoulder_pitch']]+=v[0]
                out[ids[s]['shoulder_roll']]+=sign*v[1]
                out[ids[s]['elbow_pitch']]+=v[2]
            return out
        def com(v):
            fk=m.fk(arm_q(v))
            return np.average([R@apply(fk[b],c) for b,c in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
        # Fixed explicit bounds: this is a static reach envelope, not a sweep
        # of physical controllers or permission to use the maximizing pose.
        bounds=[(-1.2,0.),(-.6,0.),(-.35,.6)]
        searches=[]
        for start in [(-.35,0.,-.35),(-.8,-.3,0.)]:
            fit=minimize(lambda v:-float(com(v)[0]),start,method='SLSQP',bounds=bounds,options=dict(maxiter=60,ftol=1e-12))
            searches.append(dict(success=bool(fit.success),message=str(fit.message),offsets=fit.x.tolist(),local_com=com(fit.x).tolist()))
        best=max(searches,key=lambda r:r['local_com'][0]);v=np.asarray(best['offsets']);seed=arm_q(v)
        sr=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h])
        def endpoint(x,detail=False):
            qi,root,err=solve(m,seed,feet,n,d,x,np.deg2rad(38.),gap=.001,y=origin[1]);T=transform(R,root[:3]);fk=m.fk(qi)
            pelvis=apply(T,m.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002;local=(pelvis-center)@sr
            bias,stat=static_bias(m,inertia,qi,root,float(pelvis[patch,0].mean()),0.)
            foot=np.concatenate([apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
            rear=float(stat['com'][0]-sole[:,0].min())
            if not detail:return rear
            clear=[];floor=[]
            for name,vertices in m.vertices.items():
                world=apply(T@fk[name],vertices)
                if name not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append((float(world[:,2].min()),name))
                if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                localv=(world-center)@sr;localv[:,0]-=.4;localv=clipped_vertices(localv)
                if len(localv):clear.append((float(localv[:,2].min()),name))
            cmd=command.copy();cmd[legs]=qi[legs]+bias
            armids=[j for x in ids.values() for j in x.values()];cmd[armids]+=qi[armids]-q[armids]
            margin=min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in legs+armids)
            return dict(x=float(x),root=root.tolist(),q=qi.tolist(),command=cmd.tolist(),com=stat['com'],rear_cop_margin_m=rear,front_cop_margin_m=float(sole[:,0].max()-stat['com'][0]),clearance_m=min(clear)[0],nearest_body=min(clear)[1],floor_clearance_m=min(floor)[0],patch_margin_m=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),ik_error=float(err),command_joint_margin_rad=float(margin),terminal=terminal_posture_check(root),assumed_seat_share=0.)
        bracket=[endpoint(-.3),endpoint(-.1)]
        ep=endpoint(float(brentq(lambda x:endpoint(x)-.0055,-.3,-.1,xtol=1e-9)),True) if bracket[0]<.0055<bracket[1] else None
        row=dict(case=case,height_m=float(h),slope_deg=float(slope),fixed_previous_arm_com=com([-.35,0.,-.35]).tolist(),maximum_bounded_arm_com=best['local_com'],maximum_forward_com_gain_m=float(best['local_com'][0]-com([-.35,0.,-.35])[0]),bounds=bounds,searches=searches,endpoint=ep,executable_motion=False,whole_path_checked=False,self_collision_checked=False,physical_admission=False)
        rows.append(row);(OUT/f'case_{case}.json').write_text(json.dumps(row,indent=2));print({k:row[k] for k in ['case','maximum_forward_com_gain_m']}|dict(endpoint_clearance_m=None if ep is None else ep['clearance_m']),flush=True)
    paths=[Path(__file__),mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',Path('scripts/low_seat_terminal_contract_42.py')]
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_ticks=0,optimizer_updates=0,holdout_used=False,scope='Static upper-body first-moment bound at38deg, followed by fixed-foot/no-seat-load endpoint balance. Two bounded local starts. No self-collision, path, rate, dynamics, contact transition or measured-support validation; not an executable candidate. No runtime or physical criterion changed.',source_input_sha256={str(p.resolve()):sha(p) for p in paths},rows=rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2))

if __name__=='__main__':main()

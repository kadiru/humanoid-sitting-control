"""Offline constrained support-posture calculation, not a physical controller.

Use recorded 1 s sensors and fixed feet. Minimize symmetric arm displacement
subject to a 7 mm forbidden-body clearance reserve and the existing 5 mm CoP,
15 mm seat-patch and 3 mm IK screens. Expanded bounded posture family:
53degree pitch (unchanged55degree physical label), shoulder offset2.1rad and
elbow offset1.1rad. Not a runtime change. These extra degrees of freedom
are admitted only to this local mathematical search, never directly to physics. A bounded local search is not a proof of
feasibility, dynamic balance, or impossibility. No physics or learning.
"""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, transform, level_rotation
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
from low_seat_joint_path_controller_17 import measured

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OUT=D/'support_posture_envelope_34'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    source=D/'normal_landing3s_06/candidate'
    tr=np.load(source/'trace.npz'); sensors=np.load(source/'sensor_inputs.npz')['sensors']
    mp=Path('runs/g1_kinematics_geometry_04/model.json')
    ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'
    cp=D.parent/'contact_effort_repair_01/plane_calibration.json'
    m=BodyModel(mp); inertia=read(ip); cal=read(cp); md=read(mp)
    limits={j['name']:np.deg2rad(j['limits']) for j in md['joints'] if j['type']=='PhysicsRevoluteJoint'}
    shoulder=[m.names.index(s+'_shoulder_pitch_joint') for s in ['left','right']]
    elbow=[m.names.index(s+'_elbow_pitch_joint') for s in ['left','right']]
    legs=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    arm_ids=shoulder+elbow; at=int(np.argmin(abs(tr['t']-1.))); rows=[]
    for case in [2,10]:
        q=tr['q'][at,case].astype(float); command=tr['command'][at,case].astype(float)
        frame=sensors[0,10,case,270:405]; assert np.array_equal(frame[:23],q[:23].astype(np.float32))
        g=frame[69:72].astype(float); geom=tr['geometry'][at,case]
        assert geom[11] and geom[12]<=.5
        n,d,origin,h,slope,_,_=measured(m,q,g,geom,[tr['estimated_root_x'][at,case],0,0],cal)
        R=level_rotation(g); fk=m.fk(q)
        feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        SR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix(); center=np.array([-.4,0,h])
        cache={}; history=[]
        def evaluate(x):
            key=tuple(float(v) for v in x)
            if key in cache:return cache[key]
            seed=q.copy(); seed[shoulder]+=key[2]; seed[elbow]+=key[3]
            qi,root,error=solve(m,seed,feet,n,d,key[0],key[1],gap=.001,y=origin[1])
            T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]); f=m.fk(qi)
            pelvis=apply(T,m.vertices['pelvis']); depth=pelvis@n+d; patch=depth<depth.min()+.002; local=(pelvis-center)@SR
            bias,stat=static_bias(m,inertia,qi,root,float(pelvis[patch,0].mean()),0.)
            foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']])
            sole=foot[foot[:,2]<foot[:,2].min()+.003]; clear=[]; floor=[]
            for name,verts in m.vertices.items():
                world=apply(T@f[name],verts)
                if name not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append((float(world[:,2].min()),name))
                if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                v=(world-center)@SR; v[:,0]-=.4; v=clipped_vertices(v)
                if len(v):clear.append((float(v[:,2].min()),name))
            cmd=command.copy();cmd[legs]=qi[legs]+bias;cmd[arm_ids]+=qi[arm_ids]-q[arm_ids]
            joint_margin=min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in legs+arm_ids)
            v=dict(x=key[0],pitch=key[1],shoulder_offset_rad=key[2],elbow_offset_rad=key[3],clearance_m=min(clear)[0],nearest_body=min(clear)[1],cop_margin_m=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x'])),patch_margin_m=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),floor_clearance_m=min(floor)[0],nearest_floor_body=min(floor)[1],joint_command_margin_rad=float(joint_margin),ik_error=float(error),gap_m=float(depth.min()),q=qi.tolist(),root=root.tolist(),command=cmd.tolist(),arm_displacement_squared=float(key[2]**2+key[3]**2))
            v['constraint_pass']=bool(v['clearance_m']>=.007 and v['cop_margin_m']>=.005 and v['patch_margin_m']>=.015 and v['floor_clearance_m']>=.005 and v['joint_command_margin_rad']>=.03 and error<=.003)
            cache[key]=v;history.append(v);return v
        def constraints(x):
            v=evaluate(x)
            return np.array([(v['clearance_m']-.00725)*100,(v['cop_margin_m']-.00525)*100,(v['patch_margin_m']-.01525)*100,(v['floor_clearance_m']-.00525)*100,v['joint_command_margin_rad']-.031,(.0029-v['ik_error'])*100])
        def objective(x):return x[2]**2+x[3]**2+25*max(0.,x[1]-.9)**2
        searches=[]
        for start in [(-.22,.92,-1.4,-.6),(-.24,.925,-1.9,-.8)]:
            fit=minimize(objective,start,method='SLSQP',bounds=[(-.26,-.1),(.3,float(np.deg2rad(53.))),(-2.1,0.),(-1.1,0.)],constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=80,ftol=1e-9,eps=1e-5))
            searches.append(dict(start=start,success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),endpoint=evaluate(fit.x)))
        feasible=[v for v in history if v['constraint_pass']]
        best=min(feasible,key=lambda v:v['arm_displacement_squared']) if feasible else None
        row=dict(case=case,measured_height_m=h,measured_slope_deg=slope,recorded_start_s=1.,input_q=q.tolist(),input_command=command.tolist(),input_gravity=g.tolist(),input_geometry=geom.tolist(),input_reference=[float(tr['estimated_root_x'][at,case]),0.,0.],searches=searches,best_tested_feasible=best,evaluations=len(history),history=history)
        rows.append(row);(OUT/f'case_{case}.json').write_text(json.dumps(row,indent=2));print(case,{k:v for k,v in (best or searches[-1]['endpoint']).items() if k not in ['q','root','command']},flush=True)
    paths=[Path(__file__),mp,ip,cp,source/'trace.npz',source/'sensor_inputs.npz',Path('scripts/build_near_seat_candidates.py'),Path('scripts/sagittal_static_bias.py')]
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline constrained minimum-displacement upper-body posture calculation with actual recorded1s sensors and fixed feet. New bounded research family, shoulder offsets[-2.1,0],elbow[-1.1,0],pitch<=53degrees; this is an explicit expanded offline posture family with2degree nominal margin below the UNCHANGED55degree fall label; no dynamic margin claim. Only two harder geometries searched; original33 preserved.7mm reserve and support screens are model research checks, not changed physical thresholds. No whole-path/dynamic/perception-availability or self-collision certificate, no executable motion.',physics_ticks=0,ppo_updates=0,holdout_used=False,source_input_hashes={str(p.resolve()):sha(p) for p in paths},rows=rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()

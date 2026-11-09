"""Bounded offline endpoint study under the unchanged qualifying tilt limit.

Fixed measured1s feet and previously tested arm offset. Two explicit static
contact assumptions distinguish foot-only approach support from hypothetical
seat-supported endpoint allocation. Neither is an executable motion.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'terminal_screen_42'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    def root(deg):return np.r_[0,0,.45,Rotation.from_euler('y',deg,degrees=True).as_quat()]
    assert not terminal_posture_check(root(np.rad2deg(.9)))['meets_physical_posture_requirement']
    assert terminal_posture_check(root(37.9))['passed']
    assert terminal_posture_check(root(39.))['meets_physical_posture_requirement'] and not terminal_posture_check(root(39.))['passed']
    assert not terminal_posture_check(np.full(7,np.nan))['passed']
    checks=dict(old_09rad_rejected=True,inside_research_reserve_passes=True,physical_and_research_limits_distinct=True,nonfinite_rejected=True)
    (OUT/'contract_checks.json').write_text(json.dumps(checks,indent=2))
    source=D/'normal_landing3s_06/candidate';tr=np.load(source/'trace.npz');sensors=np.load(source/'sensor_inputs.npz')['sensors']
    mp=Path('runs/g1_kinematics_geometry_04/model.json');ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json'
    m=BodyModel(mp);inertia=read(ip);cal=read(cp);md=read(mp)
    limits={j['name']:np.deg2rad(j['limits']) for j in md['joints'] if j['type']=='PhysicsRevoluteJoint'}
    arms=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    legs=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    at=int(np.argmin(abs(tr['t']-1.)));rows=[]
    for case in [0,2,6,10]:
        q=tr['q'][at,case].astype(float);command=tr['command'][at,case].astype(float);frame=sensors[0,10,case,270:405];g=frame[69:72].astype(float);geom=tr['geometry'][at,case]
        assert np.array_equal(frame[:23],q[:23].astype(np.float32)) and geom[11] and geom[12]<=.5
        n,d,origin,h,slope,_,_=measured(m,q,g,geom,[tr['estimated_root_x'][at,case],0,0],cal);R=level_rotation(g);fk=m.fk(q)
        feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
        SR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0,h]);seed=q.copy();seed[arms]-=.35
        for assumption,fraction in [('foot_only_before_contact',0.),('hypothetical_seat_supported_endpoint',1.)]:
            cache={};history=[]
            def evaluate(x):
                key=tuple(float(v) for v in x)
                if key in cache:return cache[key]
                qi,rootpose,err=solve(m,seed,feet,n,d,key[0],key[1],gap=.001,y=origin[1]);T=transform(Rotation.from_quat(rootpose[3:]).as_matrix(),rootpose[:3]);f=m.fk(qi)
                pelvis=apply(T,m.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002;local=(pelvis-center)@SR
                bias,stat=static_bias(m,inertia,qi,rootpose,float(pelvis[patch,0].mean()),fraction)
                foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
                clear=[];floor=[]
                for name,verts in m.vertices.items():
                    world=apply(T@f[name],verts)
                    if name not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append((float(world[:,2].min()),name))
                    if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                    v=(world-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
                    if len(v):clear.append((float(v[:,2].min()),name))
                cmd=command.copy();cmd[legs]=qi[legs]+bias;cmd[arms]+=qi[arms]-q[arms]
                margin=min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in legs+arms)
                v=dict(x=key[0],pitch_rad=key[1],clearance_m=min(clear)[0],nearest_body=min(clear)[1],cop_margin_m=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x'])),patch_margin_m=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),floor_clearance_m=min(floor)[0],joint_command_margin_rad=float(margin),ik_error=float(err),terminal=terminal_posture_check(rootpose),hypothetical_vertical_seat_share=stat['predicted_seat_bw'],q=qi.tolist(),root=rootpose.tolist(),command=cmd.tolist())
                v['other_screens_passed']=bool(v['terminal']['passed'] and v['cop_margin_m']>=.005 and v['patch_margin_m']>=.015 and v['floor_clearance_m']>=.005 and margin>=.03 and err<=.003 and (fraction==0 or stat['predicted_seat_bw']*np.cos(np.deg2rad(slope))>=.05))
                v['all_endpoint_screens_passed']=bool(v['other_screens_passed'] and v['clearance_m']>=.003)
                cache[key]=v;history.append(v);return v
            def constraint(x):
                v=evaluate(x);return np.array([(v['cop_margin_m']-.00525)*100,(v['patch_margin_m']-.01525)*100,(v['floor_clearance_m']-.00525)*100,v['joint_command_margin_rad']-.031,(.0029-v['ik_error'])*100,(38.-v['terminal']['model_tilt_deg'])*.1]+([(v['hypothetical_vertical_seat_share']*np.cos(np.deg2rad(slope))-.052)*10] if fraction else []))
            searches=[]
            for start in [(-.19,np.deg2rad(37.)),(-.23,np.deg2rad(25.))]:
                fit=minimize(lambda x:-evaluate(x)['clearance_m']*100,start,method='SLSQP',bounds=[(-.28,-.1),(np.deg2rad(5.),np.deg2rad(38.))],constraints=[dict(type='ineq',fun=constraint)],options=dict(maxiter=55,ftol=1e-9,eps=1e-5))
                searches.append(dict(success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),endpoint=evaluate(fit.x)))
            valid=[v for v in history if v['other_screens_passed']];best=max(valid,key=lambda v:v['clearance_m']) if valid else None
            row=dict(case=case,measured_height_m=h,measured_slope_deg=slope,assumption=assumption,evaluations=len(history),best_other_screens_feasible=best,searches=searches,history=history);rows.append(row)
            (OUT/f'case_{case}_{assumption}.json').write_text(json.dumps(row,indent=2));print(case,assumption,{k:best[k] for k in ['clearance_m','pitch_rad','x','hypothetical_vertical_seat_share','all_endpoint_screens_passed']} if best else 'NO OTHER-SCREEN FEASIBLE POINT',flush=True)
    paths=[Path(__file__),Path('scripts/low_seat_terminal_contract_42.py'),Path('sitting_task/src/humanoid_sitting/resting_objective.py'),mp,ip,cp,source/'trace.npz',source/'sensor_inputs.npz']
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_ticks=0,optimizer_updates=0,holdout_used=False,executable_motion=False,
        scope='Bounded2-start local endpoint searches. Fixed actual1s feet and fixed-.35rad shoulder/elbow offsets, rootx[-.28,-.10],pitch5–38deg. The38deg research screen reserves2deg under the UNCHANGED40deg physical support criterion. Existing3mm model clearance/5mm footCoP/15mm pelvis patch screens. Hypothetical seat support is vertical static allocation, not measured legal loading. No step, path, rate, dynamics or support-transition certificate. Limited failures do not prove impossibility.',
        source_input_sha256={str(p.resolve()):sha(p) for p in paths},contract_checks=checks,rows=rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()

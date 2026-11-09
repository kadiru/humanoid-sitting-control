"""Independent arithmetic/FK audit of offline44/45, no physical admission."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, transform
from low_seat_terminal_contract_42 import terminal_posture_check
from low_seat_joint_path_controller_17 import measured
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'model_audit_v2_46'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    a=read(D/'rolling_contact_screen_v2_44/results.json');b=read(D/'support_authority_45/results.json')
    for source in [a['sources'],b['source_input_sha256']]:
        for p,h in source.items():assert sha(p)==h,p
    original=Path('scripts/screen_low_seat_rolling_contact_44.py').read_text();fixed=Path('scripts/screen_low_seat_rolling_contact_v2_44.py').read_text()
    assert fixed.replace("OUT=D/'rolling_contact_screen_v2_44'","OUT=D/'rolling_contact_screen_44'").replace('whole_path_model_passed=bool(not failed and rates.max()<=.175)','whole_path_model_passed=not failed and rates.max()<=.175')==original
    src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');sen=np.load(src/'sensor_inputs.npz')['sensors']
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');inertia=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json')
    mass=np.asarray(inertia['mass']);results=[];at=int(np.argmin(abs(tr['t']-1.)));assert abs(tr['t'][at]-1.)<1e-9
    for row in a['rows']:
        case=row['case'];q0=tr['q'][at,case].astype(float);frame=sen[0,10,case,270:405]
        n,d,origin,h,slope,_,_=measured(m,q0,frame[69:72],tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],cal)
        n=np.asarray(n);n/=np.linalg.norm(n);tangent=np.array([n[2],0.,-n[0]]);tangent/=np.linalg.norm(tangent)
        sr=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);previous=None;errors=[];roots=[];foot_initial=None;foot_errors=[];geometric_failures=0
        for r in row['path']:
            q=np.asarray(r['q']);root=np.asarray(r['root']);R=Rotation.from_quat(root[3:]).as_matrix();T=transform(R,root[:3]);fk=m.fk(q)
            p=apply(T,m.vertices['pelvis']);depth=p@n+d;patch=depth<depth.min()+.002;contactx=float(p[patch,0].mean())
            com=np.average([apply(T@fk[body],c) for body,c in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
            foot=np.concatenate([apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
            target=float(sole[:,0].min()+.0055);required=max(0.,(target-com[0])/(target-contactx));share=r['achievable_static_allocation_share']
            cop=(com[0]-share*contactx)/(1-share)
            errors.extend([abs(required-r['assumed_minimum_vertical_seat_share']),abs(min(cop-sole[:,0].min(),sole[:,0].max()-cop)-r['foot_cop_margin_m']),abs(depth.min()-r['pelvis_gap_m'])])
            if previous is not None:
                oldroot,oldR,oldpoint=previous
                residue=float(tangent@(root[:3]+R@oldpoint-oldroot[:3]-oldR@oldpoint))
                errors.append(abs(residue-r['previous_contact_material_tangent_residual_m']))
            previous=(root,R,m.vertices['pelvis'][int(np.argmin(depth))])
            footposes=np.array([T@fk[s+'_ankle_roll_link'] for s in ['left','right']])
            if foot_initial is None:foot_initial=footposes
            foot_errors.append(float(abs(footposes-foot_initial).max()))
            clear=[]
            for name,verts in m.vertices.items():
                if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                v=(apply(T@fk[name],verts)-center)@sr;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):clear.append(float(v[:,2].min()))
            errors.append(abs(min(clear)-r['clearance_m']))
            if min(clear)<.003:geometric_failures+=1
            assert terminal_posture_check(root)==r['terminal_check']
            roots.append(root)
        times=np.array([r['time_s'] for r in row['path']]);cmd=np.array([r['command'] for r in row['path']]);rates=np.abs(np.diff(cmd,axis=0))/np.diff(times)[:,None]
        assert abs(float(rates.max())-row['nominal_peak_command_rate_rad_s'])<1e-12
        assert max(errors)<1e-9,(case,max(errors))
        assert row['whole_path_model_passed'] is False
        # Arc lower bound is necessary, not a schedule or controller change.
        arc=float(np.max(abs(np.diff(cmd,axis=0)),axis=1).sum())
        results.append(dict(case=case,path_knots=len(times),maximum_independent_numeric_error=max(errors),maximum_relative_foot_transform_component_error=max(foot_errors),clearance_failed_knots=geometric_failures,peak_command_rate_rad_s=float(rates.max()),command_arc_rad=arc,necessary_constant_rate_seconds_at_015=arc/.15,necessary_constant_rate_seconds_at_0175=arc/.175,original_model_gate_passed=False))
    authority=[]
    for row in b['rows']:
        ep=row['endpoint'];root=np.asarray(ep['root']);q=np.asarray(ep['q']);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q)
        com=np.average([apply(T@fk[body],c) for body,c in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
        assert np.max(abs(com-ep['com']))<1e-12
        assert abs(row['maximum_forward_com_gain_m']-(row['maximum_bounded_arm_com'][0]-row['fixed_previous_arm_com'][0]))<1e-12
        assert terminal_posture_check(root)==ep['terminal'] and ep['assumed_seat_share']==0
        authority.append(dict(case=row['case'],com_replay_max_error=float(np.max(abs(com-ep['com']))),physical_admission=False))
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,physical_admission=False,source_input_hashes_passed=True,serialization_only_recovery_verified=True,rolling=results,authority=authority,physics_ticks=0,optimizer_updates=0,holdout_used=False,limits='Checks recorded model arithmetic/FK/clearance/rates and source identity. Static allocation is hypothetical. No contact generation, friction/dynamics, collision cooking or executable approach certificate. Arm authority has no self-collision/path validation. All44 model gates remain failed.')
    (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

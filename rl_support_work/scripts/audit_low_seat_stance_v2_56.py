"""Independent model-data and prerequisite audit; no physical admission."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from plan_seat_load_targets import clipped_vertices
from low_seat_terminal_contract_42 import terminal_posture_check
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'stance_independent_audit_v2_56'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=False);a=read(D/'stance_endpoint_screen_52/results.json');b=read(D/'supported_step_commands_v3_56/results.json')
 for source in [a,b]:
  assert source['physics_ticks']==source['optimizer_updates']==0 and not source['holdout_used']
  for p,h in source['source_input_sha256'].items():assert sha(p)==h,p
 old=Path('scripts/screen_low_seat_supported_step_53.py').read_text();new=Path('scripts/screen_low_seat_supported_step_v2_53.py').read_text();assert new.replace("OUT=D/'supported_step_screen_v2_53'","OUT=D/'supported_step_screen_53'").replace('  gates={k:bool(v) for k,v in gates.items()}\n','')==old
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1.)));assert abs(tr['t'][at]-1)<1e-9
 def context(case):
  q=tr['q'][at,case].astype(float);g=ss[0,10,case,339:342];assert np.array_equal(q[:23].astype(np.float32),ss[0,10,case,270:293]);n,d,o,h,slope,_,_=measured(m,q,g,tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],cal);R=level_rotation(g);f=m.fk(q);feet={s:(R@f[s+'_ankle_roll_link'][:3,3]+o,R@f[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']};return q,n,d,o,h,slope,feet
 def geometry(q,root,h,slope):
  T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(q);SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);clear=[];floor=[]
  for name,v in m.vertices.items():
   world=apply(T@f[name],v)
   if name not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(world[:,2].min()))
   if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   v=(world-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
   if len(v):clear.append(float(v[:,2].min()))
  C=np.average([apply(T@f[bb],c) for bb,c in zip(ine['body_names'],ine['local_com'])],axis=0,weights=ine['mass']);return T,f,min(clear) if clear else None,min(floor),C
 endpoints=[]
 for row in a['rows']:
  ep=row['best_other_screens_feasible'];q0,n,d,o,h,slope,feet=context(row['case']);q=np.asarray(ep['q']);root=np.asarray(ep['root']);T,f,clear,floor,C=geometry(q,root,h,slope)
  pp=apply(T,m.vertices['pelvis']);depth=pp@n+d;patch=depth<depth.min()+.002;seatx=pp[patch,0].mean();fx=np.mean([(T@f[s+'_ankle_roll_link'])[0,3] for s in feet]);share=0. if row['assumption']=='foot_only_before_contact' else float(np.clip((fx-C[0])/(fx-seatx),0,.9));cop=(C[0]-share*seatx)/(1-share)
  foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in feet]);sole=foot[foot[:,2]<foot[:,2].min()+.003];copmargin=min(cop-sole[:,0].min(),sole[:,0].max()-cop)
  pe=max(np.linalg.norm((T@f[s+'_ankle_roll_link'])[:3,3]-(p+[ep['foot_shift_m'],0,0])) for s,(p,r) in feet.items());re=max(np.linalg.norm(Rotation.from_matrix(r.T@(T@f[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s,(p,r) in feet.items())
  errors=[abs(clear-ep['clearance_m']),abs(floor-ep['floor_clearance_m']),abs(copmargin-ep['cop_margin_m']),abs(share-ep['hypothetical_vertical_seat_share']),abs(pe-ep['full_foot_position_error_m']),abs(re-ep['full_foot_rotation_error_rad'])];assert max(errors)<1e-9
  assert terminal_posture_check(root)==ep['terminal'];assert abs(ep['foot_shift_m'])<=.06+1e-12
  endpoints.append(dict(case=row['case'],assumption=row['assumption'],clearance_m=clear,foot_shift_m=ep['foot_shift_m'],maximum_replay_error=max(errors),endpoint_screen_passed=ep['all_endpoint_screens_passed'],physical_admission=False))
 steps=[];q0,n,d,o,h,slope,feet=context(2)
 nonleg=[i for i,nm in enumerate(m.names) if not any(x in nm for x in ['hip_','knee_','ankle_'])];fingers=[i for i,nm in enumerate(m.names) if any('_'+x+'_joint' in nm for x in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
 original=read(D/'supported_step_screen_v2_53/results.json')
 for row in b['rows']:
  errors=[];force_moment=[];stance_drift=[];horizontal_while_low=[]
  for j,nd in enumerate(row['nodes']):
   q=np.asarray(nd['q']);root=np.asarray(nd['root']);T,f,clear,floor,C=geometry(q,root,h,slope);soles={}
   for s in feet:
    vv=apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']);soles[s]=vv[vv[:,2]<vv[:,2].min()+.003]
   poly=np.concatenate([soles[s] for s in nd['support']]);H=ConvexHull(poly[:,:2]).equations;mar=float(np.min(-(H[:,:2]@C[:2]+H[:,2])))
   pe=max(np.linalg.norm((T@f[s+'_ankle_roll_link'])[:3,3]-nd['foot_targets'][s]) for s in feet);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@f[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in feet)
   errors.extend([float(np.max(abs(C-nd['com']))),abs(mar-nd['static_cop_margin_m']),abs(pe-nd['foot_pose_error_m']),abs(re-nd['foot_rotation_error_rad']),abs(floor-nd['floor_clearance_m'])]);
   if clear is not None:errors.append(abs(clear-nd['forbidden_seat_clearance_m']))
   if j and nd['phase']==row['nodes'][j-1]['phase']:
    prev=row['nodes'][j-1]
    for s in feet:
     dx=abs(nd['foot_targets'][s][0]-prev['foot_targets'][s][0]);z=min(nd['foot_targets'][s][2],prev['foot_targets'][s][2])-feet[s][0][2]
     if dx>1e-12:horizontal_while_low.append(z)
     if s in nd['support'] and s in prev['support']:stance_drift.append(float(np.linalg.norm(np.asarray(nd['foot_targets'][s])-prev['foot_targets'][s])))
  assert max(errors)<1e-9;assert max(stance_drift,default=0)<1e-12;assert min(horizontal_while_low)>.034
  commands=np.asarray([x['command'] for x in row['nodes']]);assert np.all(commands[:,fingers]==0);assert np.array_equal(commands[:,nonleg],np.broadcast_to(tr['command'][at,2,nonleg],commands[:,nonleg].shape));oldrow=next(x for x in original['rows'] if x['shift_m']==row['shift_m']);assert all(all(n[k]==o[k] for k in n if k!='command') for n,o in zip(row['nodes'],oldrow['nodes']));arc=float(np.sum(np.max(abs(np.diff(commands,axis=0)),axis=1)));assert abs(arc-row['necessary_command_arc_rad'])<1e-12
  initialjump=float(np.max(abs(commands[0]-tr['command'][at,2])));qcmd=np.asarray([x['q'] for x in row['nodes']]);posearc=float(np.sum(np.max(abs(np.diff(qcmd,axis=0)),axis=1)))
  steps.append(dict(shift_m=row['shift_m'],sampled_nodes=len(row['nodes']),maximum_replay_error=max(errors),maximum_stance_target_motion_m=max(stance_drift),minimum_lift_during_horizontal_translation_m=min(horizontal_while_low),command_arc_rad=arc,necessary_command_seconds_at015=arc/.15,necessary_command_seconds_with_initial_jump_at015=(arc+initialjump)/.15,initial_command_join_rad=initialjump,kinematic_pose_arc_rad=posearc,original_gate=row['gate'],original_all_prerequisite_gates_passed=row['all_prerequisite_gates_passed'],physical_admission=False,fixed14_fingers_exact=True,nonleg_commands_exact_recorded=True))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,physical_admission=False,physics_ticks=0,optimizer_updates=0,holdout_used=False,source_hashes_passed=True,serialization_only_recovery_verified=True,endpoints=endpoints,steps=steps,limitations='Numerical model/FK/clearance/CoP/command-arc replay only. Stepping samples hold stance target fixed and lift before horizontal swing; not a simulator step. Actual forces, dynamic ZMP/friction, continuous collision/IK, self-collision, sensing/tracking and subsequent contact transfer remain unvalidated. The recorded positive initial command join is additional required motion, not discarded. Static-source command bounds/timing are necessary screens, not dynamic certification. Failed model gates stay failed.')
 (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['endpoints','steps']},indent=2))
if __name__=='__main__':main()

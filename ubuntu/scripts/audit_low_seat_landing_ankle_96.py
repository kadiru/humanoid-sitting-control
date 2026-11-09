"""Independent replay of offline single-step poses; no admission or simulation."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'landing_ankle_path_audit_96'
read=lambda p:json.loads(Path(p).read_text())
def smooth(u):return u*u*(3-2*u)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 O.mkdir(exist_ok=False);source=D/'landing_ankle_path_95/results.json';data=read(source)
 for p,h in data['source_input_sha256'].items():assert sha(p)==h,p
 z=data['dense'];res=dict(rows=[dict(source=str(source),parameters=z['parameters'],length_m=data['endpoint']['parameters'][5],endpoint=data['endpoint'],path=z['path'],pose_arc_rad=z['pose_arc_rad'],pose_only_necessary_time_at_existing_cap_s=z['nominal_pose_schedule_at_0_15_rad_s'],failed_knots=len(z['failed_knots']),first_failed_knot=z['failed_knots'][0] if z['failed_knots'] else None,whole_path_screen_passed=z['model_geometry_passed'] and z['nominal_pose_schedule_at_0_15_rad_s']<=13.5,minimum_swing_target_lift_during_horizontal_motion_m=z['minimum_target_lift_while_horizontal_m'],min_clearance_m=z['min_clearance_m'],max_foot_position_error_m=z['max_foot_position_error_m'],max_foot_orientation_error_rad=z['max_foot_orientation_error_rad'])])
 m=BodyModel(D/'cooked_model_50/model.json');inertia=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];c=2;at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);f0=m.fk(q0);sides=['left','right'];feet0={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};mass=np.array(inertia['mass']);SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in json.loads((D/'cooked_model_50/model.json').read_text())['joints'] if j['type']=='PhysicsRevoluteJoint'};legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];fixed=[j for j in range(len(m.names)) if j not in arms+legs];out=[]
 for row in res['rows']:
  assert row['endpoint'] is not None
  records=[('endpoint',row['endpoint'])]+[(f'knot_{i}',e) for i,e in enumerate(row['path'])];errors=[];failed=[];qpath=[]
  for label,e in records:
   q=np.array(e['q']);root=np.array(e['root']);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);assert np.isfinite(q).all() and np.isfinite(root).all();assert np.array_equal(q[fixed],q0[fixed]);
   for j in legs:assert limits[m.names[j]][0]+.031-1e-10<=q[j]<=limits[m.names[j]][1]-.031+1e-10
   f=e.get('approach_fraction',1.);
   if label!='endpoint':
    u=e['u'];a,b,lean,pitch_bump=row['parameters'];assert 0<=pitch_bump<=np.deg2rad(15); wanted=a*smooth(u) if e['phase']=='shift' else (a+(b-a)*smooth(u) if e['phase']=='swing' else b+(1-b)*smooth(u) if e['phase']=='approach' else 1.);assert abs(f-wanted)<1e-13
   assert np.max(abs(q[arms]-(q0[arms]-.35*f)))<1e-13
   feet={s:(np.array(e['foot_targets'][s]['position']),np.array(e['foot_targets'][s]['rotation'])) for s in sides};assert np.array_equal(feet['left'][0],feet0['left'][0]) and np.array_equal(feet['left'][1],feet0['left'][1]);
   yaw=data['endpoint']['parameters'][6]*(smooth(np.clip((e.get('u',1.)-.2)/.6,0,1)) if e.get('phase')=='swing' else 0. if e.get('phase')=='shift' else 1.);baseR=Rotation.from_euler('z',yaw).as_matrix()@feet0['right'][1];delta=Rotation.from_matrix(baseR.T@feet['right'][1]).as_euler('xyz');bound=e.get('swing_attitude_bound_rad',0.);assert abs(delta[2])<1e-12 and max(abs(delta[:2]))<=bound+1e-12
   if e.get('phase')!='swing':assert np.max(abs(feet['right'][1]-baseR))<1e-14
   C=np.average([apply(T@fk[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass);foot={s:apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in sides};v=np.concatenate([foot[s] for s in e['stance']]);sole=v[v[:,2]<v[:,2].min()+.003]
   try:H=ConvexHull(sole[:,:2]).equations;cop=float(np.min(-(H[:,:2]@C[:2]+H[:,2])));deg=False
   except QhullError:cop=-1.;deg=True
   pelvis=apply(T,m.vertices['pelvis']);dep=pelvis@n+d;patch=dep<dep.min()+.002;local=(pelvis-center)@SR;pm=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()));clear=[];floor=[]
   for b,v in m.vertices.items():
    w=apply(T@fk[b],v)
    if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
    if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
    v=(w-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
    if len(v):clear.append((float(v[:,2].min()),b))
   mi=min(clear) if clear else (1.,'none_over_finite_seat');pe=max(np.linalg.norm((T@fk[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in sides);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@fk[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in sides);term=terminal_posture_check(root);vals=dict(clearance_m=mi[0],pelvis_gap_m=float(dep.min()),patch_margin_m=pm,foot_cop_margin_m=cop,floor_clearance_m=min(floor),foot_position_error_m=pe,foot_orientation_error_rad=re);err=max(abs(e[k]-v) for k,v in vals.items());assert err<1e-9;errors.append(err);assert e['nearest_body']==mi[1] and e['support_degenerate']==deg and e['terminal']==term
   other=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and term['passed'] and (dep.min()>=.02 or pm>=.015) and dep.min()>=-1e-8);passed=bool(other and mi[0]>=.003);assert e['other_model_passed']==other and e['model_pose_passed']==passed
   if label!='endpoint':
    i=int(label.split('_')[1]);qpath.append(q)
    if not passed:failed.append(i)
  path=row['path'];
  for e in path:
   if e['phase']=='swing':assert abs(e['swing_attitude_bound_rad']-np.deg2rad(8)*min(smooth(min(e['u']/.2,1)),smooth(min((1-e['u'])/.2,1))))<1e-12
  Q=np.array(qpath);arc=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum());assert abs(arc-row['pose_arc_rad'])<1e-10 and abs(arc/.15-row['pose_only_necessary_time_at_existing_cap_s'])<1e-9
  assert row['failed_knots']==len(failed) and row['first_failed_knot']==(failed[0] if failed else None);assert row['whole_path_screen_passed']==bool(not failed and arc/.15<=13.5)
  moves=[]
  for a,b in zip(path,path[1:]):
   p=np.array(a['foot_targets']['right']['position']);q=np.array(b['foot_targets']['right']['position'])
   if np.linalg.norm((p-q)[:2])>1e-8:moves.append(min(p[2],q[2])-feet0['right'][0][2])
  assert abs(min(moves)-row['minimum_swing_target_lift_during_horizontal_motion_m'])<1e-12 and min(moves)>.03
  assert np.max(abs(np.array(path[-1]['foot_targets']['right']['position'])-(feet0['right'][0]+[-row['length_m'],0,0])))<1e-14
  out.append(dict(source=row['source'],parameters=row['parameters'],length_m=row['length_m'],knots=len(path),max_replay_error=max(errors),failed_knots=failed,pose_schedule_metric_passed=row['whole_path_screen_passed'],min_clearance_mm=row['min_clearance_m']*1000,max_foot_position_error_mm=row['max_foot_position_error_m']*1000,max_foot_orientation_error_deg=np.rad2deg(row['max_foot_orientation_error_rad']),nominal_pose_schedule_at_0_15_rad_s=arc/.15,min_actual_swing_sole_height_m=min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8)))
 # Independent actuator/virtual-work replay for every saved dense knot.
 command0=tr['command'][at,c].astype(float);moving=legs+arms;kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);W=mass.sum()*9.81;commands=[];tensors=[];bounds=[];allocation_fail=[];virtual_errors=[];max_equilibrium=0.;effort_use=[]
 for k,(e,v) in enumerate(zip(z['path'],z['command_nodes'])):
  q=np.array(e['q']);root=e['root'];T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);C=np.average([apply(T@fk[b],cc) for b,cc in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
  if not v['allocation']['passed']:allocation_fail.append(k)
  if 'command' not in v:continue
  P=np.array(v['corner_world_points']);w=np.array(v['corner_fractions']);sides2=v['corner_sides'];pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);normal=v.get('assumed_seat_normal_fraction',0.);sv=normal/n[2];assert normal==0 or abs(pd.min())<1e-8;eq=max(abs(w.sum()+sv-1),max(abs((w[:,None]*P[:,:2]).sum(0)+sv*seat[:2]-C[:2])));max_equilibrium=max(max_equilibrium,float(eq));assert eq<1e-8 and min(w)>=-1e-10;assert abs(sum(ww for ww,ss in zip(w,sides2) if ss=='right')-v['planned_right_load_fraction'])<1e-8
  if e['phase']=='swing':assert all(abs(ww)<1e-8 for ww,ss in zip(w,sides2) if ss=='right')
  local=[np.linalg.inv(T@fk[ss+'_ankle_roll_link'])@np.r_[p,1.] for p,ss in zip(P,sides2)];tau=[]
  for j in moving:
   values=[]
   for sign in [-1,1]:
    qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);V=sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(inertia['body_names'],mass,inertia['local_com']));work=sum(W*ww*(T@ff[ss+'_ankle_roll_link']@ll)[2] for ww,ss,ll in zip(w,sides2,local));values.append(V-work)
   tau.append((values[1]-values[0])/2e-5)
  tau=np.array(tau);terr=float(max(abs(tau-v['planned_static_torque_Nm'])));assert terr<1e-6;virtual_errors.append(terr)
  if k==0:tau0=tau.copy()
  expected=command0.copy();expected[moving]+=q[moving]-q0[moving]+(tau-tau0)/kp
  if k==0:expected=command0.copy()
  cmd=np.array(v['command']);assert max(abs(cmd-expected))<2e-8;fixed_cmd=[j for j in range(len(m.names)) if j not in moving];assert np.array_equal(cmd[fixed_cmd],command0[fixed_cmd]);cm=float(min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in moving));eu=float(max(abs(tau)/effort));assert abs(cm-v['min_command_margin_rad'])<1e-12 and abs(eu-v['maximum_effort_utilization'])<1e-7;bounds.append(cm);effort_use.append(eu);commands.append(cmd)
 assert len(commands)==len(z['path'])==115;assert np.array_equal(commands[0],command0);command_arc=float(np.max(abs(np.diff(commands,axis=0)),axis=1).sum());assert abs(command_arc-z['command_arc_rad'])<1e-12;assert abs(command_arc/.15-z['minimum_time_for_this_command_polyline_at015_s'])<1e-12
 static_pass=bool(not allocation_fail and min(bounds)>=.03 and max(effort_use)<=1);assert static_pass==z['model_command_passed'];assert bool(command_arc/.15<=13.5)==z['command_schedule_passed']
 ca=dict(nodes=115,max_virtual_work_error_Nm=max(virtual_errors),max_equilibrium_residual=max_equilibrium,minimum_command_margin_rad=min(bounds),maximum_effort_utilization=max(effort_use),force_allocation_failures=allocation_fail,model_command_passed=static_pass,specific_command_time_s=command_arc/.15,command_schedule_passed=z['command_schedule_passed'],initial_command_exact=True,all_fixed_commands_exact=True)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent all115-node corrected geometry/full-foot/stance/airborne-attitude/tilt/joint replay plus separate16joint finite-difference virtual work, nonnegative static force equilibrium, zero modeled seat loading before zero-gap contact, measured loads NOT established, exact recorded initial/fixed commands, command bounds and specific command-polyline timing. Reports separate geometry/command/time gates. No actual commands applied, no sensed support/dynamics/physical admission. A model pass is not permission to start PPO or holdout.',source_sha256={str(source):sha(source),str(Path(__file__).resolve()):sha(__file__)},geometry=out,command=ca,physics_ticks=0,optimizer_updates=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

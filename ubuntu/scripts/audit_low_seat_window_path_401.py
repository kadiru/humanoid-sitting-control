"""Independent complete-path arithmetic audit400. Rebuild geometry/virtual work/original rows, the continuous-window envelope and all675 idealclock samples. Independently replay every linear primal and nonlinear-acceptance decision from saved arrays; no optimization or projection. Arithmetic pass is not candidate admission."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'window_path_audit_401'
read=lambda p:json.loads(Path(p).read_text())
def smooth(u):return u*u*(3-2*u)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def audit_steps(fixture,data):
 current=fixture['v'].copy();history=data['algorithm']['history'];records=[];trust=1.
 scale=fixture['scale'];keep=fixture['keep'];radius=np.r_[np.full(786,.02),np.full(109,.1),np.full(5,5.)]
 for r in history:
  folder=D/'window_path_400'/('step_%02d'%r['iteration']);f=dict(np.load(folder/'linear_problem.npz'));assert np.array_equal(f['v'],current);assert float(f['radius'])==trust
  m=int(keep.sum());A=np.zeros((m+3,901));low=np.full(m+3,-np.inf);high=np.full(m+3,np.inf)
  A[:m,:900]=f['I'][keep]/scale[keep,None];A[:m,900]=1.;low[:m]=-f['ineq'][keep]/scale[keep]
  A[m,:900]=f['E'][0];A[m,900]=1.;low[m]=-f['eq'][0]
  A[m+1,:900]=-f['E'][0];A[m+1,900]=1.;low[m+1]=f['eq'][0]
  A[m+2,895:900]=1.;low[m+2]=high[m+2]=0.
  assert np.array_equal(A,f['A']) and np.array_equal(low,f['lower']) and np.array_equal(high,f['upper'])
  assert np.array_equal(f['cost'],np.r_[np.zeros(900),1.]) and np.array_equal(f['integrality'],np.r_[np.zeros(895),np.ones(5),0])
  assert np.all(f['bounds_lower'][:900]>=-radius*trust) and np.all(f['bounds_upper'][:900]<=radius*trust)
  if not r['candidate_available']:records.append(dict(iteration=r['iteration'],matrix_exact=True,candidate_available=False));break
  u=np.load(folder/'solution.npy');au=A@u;lin=min(float((au-low)[np.isfinite(low)].min()),float((high-au)[np.isfinite(high)].min()));bound=min(float((u-f['bounds_lower']).min()),float((f['bounds_upper']-u).min()));integer=float(abs(u[895:900]-np.rint(u[895:900])).max())
  assert abs(lin-r['linear_minimum'])<1e-12 and abs(bound-r['bound_minimum'])<1e-12 and integer==r['integer_decode_error']
  passed=bool(lin>=-1e-7 and bound>=-1e-7 and integer<1e-7);assert passed==r['linear_numerical_passed']
  row=dict(iteration=r['iteration'],matrix_exact=True,linear_minimum=lin,bound_minimum=bound,integer_error=integer,passed=passed)
  if not passed:records.append(row);break
  trial=np.load(folder/'trial.npy');wanted=current+u[:900];wanted[895:900]=current[895:900]+np.rint(u[895:900]);assert np.array_equal(trial,wanted)
  if not (folder/'trial_constraints.npz').exists():assert 'domain failure' in r['termination'];records.append(row);break
  q=dict(np.load(folder/'trial_constraints.npz'));before=max(0.,float(abs(f['eq']).max()),float((-f['ineq'][keep]/scale[keep]).max()));after=max(0.,float(abs(q['eq']).max()),float((-q['ineq'][keep]/scale[keep]).max()));pred=before-float(u[-1]);reduction=before-after;ratio=reduction/pred if pred>0 else None;accept=bool(pred>0 and reduction>0 and ratio>=.1)
  assert abs(before-r['current_merit'])<1e-12 and abs(after-r['trial_merit'])<1e-12 and accept==r['accepted']
  if accept:
   current=trial
   if ratio>.75:trust=min(1.,2*trust)
  else:trust*=.5
  row.update(accepted=accept,actual_merit=after,acceptance_ratio=ratio);records.append(row)
 assert np.array_equal(current,np.load(D/'window_path_400/parameters.npy'))
 assert len(history)<=12 and sum(r['accepted'] for r in history)==data['algorithm']['accepted_steps']
 return dict(records=records,unchanged_independent_linear_bound=1e-7,maximum_subproblems=12,source_matrices_rebuilt_exactly=True,nonlinear_acceptance_replayed=True,independent_global_feasibility_or_optimality_certificate=False)

def main():
 O.mkdir(exist_ok=False);source=D/'window_path_400/results.json';data=read(source)
 for p,h in data['source_input_sha256'].items():assert sha(p)==h,p
 z=data['dense'];original_data=read(D/'clearance_lift_path_330/results.json');original=original_data['dense']['path'];changed=data['revision']['changed_pose_knots'];Q=np.array([v['q'] for v in z['path']]);z['pose_arc_rad']=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum());z['nominal_pose_schedule_at_0_15_rad_s']=z['pose_arc_rad']/.15;z['minimum_target_lift_while_horizontal_m']=original_data['dense']['minimum_target_lift_while_horizontal_m'];z['command_arc_rad']=float(np.max(abs(np.diff(np.array([v['command'] for v in z['command_nodes']]),axis=0)),axis=1).sum());z['minimum_time_for_this_command_polyline_at015_s']=z['command_arc_rad']/.15;z['command_schedule_passed']=z['command_arc_rad']/.15<=13.5;res=dict(rows=[dict(source=str(source),parameters=z['parameters'],length_m=data['endpoint']['parameters'][5],endpoint=data['endpoint'],path=z['path'],pose_arc_rad=z['pose_arc_rad'],pose_only_necessary_time_at_existing_cap_s=z['nominal_pose_schedule_at_0_15_rad_s'],failed_knots=len(z['failed_knots']),first_failed_knot=z['failed_knots'][0] if z['failed_knots'] else None,whole_path_screen_passed=z['model_geometry_passed'] and z['nominal_pose_schedule_at_0_15_rad_s']<=13.5,minimum_swing_target_lift_during_horizontal_motion_m=z['minimum_target_lift_while_horizontal_m'],min_clearance_m=z['min_clearance_m'],max_foot_position_error_m=z['max_foot_position_error_m'],max_foot_orientation_error_rad=z['max_foot_orientation_error_rad'])])
 m=BodyModel(D/'cooked_model_50/model.json');inertia=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];c=2;at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);f0=m.fk(q0);sides=['left','right'];feet0={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};mass=np.array(inertia['mass']);SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];limits={j['name']:np.deg2rad(j['limits']) for j in json.loads((D/'cooked_model_50/model.json').read_text())['joints'] if j['type']=='PhysicsRevoluteJoint'};legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];fixed=[j for j in range(len(m.names)) if j not in arms+legs];out=[]
 for row in res['rows']:
  assert row['endpoint'] is not None
  records=[('endpoint',row['endpoint'])]+[(f'knot_{i}',e) for i,e in enumerate(row['path'])];errors=[];failed=[];qpath=[]
  for label,e in records:
   q=np.array(e['q']);root=np.array(e['root']);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);assert np.isfinite(q).all() and np.isfinite(root).all();assert np.array_equal(q[fixed],q0[fixed]);
   # Dependent joint bounds are reported by primal replay below, not assumed satisfied.
   f=e.get('approach_fraction',1.);
   if label!='endpoint':
    index=int(label.split('_')[1]);u=e['u'];a,b,lean,rearward_shift=row['parameters'];assert 0<=f<=1
    if index in changed:
     prev=row['path'][index-1] # Ordering is replayed below without assuming the optimization passed.
     old=original[index];oldT=transform(Rotation.from_quat(old['root'][3:]).as_matrix(),np.array(old['root'][:3]));oldfk=m.fk(np.array(old['q']));rr=[]
     for side in sides:
      oldF=oldT@oldfk[side+'_ankle_roll_link'];F=T@fk[side+'_ankle_roll_link'];dR=oldF[:3,:3].T@F[:3,:3];rr.extend(F[:3,3]-oldF[:3,3]);rr.extend(.1*np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]]))
     rr.extend(np.array(e['com'][:2])-np.array(e['com_target_xy']));assert abs(max(abs(np.array(rr)))-e['multiknot_constraint_residual'])<1e-12
    else:assert e==original[index]
   if label!='endpoint':
    # Reconstruct prescribed COM target and root attitude independently of IK.
    old69=read(D/'coupled_step_refined_69/results.json');base=np.asarray(old69['dense']['path'][16]['com_target_xy']);leftv=apply(transform(R0,origin)@f0['left_ankle_roll_link'],m.vertices['left_ankle_roll_link']);leftv=leftv[leftv[:,2]<leftv[:,2].min()+.003];AH=ConvexHull(leftv[:,:2]).equations;nearest=AH[int(np.argmin(-(AH[:,:2]@base+AH[:,2]))),:2];axis=np.array([-nearest[1],nearest[0]]);axis*=1 if axis[0]>0 else -1;assert np.max(abs(axis-data['stance_target_constraint']['sole_longitudinal_axis']))<1e-12;target=base-rearward_shift*axis;endC=np.array(row['endpoint']['com']);startC=np.array(row['path'][0]['com']);expected_xy=startC[:2]+smooth(u)*(target-startC[:2]) if e['phase']=='shift' else target if e['phase']=='swing' else target+smooth(u)*(endC[:2]-target) if e['phase']=='approach' else endC[:2]
    assert max(abs(expected_xy-e.get('original_nominal_com_target_xy',e['com_target_xy'])))<1e-12
    interp=Slerp([0,1],Rotation.concatenate([Rotation.from_matrix(R0),Rotation.from_quat(row['endpoint']['root'][3:])]))
    blend=smooth(u) if e['phase']=='shift' else 1. if e['phase']=='swing' else 1-smooth(u) if e['phase']=='approach' else 0.
    expected_R=Rotation.from_rotvec([lean*blend,0,0]).as_matrix()@interp(e.get('orientation_fraction',f)).as_matrix();assert np.max(abs(T[:3,:3]-expected_R))<1e-12
    assert abs(root[2]-(origin[2]+f*(row['endpoint']['root'][2]-origin[2])))<1e-12
    left0=apply(transform(R0,origin)@f0['left_ankle_roll_link'],m.vertices['left_ankle_roll_link']);sole0=left0[left0[:,2]<left0[:,2].min()+.003];LH=ConvexHull(sole0[:,:2]).equations;assert min(-(LH[:,:2]@target+LH[:,2]))>=.0055-1e-12
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
  assert abs(min(moves)-row['minimum_swing_target_lift_during_horizontal_motion_m'])<1e-12
  # New height independently derived from unchanged cooked geometry/rotation targets.
  old=read(D/'clearance_lift_path_330/results.json')['dense']['path']
  required=[]
  for e in old:
   if e['phase']=='swing' and .2<=e['u']<=.8:
    R=np.array(e['foot_targets']['right']['rotation'])
    zmin=min(sum(float(v[j])*float(R[2,j]) for j in range(3)) for v in m.vertices['right_ankle_roll_link'])
    required.append(.005+.003-float(feet0['right'][0][2])-zmin)
  height=max(required)+.005
  assert abs(height-original_data['revision']['height_m'])<1e-14
  unchanged=[];horizontal_actual=[]
  for k,e in enumerate(path):
   if k not in changed:
    assert e==old[k];unchanged.append(k)
   elif e['phase']=='swing':
    u=e['u'];lift=height*smooth(u/.2) if u<.2 else height*smooth((1-u)/.2) if u>.8 else height
    travel=smooth(np.clip((u-.2)/.6,0,1));wanted=feet0['right'][0]+[-row['length_m']*travel,0,lift]
    assert max(abs(wanted-np.array(e['foot_targets']['right']['position'])))<1e-14
    assert e['foot_targets']['right']['rotation']==old[k]['foot_targets']['right']['rotation']
  assert unchanged==data['unchanged_pose_knots']
  for a,b in zip(path,path[1:]):
   if np.linalg.norm(np.array(a['foot_targets']['right']['position'])[:2]-np.array(b['foot_targets']['right']['position'])[:2])>1e-8:
    for e in [a,b]:
     T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));F=T@m.fk(np.array(e['q']))['right_ankle_roll_link']
     horizontal_actual.append(float(apply(F,m.vertices['right_ankle_roll_link'])[:,2].min()))
  horizontal_sole_screen_passed=bool(min(horizontal_actual)>=.005) # Report this original screen even if the optimizer fails.
  assert z['model_geometry_passed']==bool(not failed and min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8)>=.005)
  assert np.max(abs(np.array(path[-1]['foot_targets']['right']['position'])-(feet0['right'][0]+[-row['length_m'],0,0])))<1e-14
  out.append(dict(source=row['source'],parameters=row['parameters'],length_m=row['length_m'],knots=len(path),max_replay_error=max(errors),failed_knots=failed,pose_schedule_metric_passed=row['whole_path_screen_passed'],min_clearance_mm=row['min_clearance_m']*1000,max_foot_position_error_mm=row['max_foot_position_error_m']*1000,max_foot_orientation_error_deg=np.rad2deg(row['max_foot_orientation_error_rad']),nominal_pose_schedule_at_0_15_rad_s=arc/.15,min_actual_swing_sole_height_m=min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8)))
 # Independent actuator/virtual-work replay for every saved dense knot.
 command0=tr['command'][at,c].astype(float);moving=legs+arms;kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);W=mass.sum()*9.81;commands=[];tensors=[];bounds=[];allocation_fail=[];virtual_errors=[];max_equilibrium=0.;effort_use=[]
 for k,(e,v) in enumerate(zip(z['path'],z['command_nodes'])):
  q=np.array(e['q']);root=e['root'];T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);C=np.average([apply(T@fk[b],cc) for b,cc in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
  if min(v['corner_fractions'])<0 or v['max_equilibrium_residual']>=1e-8:allocation_fail.append(k)
  if 'command' not in v:continue
  P=np.array(v['corner_world_points']);w=np.array(v['corner_fractions']);sides2=v['corner_sides'];pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);normal=v.get('assumed_seat_normal_fraction',0.);sv=normal/n[2];assert normal==0 or abs(pd.min())<1e-8;eq=max(abs(w.sum()+sv-1),max(abs((w[:,None]*P[:,:2]).sum(0)+sv*seat[:2]-C[:2])));max_equilibrium=max(max_equilibrium,float(eq));assert abs(eq-v['max_equilibrium_residual'])<1e-12;assert abs(sum(ww for ww,ss in zip(w,sides2) if ss=='right')-v['planned_right_load_fraction'])<1e-8
  if 16<=k<=64:assert all(abs(ww)<1e-8 for ww,ss in zip(w,sides2) if ss=='right')
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
 #Independent reduced-primal reconstruction from saved poses, force coordinates and targets.
 initial=np.load(D/'window_path_400/initial.npz');fin=np.load(D/'window_path_400/final.npz');XX=fin['full_coordinates'];outer=fin['x'];V=outer[:114*9].reshape(114,9);dt=outer[114*9:];assert np.array_equal(V[:,:2],XX[:,2:4]);assert np.array_equal(V[:,2:4],XX[:,21:23]);assert np.array_equal(V[:,4:],XX[:,16:21]);lo=initial['command_lo'];hi=initial['command_hi'];rates=initial['rates'];dependent=np.array([0,1]+list(range(4,16)));local=[];Y=[];gap=[];maximum_kinematic=0.
 for k in range(1,115):
  i=k-1;e=path[k];old=original[k];q=np.array(e['q']);T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));fk=m.fk(q);ot=transform(Rotation.from_quat(old['root'][3:]).as_matrix(),np.array(old['root'][:3]));of=m.fk(np.array(old['q']));eq=[]
  for side in sides:
   F=T@fk[side+'_ankle_roll_link'];F0=ot@of[side+'_ankle_roll_link'];A=F0[:3,:3].T@F[:3,:3];eq.extend(F[:3,3]-F0[:3,3]);eq.extend(.1*np.array([A[2,1]-A[1,2],A[0,2]-A[2,0],A[1,0]-A[0,1]]))
  C=np.average([apply(T@fk[b],cc) for b,cc in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass);eq.extend(C[:2]-np.array(old['com'][:2])-.1*XX[i,21:23]);
  if k<=103:assert abs(np.array(e['com_target_xy'])-(np.array(old['com'][:2])+.1*XX[i,21:23])).max()<1e-14
  maximum_kinematic=max(maximum_kinematic,float(max(abs(np.array(eq)))))
  w=initial['base_weights'][i]+initial['null'][i]@XX[i,16:21]+initial['wcom'][i]@XX[i,21:23];cnt=int(initial['counts'][i]);assert max(abs(w[:cnt]-z['command_nodes'][k]['corner_fractions']))<1e-14;assert np.array_equal(XX[i,:2],np.array(e['root'][:2]));assert np.array_equal(XX[i,4:16],q[legs]);assert XX[i,2]==e['approach_fraction'] and XX[i,3]==e.get('orientation_fraction',e['approach_fraction'])
  cmd=np.array(z['command_nodes'][k]['command'])[moving];tau=np.array(z['command_nodes'][k]['planned_static_torque_Nm']);local.extend(np.r_[cmd-lo,hi-cmd,effort-tau,effort+tau,w,1-w]);g=float(min(apply(T,m.vertices['pelvis'])@n+d));gap.append(g);Y.append(np.r_[eq,cmd,tau,w,g])
 gap=np.r_[original[0]['pelvis_gap_m'],gap];eq=np.array([(gap[99]-.012)/.012]);dep=np.c_[XX[:103,dependent]-initial['lower'][:103,dependent],initial['upper'][:103,dependent]-XX[:103,dependent]];mono=np.diff(np.vstack([[0.,0.],V[:103,:2],[1.,1.]]),axis=0).ravel();cmd=np.array([v['command'] for v in z['command_nodes']])[:,moving];dd=np.diff(cmd,axis=0);hrhi=(hi-cmd[1:])*dt[:,None]-.18*dd;hrlo=(cmd[1:]-lo)*dt[:,None]+.18*dd
 iq=np.r_[local,dep.ravel(),mono,(rates*dt[:,None]-dd).ravel(),(rates*dt[:,None]+dd).ravel(),hrhi.ravel(),hrlo.ravel(),gap[1:99]-.012,.012-gap[100:105],gap[1:105]+1e-8,gap[65:104]-gap[66:105],.006*dt[99:104]-(gap[99:104]-gap[100:105]),13.5-.5-dt.sum()]
 support=[];support_h=[];support_k=[]
 for k in range(1,104):
  e=original[k];ot=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));of=m.fk(np.array(e['q']));vv=np.concatenate([apply(ot@of[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in e['stance']]);sole=vv[vv[:,2]<vv[:,2].min()+.003];HH=ConvexHull(sole[:,:2]).equations;target=np.array(e['com'][:2])+.1*XX[k-1,21:23]
  support.extend(-(HH[:,:2]@target+HH[:,2])-.005);support_h.extend(HH);support_k.extend([k-1]*len(HH))
 assert np.array_equal(np.array(support_k),initial['support_k']);assert abs(np.array(support_h)-initial['support_h']).max()<1e-12
 iq=np.r_[iq,support];raw_iq=iq.copy();extra=np.load(D/'window_path_400/event_clock.npz');assert abs(raw_iq-extra['raw_ineq']).max()<1e-10;fixture=np.load(D/'event_clock_preparation_v2_386/fixture.npz')
 groups=[(0,16),(16,64),(64,99),(99,104),(104,114)];budgets=list(map(int,extra['budgets']));assert sum(budgets)==650 and budgets[3]>=101
 import math
 durations=[];off=0
 for (a,b),budget in zip(groups,budgets):
  logits=list(map(float,extra['parameters'][786+off:786+off+b-a-1]))+[0.];top=max(logits);weights=[math.exp(v-top) for v in logits];summed=math.fsum(weights);durations.extend(.02+(.02*budget-.02*(b-a))*w/summed for w in weights);off+=b-a-1
 assert abs(np.array(durations)-dt).max()<1e-12
 samples=[cmd[0].copy()];overlaps=[]
 for (a,b),budget in zip(groups,budgets):
  tt=[math.fsum(durations[a:j]) for j in range(a,b+1)]
  for tick in range(1,budget+1):
   now=tick*.02;ov=np.zeros(114)
   for j in range(a,b):ov[j]=max(0.,min(now,tt[j-a+1])-max(now-.02,tt[j-a]))
   overlaps.append(ov)
   if tick==budget:samples.append(cmd[b].copy())
   else:
    j=next(j for j in range(a,b) if now<tt[j-a+1]);alpha=(now-tt[j-a])/durations[j];samples.append(cmd[j]+alpha*(cmd[j+1]-cmd[j]))
  if b==99:
   for _ in range(25):samples.append(cmd[b].copy());overlaps.append(np.zeros(114))
 samples=np.array(samples);difference=np.diff(samples,axis=0);target=10*samples[1:]-9*samples[:-1];clockraw=np.c_[.02*rates-difference,.02*rates+difference,hi-target,target-lo];ulp=extra['ulp'];clockfull=clockraw-np.r_[4*ulp,4*ulp,32*ulp,32*ulp]
 clockerr=float(max(abs(samples-extra['clock']).max(),abs(clockraw-extra['clock_raw']).max(),abs(clockfull-extra['clock_full']).max()));assert clockerr<1e-10
 overlap_error=float(abs(np.array(overlaps)@(np.diff(cmd,axis=0)/dt[:,None])-difference).max());assert overlap_error<1e-10
 tight=np.zeros(20284);epsilon=2.**(-23)
 for i in range(114):
  for j in range(8):
   for off in [64,72]:
    if fixture['keep'][i*80+off+j]:tight[i*80+off+j]=8*epsilon
 tight[12004:12212]=8*epsilon;tight[19760:]=8*epsilon
 for start in [12212,14036]:tight[start:start+1824]=(dt[:,None]/.02*4*ulp).ravel()
 for start in [15860,17684]:tight[start:start+1824]=(dt[:,None]*32*ulp).ravel()
 assert abs(tight-extra['tighten']).max()<1e-12
 # Independent full-time target envelope: every command vertex and vertex+.02.
 times=np.r_[0.,np.cumsum(np.r_[dt[:99],.5,dt[99:]])];vertices=cmd[np.r_[np.arange(100),99,np.arange(100,115)]]
 def cv(t):
  return np.array([np.interp(t,times,vertices[:,j]) for j in range(16)])
 query=np.array([min(float(t+s),float(times[-1])) for t in times for s in [0.,.02]])
 wt=np.array([10*cv(t)-9*cv(t-.02) for t in query]);wr=np.c_[hi-wt,wt-lo];wf=wr-np.r_[32*ulp,32*ulp]
 savedwindow=dict(np.load(D/'window_path_400/window.npz'))
 window_error=float(max(abs(wt-savedwindow['targets']).max(),abs(wr-savedwindow['raw']).max(),abs(wf-savedwindow['tightened']).max()));assert window_error<1e-10
 assert np.all(target.min(0)>=wt.min(0)-1e-12) and np.all(target.max(0)<=wt.max(0)+1e-12)
 iq=np.r_[iq-tight,wf.ravel()]
 eqerr=float(abs(eq-fin['eq']).max());iqerr=float(abs(iq-fin['ineq']).max());yerr=float(abs(np.array(Y)-fin['local_outputs']).max());assert max(eqerr,iqerr,yerr)<1e-10;assert maximum_kinematic<1.01e-13
 passed=bool(max(abs(eq))<=1e-8 and min(iq)>=-1e-8);assert passed==data['optimizer']['numerical_constraints_passed'];assert abs(float(dt.sum()+.5)-data['optimizer']['objective_time_s'])<1e-12;assert z['guarded_timing_constraints_passed']==bool(passed and dt.sum()+.5<=13.5)
 # Reconstruct preserved numerical scales/constant omissions and nonlinear merit.
 fixture=dict(np.load(D/'window_preparation_398/fixture.npz'));scale=fixture['scale'];keep=fixture['keep'];hard=fin['hard'];slack=float(fin['slack'])
 assert np.array_equal(scale,fin['scale']) and np.array_equal(keep,fin['keep']);assert slack>=0
 removed=np.flatnonzero(~keep)
 for index in removed:
  if index==19759:
   assert sum(budgets)+25==675 and abs(iq[index])<1e-12;continue
  i=int(index)//80;col=int(index)%80;slot=(col-64)%8
  assert col>=64 and not np.count_nonzero(initial['null'][i,slot]) and not np.count_nonzero(initial['wcom'][i,slot])
  wanted=float(initial['base_weights'][i,slot]) if col<72 else 1-float(initial['base_weights'][i,slot])
  assert wanted>=0 and iq[index]==wanted
 scaled=max(0.,float(np.max(-iq[keep]/scale[keep])),abs(float(eq[0])))
 assert abs(scaled-data['algorithm']['final_merit'])<1e-8 and abs(scaled-slack)<1e-8
 run=audit_steps(fixture,data)
 epigraph=dict(slack=slack,scaled_original_violation=scaled,all_original_rows_replayed=19760,additional_support_rows=524,all_window_rows=7424,all_clock_rows=43200,removed_contact_constant_rows=392,removed_exact_integer_time_rows=1,original_numerical_feasibility=passed,positive_slack_is_not_admission=True,linear_and_acceptance_replay=run,window_replay_error=window_error)
 raw=cmd[1:]+.18*dd/dt[:,None];result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,geometry=out,command=ca,feasibility_epigraph=epigraph,primal=dict(equalities=len(eq),inequalities=len(iq),max_equation_replay_error=eqerr,max_inequality_replay_error=iqerr,max_local_replay_error=yerr,maximum_kinematic_residual=maximum_kinematic,minimum_dependent_planning_bound=float(dep.min()),numerical_constraints_passed=passed,minimum_inequality=float(iq.min())),timing=dict(event_budgets=budgets,clock_replay_error=clockerr,clock_overlap_error=overlap_error,clock_original_rate_minimum=float(clockraw[:,:32].min()),clock_original_target_minimum=float(clockraw[:,32:].min()),integer_movement_ticks=650,minimum_synthetic_quiet_ticks=25,integer_duration_s=13.5,original_minimum_inequality=float(raw_iq.min()),new_construction_minimum_inequality=float(iq.min()),continuous_guarded_s=float(dt.sum()+.5),raw_target_minimum_headroom_rad=float(min((raw-lo).min(),(hi-raw).min())),hover_gap_m=float(gap[99]),actual_float32_schedule=False,quiet_guard_exercised=False),physics_ticks=0,policy_optimizer_updates=0,physical_admission=False,repair_complete=False,source_sha256={str(source):sha(source),str(Path(__file__).resolve()):sha(__file__)})
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

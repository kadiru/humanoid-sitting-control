"""Evaluate one complete path reconstructed from357's saved dependent-coordinate projection.
Fixed351 descent/orientation phases and corner forces; no projection, force/time solve or physics rerun.
Original geometry/command/rate/headroom/hover constraints are recomputed, never assumed passed.
"""
from pathlib import Path
import json,hashlib,datetime,copy
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
from assess_low_seat_constraint_coordinates_357 import Coordinates
sides=['left','right']
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(root,R,q,feet,stance):
  T=transform(R,root);fk=m.fk(q);C=np.average([apply(T@fk[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass);foot={s:apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in sides};st=np.concatenate([foot[s] for s in stance]);sole=st[st[:,2]<st[:,2].min()+.003]
  try:H=ConvexHull(sole[:,:2]).equations;cop=float(np.min(-(H[:,:2]@C[:2]+H[:,2])));deg=False
  except QhullError:cop=-1.;deg=True
  pelvis=apply(T,m.vertices['pelvis']);dep=pelvis@n+d;patch=dep<dep.min()+.002;loc=(pelvis-center)@SR;pm=float(min(.25-abs(loc[patch,0]).max(),.45-abs(loc[patch,1]).max()));clear=[];floor=[]
  for b,v in m.vertices.items():
   w=apply(T@fk[b],v)
   if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
   if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   v=(w-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
   if len(v):clear.append((float(v[:,2].min()),b))
  pe=max(np.linalg.norm((T@fk[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in sides);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@fk[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in sides);pose=np.r_[root,Rotation.from_matrix(R).as_quat()];term=terminal_posture_check(pose);mi=min(clear) if clear else (1.,'none_over_finite_seat')
  out=dict(root=pose.tolist(),q=q.tolist(),com=C.tolist(),foot_targets={s:dict(position=feet[s][0].tolist(),rotation=feet[s][1].tolist()) for s in sides},stance=stance,clearance_m=mi[0],nearest_body=mi[1],pelvis_gap_m=float(dep.min()),patch_margin_m=pm,foot_cop_margin_m=cop,support_degenerate=deg,floor_clearance_m=min(floor),foot_position_error_m=float(pe),foot_orientation_error_rad=float(re),foot_min_z={s:float(v[:,2].min()) for s,v in foot.items()},terminal=term)
  out['other_model_passed']=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and term['passed'] and (dep.min()>=.02 or pm>=.015) and dep.min()>=-1e-8)
  out['model_pose_passed']=bool(out['other_model_passed'] and mi[0]>=.003);return out

def main():
 global m,inertia,mass,n,d,SR,center
 O=D/'projected_complete_path_v2_359';assert not O.exists();O.mkdir();M=Coordinates();m=M.m;inertia=M.i;mass=np.array(inertia['mass']);base=M.base;original=base['dense']['path'];source=D/'guarded_path_351/results.json';prior=read(source);projdir=D/'constraint_coordinates_357';pr=read(projdir/'results.json');pa=np.load(projdir/'arrays.npz');zz=pa['projected_z'];assert pr['projection']['converged']==103;assert read(D/'constraint_coordinates_audit_358/independent_audit.json')['passed']
 src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,2].astype(float);command0=tr['command'][at,2].astype(float);cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');n,d,origin,h,slope,_,_=measured(m,q0,ss[0,10,2,339:342],tr['geometry'][at,2],[tr['estimated_root_x'][at,2],0,0],cal);SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);legs=M.legs;arms=M.arms;moving=legs+arms;names=m.names;W=mass.sum()*9.81;kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);limits={j['name']:np.deg2rad(j['limits']) for j in read(M.mp)['joints'] if j['type']=='PhysicsRevoluteJoint'};joint={j['name']:j for j in m.joints};parent={j['body1'][0]:j['body0'][0] for j in m.joints if j['body0']}
 def below(body,ancestor):
  while True:
   if body==ancestor:return True
   if body not in parent:return False
   body=parent[body]
 path=copy.deepcopy(original)
 for k in range(1,104):
  old=original[k];e=prior['dense']['path'][k];_,p=M.encode(e);q,T,ff,C=M.state(k,zz[k-1],p);feet={s:(np.array(old['foot_targets'][s]['position']),np.array(old['foot_targets'][s]['rotation'])) for s in ['left','right']};v=metrics(T[:3,3],T[:3,:3],q,feet,old['stance']);v.update(phase=old['phase'],u=old['u'],swing_attitude_bound_rad=old['swing_attitude_bound_rad'],approach_fraction=float(p[0]),orientation_fraction=float(p[1]),com_target_xy=old['com_target_xy'],com_target_error_m=float(np.linalg.norm(C[:2]-old['com_target_xy'])),multiknot_constraint_residual=float(max(abs(M.residual(k,zz[k-1],p)))));path[k]=v
 nodes=[];tau0=np.array(base['dense']['command_nodes'][0]['planned_static_torque_Nm'])
 for k,e in enumerate(path):
  v=prior['dense']['command_nodes'][k];P=np.array(v['corner_world_points']);w=np.array(v['corner_fractions']);side=v['corner_sides'];q=np.array(e['q']);T=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));fk=m.fk(q);tau=[]
  for j in moving:
   name=names[j];J=T@fk[joint[name]['body0'][0]]@m.frames[name][0];axis=J[:3,'XYZ'.index(joint[name]['axis'])];originj=J[:3,3];child=joint[name]['body1'][0];mom=np.zeros(3)
   for b,mm,cc in zip(inertia['body_names'],mass,inertia['local_com']):
    if below(b,child):mom+=np.cross(apply(T@fk[b],cc)-originj,[0,0,-mm*9.81])
   for point,ww,sside in zip(P,w,side):
    if below(sside+'_ankle_roll_link',child):mom+=np.cross(point-originj,[0,0,W*ww])
   tau.append(float(-axis@mom))
  tau=np.array(tau);cmd=command0.copy();cmd[moving]+=q[moving]-q0[moving]+(tau-tau0)/kp
  if k==0:cmd=command0.copy()
  C=np.array(e['com']);normal=v['assumed_seat_normal_fraction'];pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);sv=normal/n[2];res=float(max(abs(w.sum()+sv-1),abs(w@P[:,:2]+sv*seat[:2]-C[:2]).max()));margin=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));util=float(max(abs(tau)/effort));nodes.append(dict(knot=k,command=cmd.tolist(),planned_static_torque_Nm=tau.tolist(),corner_fractions=w.tolist(),corner_world_points=P.tolist(),corner_sides=side,assumed_seat_normal_fraction=normal,planned_right_load_fraction=float(sum(ww for ww,s in zip(w,side) if s=='right')),min_command_margin_rad=margin,maximum_effort_utilization=util,max_equilibrium_residual=res))
 cmd=np.array([v['command'] for v in nodes])[:,moving];dt=np.array(prior['dense']['segment_durations_s']);initial=np.load(D/'guarded_path_351/initial.npz');rates=initial['rates'];lo=initial['command_lo'];hi=initial['command_hi'];dd=np.diff(cmd,axis=0);gaps=np.array([e['pelvis_gap_m'] for e in path]);time=np.maximum(np.max(abs(dd)/rates,axis=1),.02)
 for k in range(100,105):time[k-1]=max(time[k-1],(gaps[k-1]-gaps[k])/.006)
 raw=cmd[1:]+.18*dd/dt[:,None];projection_ok=all(r['converged'] for r in pr['projection']['rows']);geometry=all(e['model_pose_passed'] for e in path);forceok=all(min(v['corner_fractions'])>=0 and v['max_equilibrium_residual']<1e-8 for v in nodes);margin=min(v['min_command_margin_rad'] for v in nodes);util=max(v['maximum_effort_utilization'] for v in nodes);phases=np.array([[e['approach_fraction'],e.get('orientation_fraction',e['approach_fraction'])] for e in path]);rate_violation=float(np.maximum(abs(dd)-rates*dt[:,None],0).max());headroom=float(min((raw-lo).min(),(hi-raw).min()));normal_violation=float(np.maximum(gaps[99:104]-gaps[100:105]-.006*dt[99:104],0).max());timing=dict(saved351_durations_s=float(dt.sum()+.5),fixed_projected_command_reserved_rate_normal_plus_quiet_s=float(time.sum()+.5),metric_is_not_filter_schedule=True,maximum_saved_duration_rate_violation_rad=rate_violation,minimum_linear_desired_headroom_rad=headroom,hover_gap_m=float(gaps[99]),hover_equality_error_m=float(gaps[99]-.012),maximum_normal_descent_violation_m=normal_violation,minimum_phase_increment=float(np.diff(phases,axis=0).min()),actual_filter_executed=False,quiet_guard_exercised=False)
 dense=dict(parameters=base['dense']['parameters'],knots=115,path=path,command_nodes=nodes,failed_knots=[k for k,e in enumerate(path) if not e['model_pose_passed']],min_clearance_m=min(e['clearance_m'] for e in path),min_cop_margin_m=min(e['foot_cop_margin_m'] for e in path),max_foot_position_error_m=max(e['foot_position_error_m'] for e in path),max_foot_orientation_error_rad=max(e['foot_orientation_error_rad'] for e in path),min_actual_swing_sole_height_m=min(e['foot_min_z']['right'] for e in path if e['phase']=='swing' and .2<e['u']<.8),min_command_margin_rad=margin,maximum_effort_utilization=util,model_geometry_passed=bool(geometry),model_command_passed=bool(forceok and margin>=.03 and util<=1),physical_admission=False)
 horizontal=[min(a['foot_min_z']['right'],b['foot_min_z']['right']) for a,b in zip(path,path[1:]) if np.linalg.norm(np.array(a['foot_targets']['right']['position'])[:2]-np.array(b['foot_targets']['right']['position'])[:2])>1e-8];dense['minimum_horizontal_segment_sole_height_m']=float(min(horizontal));dense['model_geometry_passed']=bool(geometry and dense['min_actual_swing_sole_height_m']>=.005 and min(horizontal)>=.005)
 sources=[source,projdir/'results.json',projdir/'arrays.npz',D/'constraint_coordinates_audit_358/independent_audit.json',D/'guarded_path_351/initial.npz',M.basefile,M.mp,M.ip,src/'trace.npz',src/'sensor_inputs.npz',Path(__file__),Path(__file__).with_name('assess_low_seat_constraint_coordinates_357.py')];result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,revision=dict(changed_pose_knots=list(range(1,104)),case=2,placements=1,optimization_runs=0,projection_repeated=False,forces_exact351=True,phases_exact351=True),dense=dense,timing=timing,endpoint=base['endpoint'],stance_target_constraint=base['stance_target_constraint'],unchanged_pose_knots=[0]+list(range(104,115)),source_input_sha256={str(p.resolve()):sha(p) for p in sources},physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in dense.items() if k not in ['path','command_nodes']},indent=2));print(json.dumps(timing,indent=2),flush=True)
if __name__=='__main__':main()

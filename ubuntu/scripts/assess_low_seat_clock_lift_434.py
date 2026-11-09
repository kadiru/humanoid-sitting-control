"""First between-knot compatibility assessment of the saved430 path.

This is a bounded OFFLINE assessment, not another optimizer, a correction of430,
or execution of its command schedule. One canonical cardinal interpolant is
defined: linear joint angles/root translation/corner loads/corner world points,
shortest-arc root and prescribed-foot rotations. All saved nodes are retained.
Evaluate all676 ideal20ms locations, all675 half-tick locations, and all116
vertices including the duplicate hover endpoint. The1467 labeled records are
not collapsed or time-rounded. Check actual FK/COM/static-work consistency and
original sampled geometry. A finite sample check is not a continuous certificate.

The purpose is to test the previously UNTESTED implication that interpolating
saved commands also interpolates a valid nominal geometry/equilibrium state.
The inherited430 extra-buffer failure remains a failed gate. No targets are
constructed or passed through the float32 forward filter, even if other checks
pass. Saved ideal commands are quantized for rounding/error accounting ONLY;
quantized desired values are not actual filter outputs or sensor measurements.
There is no projection, optimizer, runtime, physics, or policy update.
"""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation, Slerp
from humanoid_sitting.shared_frame import BodyModel, apply, transform, level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'clock_lift_434'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def setup():
 data=read(D/'lattice_path_430/results.json'); path=data['dense']['path']; nodes=data['dense']['command_nodes']
 m=BodyModel(D/'cooked_model_50/model.json'); ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
 cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json'); src=D/'normal_landing3s_06/candidate'; tr=np.load(src/'trace.npz'); ss=np.load(src/'sensor_inputs.npz')['sensors']; c=2; at=int(np.argmin(abs(tr['t']-1)))
 q0=tr['q'][at,c].astype(float); command0=tr['command'][at,c].astype(float); g=ss[0,10,c,339:342]
 n,plane_d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal)
 SR=Rotation.from_euler('y',slope,degrees=True).as_matrix(); center=np.array([-.4,0,h]); mass=np.asarray(ine['mass']); W=mass.sum()*9.81
 sides=['left','right']; legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]; arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']]; moving=legs+arms; fixed=[j for j in range(37) if j not in moving]
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
 kp=np.array([200,150,150,200,160,160]*2+[40]*4); effort=np.array([300,300,300,300,20,20]*2+[300]*4)
 byname={j['name']:j for j in m.joints}; children={}
 for j in m.joints:
  if j['body0']:children.setdefault(j['body0'][0],[]).append(j['body1'][0])
 def descend(b):
  out={b}
  for c in children.get(b,[]):out|=descend(c)
  return out
 sub={name:descend(byname[name]['body1'][0]) for name in m.names}
 dt=np.asarray(data['dense']['segment_durations_s']); times=np.r_[0.,np.cumsum(np.r_[dt[:99],.5,dt[99:]])]; ids=np.r_[np.arange(100),99,np.arange(100,115)]
 Q=np.array([p['q'] for p in path]); roots=np.array([p['root'] for p in path]); C=np.array([v['command'] for v in nodes]); weights=np.array([v['corner_fractions'] for v in nodes]); P=np.array([v['corner_world_points'] for v in nodes]); normal=np.array([v['assumed_seat_normal_fraction'] for v in nodes]); com=np.array([p['com'] for p in path]); corner_sides=nodes[0]['corner_sides']
 assert all(v['corner_sides']==corner_sides for v in nodes); assert weights.shape==(115,8)
 # Actual saved foot frames are separate from their nominal prescribed targets.
 actual_feet=[]
 for q,r in zip(Q,roots):
  T=transform(Rotation.from_quat(r[3:]).as_matrix(),r[:3]); fk=m.fk(q); actual_feet.append([T@fk[s+'_ankle_roll_link'] for s in sides])
 actual_feet=np.asarray(actual_feet);tau0=np.array(nodes[0]['planned_static_torque_Nm'])
 return locals()

def rotmix(A,B,a):
 if a==0:return A.copy()
 if a==1:return B.copy()
 return A@Rotation.from_rotvec(a*Rotation.from_matrix(A.T@B).as_rotvec()).as_matrix()

def lift(s,t,vertex=None):
 times=s['times']; ids=s['ids']; path=s['path']
 if vertex is not None:i=int(vertex);a=0.;k=l=int(ids[i])
 else:
  i=min(max(int(np.searchsorted(times,t,side='right')-1),0),114);a=float((t-times[i])/(times[i+1]-times[i]));k=int(ids[i]);l=int(ids[i+1]);assert -1e-12<=a<=1+1e-12
 if k==l:a=0.
 mix=lambda v:v[k].copy() if a==0 else v[l].copy() if a==1 else v[k]+a*(v[l]-v[k])
 q=mix(s['Q']); q[s['fixed']]=s['q0'][s['fixed']]; root=mix(s['roots'][:,:3]); R=rotmix(Rotation.from_quat(s['roots'][k,3:]).as_matrix(),Rotation.from_quat(s['roots'][l,3:]).as_matrix(),a)
 feet=[];actual=[]
 for j,side in enumerate(s['sides']):
  pa=path[k]['foot_targets'][side];pb=path[l]['foot_targets'][side]
  feet.append((np.array(pa['position'])+a*(np.array(pb['position'])-pa['position']),rotmix(np.array(pa['rotation']),np.array(pb['rotation']),a)))
  A=s['actual_feet'][k,j];B=s['actual_feet'][l,j];actual.append(transform(rotmix(A[:3,:3],B[:3,:3],a),A[:3,3]+a*(B[:3,3]-A[:3,3])))
 stance=['left'] if 16<=k<64 and l>16 and a>0 else path[k]['stance'] if a==0 else path[l]['stance'] if a==1 else path[l]['stance']
 return dict(time_s=float(t),vertex=vertex,segment=i,knot0=k,knot1=l,fraction=a,q=q,root=root,R=R,feet=feet,actual_foot_reference=np.array(actual),stance=stance,weights=mix(s['weights']),points=mix(s['P']),normal=float(mix(s['normal'])),linear_command=mix(s['C']),com_reference=mix(s['com']))

def assess(s,r):
 m=s['m'];T=transform(r['R'],r['root']);fk=m.fk(r['q']);world={b:T@v for b,v in fk.items()};pts=np.array([apply(world[b],cc) for b,cc in zip(s['ine']['body_names'],s['ine']['local_com'])]);com=np.average(pts,axis=0,weights=s['mass']);foot={side:apply(world[side+'_ankle_roll_link'],m.vertices[side+'_ankle_roll_link']) for side in s['sides']};v=np.concatenate([foot[x] for x in r['stance']]);sole=v[v[:,2]<v[:,2].min()+.003]
 try:H=ConvexHull(sole[:,:2]).equations;cop=float(np.min(-(H[:,:2]@com[:2]+H[:,2])))
 except QhullError:cop=-1.
 pv=apply(T,m.vertices['pelvis']);pd=pv@s['n']+s['plane_d'];patch=pd<pd.min()+.002;loc=(pv-s['center'])@s['SR'];pm=float(min(.25-abs(loc[patch,0]).max(),.45-abs(loc[patch,1]).max()));clear=[];floor=[]
 for b,vertices in m.vertices.items():
  w=apply(world[b],vertices)
  if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
  if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
  v=(w-s['center'])@s['SR'];v[:,0]-=.4;v=clipped_vertices(v)
  if len(v):clear.append((float(v[:,2].min()),b))
 pe=max(np.linalg.norm(world[side+'_ankle_roll_link'][:3,3]-r['feet'][j][0]) for j,side in enumerate(s['sides']));re=max(np.linalg.norm(Rotation.from_matrix(r['feet'][j][1].T@world[side+'_ankle_roll_link'][:3,:3]).as_rotvec()) for j,side in enumerate(s['sides']))
 exact_foot_position=max(np.linalg.norm(world[side+'_ankle_roll_link'][:3,3]-r['actual_foot_reference'][j,:3,3]) for j,side in enumerate(s['sides']));exact_foot_orientation=max(np.linalg.norm(Rotation.from_matrix(r['actual_foot_reference'][j,:3,:3].T@world[side+'_ankle_roll_link'][:3,:3]).as_rotvec()) for j,side in enumerate(s['sides']))
 term=terminal_posture_check(np.r_[r['root'],Rotation.from_matrix(r['R']).as_quat()]);clearance,nearest=min(clear) if clear else (1.,'none_over_finite_seat')
 seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=s['n']*(seat@s['n']+s['plane_d']);sv=r['normal']/s['n'][2]
 equilibrium=np.r_[r['weights'].sum()+sv-1,(r['weights'][:,None]*r['points'][:,:2]).sum(0)+sv*seat[:2]-com[:2]]
 tau=[]
 for j in s['moving']:
  name=m.names[j];jj=s['byname'][name];J=world[jj['body0'][0]]@m.frames[name][0];axis=J[:3,'XYZ'.index(jj['axis'])];o=J[:3,3];moment=np.zeros(3)
  for b,mm,cc in zip(s['ine']['body_names'],s['mass'],pts):
   if b in s['sub'][name]:moment+=np.cross(cc-o,[0,0,-mm*9.81])
  for side,ww,p in zip(s['corner_sides'],r['weights'],r['points']):
   if side+'_ankle_roll_link' in s['sub'][name]:moment+=np.cross(p-o,[0,0,s['W']*ww])
  tau.append(-axis@moment)
 tau=np.asarray(tau);command=s['command0'].copy();command[s['moving']]+=r['q'][s['moving']]-s['q0'][s['moving']]+(tau-s['tau0'])/s['kp']
 if r['knot0']==0 and r['fraction']==0:command=s['command0'].copy()
 cm=min(min(command[j]-s['limits'][m.names[j]][0],s['limits'][m.names[j]][1]-command[j]) for j in s['moving']);util=max(abs(tau)/s['effort']);gap=float(pd.min())
 geometry=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and term['passed'] and (gap>=.02 or pm>=.015) and gap>=-1e-8 and clearance>=.003)
 rec={k:r[k] for k in ['time_s','vertex','segment','knot0','knot1','fraction','stance']}
 rec.update(clearance_m=clearance,nearest_body=nearest,foot_cop_margin_m=cop,foot_position_error_m=float(pe),foot_orientation_error_rad=float(re),pelvis_gap_m=gap,patch_margin_m=pm,floor_clearance_m=min(floor),swing_sole_min_m=float(foot['right'][:,2].min()),model_tilt_deg=term['model_tilt_deg'],sampled_geometry_passed=geometry,interpolated_foot_position_defect_m=float(exact_foot_position),interpolated_foot_orientation_defect_rad=float(exact_foot_orientation),com_interpolation_defect_m=float(max(abs(com[:2]-r['com_reference'][:2]))),equilibrium_residual=float(max(abs(equilibrium))),minimum_command_margin_rad=float(cm),effort_utilization=float(util),command_interpolation_defect_rad=float(max(abs(command-r['linear_command']))),positive_gap_with_nonzero_seat_load=bool(gap>1e-8 and r['normal']!=0),minimum_corner_fraction=float(r['weights'].min()),inactive_foot_exact_zero=bool(r['stance']!=['left'] or np.all(r['weights'][4:]==0)))
 return rec,command,tau,com,equilibrium

def main():
 launch=read(D/'clock_lift_launch_434.json')
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);s=setup(); queries=[(i*.02,None,'clock',i) for i in range(676)]+[((i+.5)*.02,None,'half_tick',i) for i in range(675)]+[(float(t),i,'vertex',i) for i,t in enumerate(s['times'])]
 records=[]; rr=[];cc=[];tt=[];cm=[];ee=[]
 for t,v,kind,index in queries:
  r=lift(s,t,v); rec,c,tau,com,e=assess(s,r);rec.update(sample_kind=kind,sample_index=index);records.append(rec);rr.append(r);cc.append(c);tt.append(tau);cm.append(com);ee.append(e)
  if len(records)%200==0:print(json.dumps(dict(completed_records=len(records),total=len(queries))),flush=True)
 cc=np.asarray(cc);tt=np.asarray(tt);initial=dict(np.load(D/'lattice_path_430/initial.npz'));clock=cc[:676,s['moving']];delta=np.diff(clock,axis=0);targets=10*clock[1:]-9*clock[:-1];rate=np.c_[.02*initial['rates']-delta,.02*initial['rates']+delta];head=np.c_[initial['command_hi']-targets,targets-initial['command_lo']]
 original=dict(np.load(D/'lattice_path_430/event_clock.npz'));ideal=original['clock'];rounded=ideal.astype(np.float32);rdelta=np.diff(rounded.astype(float),axis=0);rounded_rate=np.c_[.02*initial['rates']-rdelta,.02*initial['rates']+rdelta]
 interior=[i for i,r in enumerate(records) if r['sample_kind']!='vertex' and r['fraction'] not in [0.,1.] and r['knot0']!=r['knot1']]
 vertices=records[1351:];replay=max(max(abs(cc[1351+j]-s['C'][k])) for j,k in enumerate(s['ids']));assert replay<2e-12,replay
 for j,k in enumerate(s['ids']):
  old=s['path'][k];new=vertices[j]
  for key in ['clearance_m','foot_cop_margin_m','foot_position_error_m','foot_orientation_error_rad','pelvis_gap_m','patch_margin_m','floor_clearance_m']:assert abs(old[key]-new[key])<1e-12,(k,key,old[key],new[key])
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,records=len(records),clock_records=676,half_tick_records=675,vertex_records=116,new_interior_records=len(interior),source430_prior_gates_passed=True,source430_extra_buffer_passed=False,source430_extra_buffer_minimum=s['data']['algorithm']['minimum_buffered_hard_row'],sampled_geometry_failures=[i for i,r in enumerate(records) if not r['sampled_geometry_passed']],interior_geometry_failures=[i for i in interior if not records[i]['sampled_geometry_passed']],minimum_clearance_m=min(r['clearance_m'] for r in records),minimum_cop_margin_m=min(r['foot_cop_margin_m'] for r in records),maximum_fixed_foot_interpolation_defect_m=max(r['interpolated_foot_position_defect_m'] for r in records),maximum_com_interpolation_defect_m=max(r['com_interpolation_defect_m'] for r in records),maximum_equilibrium_residual=max(r['equilibrium_residual'] for r in records),maximum_command_interpolation_defect_rad=max(r['command_interpolation_defect_rad'] for r in records),minimum_command_margin_rad=min(r['minimum_command_margin_rad'] for r in records),maximum_effort_utilization=max(r['effort_utilization'] for r in records),recomputed_static_clock_rate_minimum_rad=float(rate.min()),recomputed_static_clock_target_minimum_rad=float(head.min()),saved_ideal_rounding_maximum_rad=float(abs(rounded.astype(float)-ideal).max()),saved_ideal_rounded_rate_minimum_rad=float(rounded_rate.min()),vertex_command_replay_error=replay,all_fixed_q_exact=bool(all(np.array_equal(r['q'][s['fixed']],s['q0'][s['fixed']]) for r in rr)),all_fixed_commands_exact=bool(np.array_equal(cc[:,s['fixed']],np.tile(s['command0'][s['fixed']],(len(cc),1)))),negative_corner_records=[i for i,r in enumerate(records) if r['minimum_corner_fraction']<0],inactive_load_failures=[i for i,r in enumerate(records) if not r['inactive_foot_exact_zero']],continuous_clearance_certified=False,nonlinear_lift_compatible=bool(all(r['sampled_geometry_passed'] and r['equilibrium_residual']<1e-8 and r['interpolated_foot_position_defect_m']<1e-10 and r['com_interpolation_defect_m']<1e-10 for r in records)),actual_float32_filter_ticks=0,optimizer_calls=0,root_projection_calls=0,physics_ticks=0,policy_updates=0,physical_admission=False,source_input_sha256=launch['source_input_sha256'])
 (O/'results.json').write_text(json.dumps(result,indent=2));(O/'records.json').write_text(json.dumps(records,indent=2));np.savez_compressed(O/'lift.npz',q=np.array([r['q'] for r in rr]),root=np.array([np.r_[r['root'],Rotation.from_matrix(r['R']).as_quat()] for r in rr]),weights=np.array([r['weights'] for r in rr]),points=np.array([r['points'] for r in rr]),seat_normal=np.array([r['normal'] for r in rr]),command=cc,torque=tt,com=np.array(cm),equilibrium=np.array(ee),linear_command=np.array([r['linear_command'] for r in rr]),actual_foot_reference=np.array([r['actual_foot_reference'] for r in rr]),com_reference=np.array([r['com_reference'] for r in rr]),clock_rate=rate,clock_target=head,rounded_ideal=rounded,rounded_ideal_rate=rounded_rate,moving=s['moving'],times=s['times'],vertex_ids=s['ids'],plane_normal=s['n'],plane_d=s['plane_d'])
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

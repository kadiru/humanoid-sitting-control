"""Independent geometry/finite-difference force replay of contact84/stance86."""
from pathlib import Path
import json,hashlib,datetime,argparse
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source',choices=['contact_transition_screen_84','contact_stance_screen_86','contact_stance_screen_89','contact_headroom_endpoint_114'],required=True);parser.add_argument('--output',required=True);args=parser.parse_args();O=D/args.output;O.mkdir(exist_ok=False);source=D/args.source/'results.json';data=read(source)
 for p,h in data['source_sha256'].items():assert sha(p)==h,p
 mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';m=BodyModel(mp);ine=read(ip);mass=np.array(ine['mass']);W=mass.sum()*9.81;names=m.names;tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');ss=np.load(D/'normal_landing3s_06/candidate/sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,2].astype(float);cmd0=tr['command'][at,2].astype(float);g=ss[0,10,2,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,2,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,2],[tr['estimated_root_x'][at,2],0,0],read(D.parent/'contact_effort_repair_01/plane_calibration.json'));R0=level_rotation(g);fk0=m.fk(q0);feet0={s:(R0@fk0[s+'_ankle_roll_link'][:3,3]+origin,R0@fk0[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);legs=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];moving=legs+arms;fixed=[j for j in range(len(names)) if j not in moving];limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);initial=read(D/'actuator_contact_path_refinement_77/results.json')['dense'];tau0=np.array(initial['command_nodes'][0]['planned_static_torque_Nm']);rows=[('best',data['best'])]+[(f'search{i}',v['endpoint']) for i,v in enumerate(data['searches'])]+[(f'path{i}',e) for i,e in enumerate(data.get('path',[]))]+[(f'history{i}',e) for i,e in enumerate(data.get('history',[]))];errors=[];virtual=[];eqs=[];checked=[]
 for label,e in rows:
  q=np.array(e['q']);pose=np.array(e['root']);T=transform(Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]);fk=m.fk(q);assert np.isfinite(q).all() and np.array_equal(q[fixed],q0[fixed]);assert max(abs(q[arms]-(q0[arms]-.35)))<1e-12
  for j in legs:assert limits[names[j]][0]+.031-1e-10<=q[j]<=limits[names[j]][1]-.031+1e-10
  feet={s:(np.array(v['position']),np.array(v['rotation'])) for s,v in e['foot_targets'].items()};length=e['parameters'][5] if len(e['parameters'])==7 else .1;yaw=e['parameters'][6] if len(e['parameters'])==7 else 0.;assert np.array_equal(feet['left'][0],feet0['left'][0]) and np.array_equal(feet['left'][1],feet0['left'][1]);assert max(abs(feet['right'][0]-(feet0['right'][0]+[-length,0,0])))<1e-13;assert max(abs(feet['right'][1]-Rotation.from_euler('z',yaw).as_matrix()@feet0['right'][1]).ravel())<1e-13
  C=np.average([apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass);foot={s:apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']};st=np.concatenate(list(foot.values()));sole=st[st[:,2]<st[:,2].min()+.003]
  try:H=ConvexHull(sole[:,:2]).equations;cop=float(np.min(-(H[:,:2]@C[:2]+H[:,2])));deg=False
  except QhullError:cop=-1.;deg=True
  v=apply(T,m.vertices['pelvis']);dep=v@n+d;patch=dep<dep.min()+.002;loc=(v-center)@SR;pm=float(min(.25-abs(loc[patch,0]).max(),.45-abs(loc[patch,1]).max()));contact=v[dep<=dep.min()+1e-7].mean(0);contact-=n*(contact@n+d);clear=[];floor=[]
  for b,v in m.vertices.items():
   w=apply(T@fk[b],v)
   if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
   if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   v=(w-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
   if len(v):clear.append((float(v[:,2].min()),b))
  mi=min(clear) if clear else (1.,'none_over_finite_seat');pe=max(np.linalg.norm((T@fk[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in feet);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@fk[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in feet);term=terminal_posture_check(pose);values=dict(clearance_m=mi[0],pelvis_gap_m=float(dep.min()),patch_margin_m=pm,foot_cop_margin_m=cop,floor_clearance_m=min(floor),foot_position_error_m=pe,foot_orientation_error_rad=re);err=max(abs(e[k]-v) for k,v in values.items());assert err<1e-9 and e['nearest_body']==mi[1] and e['terminal']==term;errors.append(err);other=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and term['passed'] and (dep.min()>=.02 or pm>=.015) and dep.min()>=-1e-8);assert other==e['other_model_passed'] and bool(other and mi[0]>=.003)==e['model_pose_passed']
  loads=e.get('load_checks',[e.get('load')]);assert all(v is not None for v in loads)
  for load in loads:
   if 'command' not in load:assert not load['passed'];continue
   P=np.array(load['corner_world_points']);w=np.array(load['corner_fractions']);sides=load['corner_sides'];sf=np.array(load['seat_force_normalized']);normal=load['normal_load_fraction'];assert abs(sf@n-normal)<1e-12 and max(abs(np.array(load['seat_contact_point'])-contact))<1e-12
   if normal>0:assert abs(dep.min())<1e-8
   eq=max(abs(w.sum()+sf[2]-1),max(abs((w[:,None]*P[:,:2]).sum(0)+sf[2]*contact[:2]-C[:2])));assert eq<1e-8 and min(w)>=-1e-10;eqs.append(eq);local=[np.linalg.inv(T@fk[s+'_ankle_roll_link'])@np.r_[p,1.] for p,s in zip(P,sides)];tau=[]
   for j in moving:
    vv=[]
    for sign in [-1,1]:
     qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);V=sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],mass,ine['local_com']));work=sum(W*ww*(T@ff[s+'_ankle_roll_link']@ll)[2] for ww,s,ll in zip(w,sides,local));vv.append(V-work)
    tau.append((vv[1]-vv[0])/2e-5)
   tau=np.array(tau);terr=max(abs(tau-load['static_torque_Nm']));assert terr<1e-6;virtual.append(terr);expected=cmd0.copy();expected[moving]+=q[moving]-q0[moving]+(tau-tau0)/kp;cmd=np.array(load['command']);assert max(abs(cmd-expected))<2e-8 and np.array_equal(cmd[fixed],cmd0[fixed]);cm=min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving);eu=max(abs(tau)/effort);assert abs(cm-load['min_command_margin_rad'])<1e-12 and abs(eu-load['maximum_effort_utilization'])<1e-7;assert load['passed']==bool(eq<1e-8 and min(w)>=-1e-10 and cm>=.03 and max(abs(np.array(load['static_torque_Nm']))/effort)<=1)
  checked.append(dict(record=label,geometry_passed=e['model_pose_passed'],housing_clearance_mm=mi[0]*1000,all_assumed_load_commands_passed=all(v['passed'] for v in loads),gap_m=float(dep.min())))

 if args.source=='contact_headroom_endpoint_114':
  eligible=[e for e in data['history'] if e['model_pose_passed'] and all(v['passed'] for v in e['load_checks'])];best=max(eligible,key=lambda e:e['min_command_margin_rad']) if eligible else max(data['history'],key=lambda e:e['min_command_margin_rad']);assert best==data['best'];assert len(data['history'])<=60;assert data['summary']['headroom_research_target_passed']==bool(best['min_command_margin_rad']>=.06)
 summary=data['summary'];assert summary['endpoint_model_passed']==data['best']['all_endpoint_model_checks_passed']
 if 'path' in data:
  assert len(data['path'])==31;assert summary['complete_sampled_transition_passed']==all(v['geometry_passed'] and v['all_assumed_load_commands_passed'] for v in checked if v['record'].startswith('path'));commands=np.array([e['load']['command'] for e in data['path']]);time=np.max(abs(np.diff(commands,axis=0)),axis=1).sum()/.15;assert abs(time-summary['command_time_at015_s'])<1e-10
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent cooked geometry and full-foot/terminal/joint replay plus finite-difference16joint virtual work, balanced nonnegative foot forces and explicitly assumed vertical seat forces. Positive-gap nodes carry zero seat force. No sensor load or physical contact is certified. Failed geometric/contact/filter gates remain failed.',records=len(rows),max_geometry_replay_error=float(max(errors)),max_virtual_work_error_Nm=float(max(virtual)),max_equilibrium_residual=float(max(eqs)),checked=checked,original_model_summary=summary,source_sha256={str(source):sha(source),str(Path(__file__).resolve()):sha(__file__)},physics_ticks=0,optimizer_updates=0,physical_admission=False);(O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['checked','source_sha256']}))
if __name__=='__main__':main()

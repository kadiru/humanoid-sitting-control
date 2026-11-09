"""Independent static endpoint replay; never physical admission."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'spatial_contact_audit_61'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 O.mkdir(exist_ok=False);source=D/'spatial_contact_screen_v2_58/results.json';a=read(source)
 for p,h in a['source_input_sha256'].items():assert sha(p)==h,p
 m=BodyModel(D/'cooked_model_50/model.json');meta=read(D/'cooked_model_50/model.json');inertia=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');p=D/'normal_landing3s_06/candidate';tr=np.load(p/'trace.npz');ss=np.load(p/'sensor_inputs.npz')['sensors'];mass=np.asarray(inertia['mass']);at=int(np.argmin(abs(tr['t']-1)))
 sides=['left','right'];legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];fixed=[i for i in range(len(m.names)) if i not in legs+arms];limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'};rows=[]
 for rr in a['rows']:
  c=rr['case'];q0=tr['q'][at,c].astype(float);g=ss[0,10,c,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,c,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,c],[tr['estimated_root_x'][at,c],0,0],cal);R0=level_rotation(g);f0=m.fk(q0);feet={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in sides};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h])
  targets=[('best',rr['best'])]+[(f'start_{i}',z['endpoint']) for i,z in enumerate(rr['searches'])]
  for name,e in targets:
   if e is None:continue
   q=np.asarray(e['q']);pose=np.asarray(e['root']);R=Rotation.from_quat(pose[3:]).as_matrix();T=transform(R,pose[:3]);fk=m.fk(q);px,py,roll,pitch,yaw=e['parameters'];assert np.max(abs(R-Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix()))<1e-12
   assert np.array_equal(q[fixed],q0[fixed]);assert np.max(abs(q[arms]-(q0[arms]-.35)))<1e-14
   assert min(min(q[j]-limits[m.names[j]][0],limits[m.names[j]][1]-q[j]) for j in legs)>=.031-1e-12
   C=np.average([apply(T@fk[b],v) for b,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass);fv=np.concatenate([apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in sides]);sole=fv[fv[:,2]<fv[:,2].min()+.003]
   try:H=ConvexHull(sole[:,:2]).equations;cop=float(np.min(-(H[:,:2]@C[:2]+H[:,2])));degen=False
   except QhullError:cop=-1.;degen=True
   pelvis=apply(T,m.vertices['pelvis']);gap=pelvis@n+d;patch=gap<gap.min()+.002;local=(pelvis-center)@SR;pm=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()));clear=[];floor=[]
   for b,v in m.vertices.items():
    w=apply(T@fk[b],v)
    if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
    if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
    v=(w-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
    if len(v):clear.append((float(v[:,2].min()),b))
   pe=max(np.linalg.norm((T@fk[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in sides);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@fk[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in sides);term=terminal_posture_check(pose)
   expected=dict(clearance_m=min(clear)[0],foot_position_error_m=pe,foot_orientation_error_rad=re,static_foot_cop_margin_m=cop,pelvis_patch_margin_m=pm,pelvis_gap_m=float(gap.min()),floor_clearance_m=min(floor));err=max(abs(e[k]-v) for k,v in expected.items());assert err<1e-9
   assert e['nearest_body']==min(clear)[1] and e['support_polygon_degenerate']==degen and e['terminal']==term
   other=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and pm>=.015 and min(floor)>=.005 and term['passed']);passed=bool(other and min(clear)[0]>=.003);assert e['other_model_constraints_passed']==other and e['endpoint_screen_passed']==passed
   rows.append(dict(case=c,endpoint=name,max_replay_error=err,model_endpoint_passed=passed,clearance_mm=1000*min(clear)[0],model_tilt_deg=term['model_tilt_deg']))
  valid=[e for e in rr['history'] if e['other_model_constraints_passed']]
  assert (rr['best'] is None)==(not valid)
  if valid:assert rr['best']['clearance_m']==max(e['clearance_m'] for e in valid)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent corrected-model endpoint arithmetic replay, actual1s sensor mapping, nominal non-leg/finger pose preservation, source hashes and best-of-recorded-search rank. No whole path, actuator command/fixed-finger command, dynamic force/contact, or physical admission is claimed.',rows=rows,source_sha256={str(source):sha(source),str(Path(__file__).resolve()):sha(__file__)},physics_ticks=0,optimizer_updates=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(passed=True,rows=len(rows),best=[r for r in rows if r['endpoint']=='best'])))
if __name__=='__main__':main()

"""Bounded full-pose fixed-foot contact planning, offline only.

Uses existing legal pelvis patch, cooked collision model, original support
criterion and six-joint leg IK. Additional spatial pelvis orientation is a
pre-training planning scope change; no runtime, physics or criterion changes.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'spatial_contact_screen_v2_58'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';src=D/'normal_landing3s_06/candidate'
 m=BodyModel(mp);md=read(mp);inertia=read(ip);cal=read(cp);tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];limits={j['name']:np.deg2rad(j['limits']) for j in md['joints'] if j['type']=='PhysicsRevoluteJoint'}
 ids={s:[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in ['left','right']};legs=ids['left']+ids['right'];arms=[m.names.index(s+'_'+j+'_joint') for s in ids for j in ['shoulder_pitch','elbow_pitch']];mass=np.asarray(inertia['mass']);rows=[]
 for case in [2,0,6,10]:
  at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,case].astype(float);command=tr['command'][at,case].astype(float);g=ss[0,10,case,339:342];assert np.array_equal(q0[:23].astype(np.float32),ss[0,10,case,270:293]);n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],cal);R0=level_rotation(g);f0=m.fk(q0);feet={s:(R0@f0[s+'_ankle_roll_link'][:3,3]+origin,R0@f0[s+'_ankle_roll_link'][:3,:3]) for s in ids};seed=q0.copy();seed[arms]-=.35;SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);cache={};history=[]
  def evaluate(x):
   key=tuple(float(v) for v in x)
   if key in cache:return cache[key]
   px,py,roll,pitch,yaw=key;R=Rotation.from_euler('xyz',[roll,pitch,yaw]).as_matrix();v=m.vertices['pelvis']@R.T;z=(.001-min(v@n)-n[0]*px-n[1]*py-d)/n[2];root=np.array([px,py,z]);T=transform(R,root);q=seed.copy()
   for side in ids:
    ii=ids[side];pos,rot=feet[side];lo=np.array([limits[m.names[j]][0]+.031 for j in ii]);hi=np.array([limits[m.names[j]][1]-.031 for j in ii])
    def residual(values):
     qq=q.copy();qq[ii]=values;F=T@m.fk(qq)[side+'_ankle_roll_link'];return np.r_[F[:3,3]-pos,.2*Rotation.from_matrix(rot.T@F[:3,:3]).as_rotvec()]
    fit=least_squares(residual,np.clip(q[ii],lo,hi),bounds=(lo,hi),max_nfev=45,xtol=1e-9,ftol=1e-9,gtol=1e-9);q[ii]=fit.x
   f=m.fk(q);C=np.average([apply(T@f[b],c) for b,c in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass);foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ids]);sole=foot[foot[:,2]<foot[:,2].min()+.003];support_degenerate=False
   try:
    H=ConvexHull(sole[:,:2]).equations;copmargin=float(np.min(-(H[:,:2]@C[:2]+H[:,2])))
   except QhullError:
    support_degenerate=True;copmargin=-1.0
   pelvis=apply(T,m.vertices['pelvis']);dep=pelvis@n+d;patch=dep<dep.min()+.002;local=(pelvis-center)@SR;patchmargin=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()));clear=[];floor=[]
   for body,v in m.vertices.items():
    w=apply(T@f[body],v)
    if body not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
    if body in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
    vv=(w-center)@SR;vv[:,0]-=.4;vv=clipped_vertices(vv)
    if len(vv):clear.append((float(vv[:,2].min()),body))
   pe=max(np.linalg.norm((T@f[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in ids);re=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(T@f[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in ids);pose=np.r_[root,Rotation.from_matrix(R).as_quat()];term=terminal_posture_check(pose)
   out=dict(parameters=key,root=pose.tolist(),q=q.tolist(),com=C.tolist(),clearance_m=min(clear)[0],nearest_body=min(clear)[1],foot_position_error_m=float(pe),foot_orientation_error_rad=float(re),static_foot_cop_margin_m=copmargin,support_polygon_degenerate=support_degenerate,pelvis_patch_margin_m=patchmargin,pelvis_gap_m=float(dep.min()),floor_clearance_m=min(floor),terminal=term,measured_height_m=h,measured_slope_deg=slope)
   out['other_model_constraints_passed']=bool(pe<=.003 and re<=np.deg2rad(.5) and copmargin>=.005 and patchmargin>=.015 and min(floor)>=.005 and term['passed']);out['endpoint_screen_passed']=bool(out['other_model_constraints_passed'] and out['clearance_m']>=.003);cache[key]=out;history.append(out);return out
  def cons(x):
   e=evaluate(x);return np.array([(e['static_foot_cop_margin_m']-.00525)*100,(e['pelvis_patch_margin_m']-.01525)*100,(.0029-e['foot_position_error_m'])*100,(np.deg2rad(.49)-e['foot_orientation_error_rad'])*10,(e['floor_clearance_m']-.00525)*100,(38-e['terminal']['model_tilt_deg'])*.1])
  searches=[]
  for start in [(-.17,0,0,np.deg2rad(20),np.deg2rad(20)),(-.17,0,0,np.deg2rad(20),np.deg2rad(-20)),(-.19,0,0,np.deg2rad(25),0.)]:
   fit=minimize(lambda x:-evaluate(x)['clearance_m']*100,start,method='SLSQP',bounds=[(-.28,-.1),(-.05,.05),(-np.deg2rad(12),np.deg2rad(12)),(np.deg2rad(5),np.deg2rad(38)),(-np.deg2rad(35),np.deg2rad(35))],constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=45,ftol=1e-8,eps=2e-5));searches.append(dict(start=list(start),success=bool(fit.success),message=str(fit.message),iterations=int(fit.nit),endpoint=evaluate(fit.x)));print(case,'start complete',fit.message,flush=True)
  valid=[x for x in history if x['other_model_constraints_passed']];best=max(valid,key=lambda x:x['clearance_m']) if valid else None;row=dict(case=case,best=best,searches=searches,history=history);rows.append(row);(O/f'case_{case}.json').write_text(json.dumps(row,indent=2));print(case,{k:best[k] for k in ['parameters','clearance_m','foot_position_error_m','foot_orientation_error_rad','endpoint_screen_passed']} if best else 'NO FEASIBLE',flush=True)
 paths=[Path(__file__),mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz',Path('scripts/low_seat_terminal_contract_42.py'),Path('scripts/run_supported_postures.py')];result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,source_input_sha256={str(p.resolve()):sha(p) for p in paths},scope='OFFLINE bounded spatial fixed-foot planning. Actual1s inputs, frozen inertia/cooked32mesh model; full6joint leg IK. Rootx[-.28,-.10],y±.05,roll±12deg,yaw±35deg,pitch5–38deg with TOTALtilt<=38deg. Existing3mm forbiddenclearance/5mm support/15mm patch screens, fixed previously tested-.35rad arms. Foot-only static support; legal pelvis patch, unchanged physical evaluator. No force generation, dynamic/command-rate/whole-path/self-collision or physical certificate. No runtime/control/asset/perception/criterion changes. Failed limited search is not impossibility proof.',rows=rows);(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()

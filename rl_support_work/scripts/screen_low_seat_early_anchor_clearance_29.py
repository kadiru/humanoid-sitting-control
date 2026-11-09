"""Bounded offline clearance feasibility search; no physics, PPO or holdout.

Earlier recorded foot anchors and previously tested arm offset. Maximize housing/body clearance
subject to existing model support/IK limits and a pitch cap below the unchanged
55-degree fall label. Local searches are not an impossibility proof.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
from low_seat_joint_path_controller_17 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'early_anchor_clearance_screen_29'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False);p=D/'normal_landing3s_06/candidate';t=np.load(p/'trace.npz');ss=np.load(p/'sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(t['t']-1.)))
 model_path=Path('runs/g1_kinematics_geometry_04/model.json');ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json'
 m=BodyModel(model_path);inertia=read(ip);cal=read(cp);results=[]
 ids=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']];arms=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
 for case in [0,2,6,10]:
  q=t['q'][at,case].astype(float);command=t['command'][at,case].astype(float)
  # Saved actor input at step10 is sampled BEFORE the1.0->1.1 step.
  frame=ss[0,10,case,270:405];assert np.array_equal(frame[:23],q[:23].astype(np.float32));gravity=frame[69:72].astype(float)
  n,d,origin,height,slope,_,_=measured(m,q,gravity,t['geometry'][at,case],[t['estimated_root_x'][at,case],0,0],cal)
  R=level_rotation(gravity);fk=m.fk(q);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']};seed=q.copy();seed[arms]-=.35
  SR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0,height]);cache={};history=[]
  def evaluate(x):
   key=tuple(float(v) for v in x)
   if key in cache:return cache[key]
   qi,root,error=solve(m,seed,feet,n,d,key[0],key[1],gap=.001,y=origin[1]);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(qi)
   pelvis=apply(T,m.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002;local=(pelvis-center)@SR
   bias,stat=static_bias(m,inertia,qi,root,float(pelvis[patch,0].mean()),0.)
   foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
   clear=[]
   for name,verts in m.vertices.items():
    if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
    v=(apply(T@f[name],verts)-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
    if len(v):clear.append((float(v[:,2].min()),name))
   c=command.copy();c[ids]=qi[ids]+bias;c[arms]-=.35
   val=dict(x=key[0],pitch=key[1],clearance_m=min(clear)[0],nearest_body=min(clear)[1],cop_margin_m=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x'])),patch_margin_m=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),ik_error=float(error),gap_m=float(depth.min()),endpoint_command_change_rad=float(abs(c-command).max()))
   val['model_constraints_pass']=bool(val['cop_margin_m']>=.005 and val['patch_margin_m']>=.015 and error<=.003 and key[1]<=.9)
   cache[key]=val;history.append(val.copy());return val
  def objective(x):return -evaluate(x)['clearance_m']
  def constraints(x):
   v=evaluate(x);return np.array([v['cop_margin_m']-.00525,v['patch_margin_m']-.01525,.0029-v['ik_error']])
  searches=[]
  for start in [(-.17,.88),(-.20,.85),(-.15,.7)]:
   result=minimize(objective,start,method='SLSQP',bounds=[(-.25,-.1),(.3,.9)],constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=60,ftol=1e-9))
   searches.append(dict(start=start,optimizer_success=bool(result.success),message=str(result.message),iterations=int(result.nit),endpoint=evaluate(result.x)))
  feasible=[v for v in history if v['model_constraints_pass']];best=max(feasible,key=lambda v:v['clearance_m']) if feasible else None
  rr=dict(case=case,measured_height_m=height,measured_slope_deg=slope,measured_gravity=gravity.tolist(),recorded_start_s=1.,fixed_arm_offset_rad=-.35,searches=searches,best_tested_feasible=best,meets_7mm_research_reserve=bool(best and best['clearance_m']>=.007),evaluations=len(history),history=history)
  results.append(rr);(OUT/f'case_{case}.json').write_text(json.dumps(rr,indent=2));print(case,best,flush=True)
 paths=[model_path,ip,cp,p/'trace.npz',p/'sensor_inputs.npz',Path(__file__)];hashes={str(x.resolve()):sha(x) for x in paths}
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline bounded local search at recorded1s measured inputs. Fixed feet, fixed previously tested arm offset; no runtime or physical threshold change.7mm is an engineering reserve motivated by measured tracking/perception error, not an admission threshold. No whole-path/rate or physical pass is claimed. Search failure is not proof of impossibility.',simulations=0,optimizer_ppo_updates=0,fresh_holdout=False,source_input_hashes=hashes,results=results)
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

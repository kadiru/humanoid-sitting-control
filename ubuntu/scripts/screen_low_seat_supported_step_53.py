"""Offline full-foot-pose stepping prerequisite, not a physical controller.

Two explicitly bounded common rearward stance displacements are examined with
lift/land phases, full six-joint leg IK, static single/double support, sampled
seat/floor clearance and all-leg gravity feedforward. No feet are moved in a
simulator. A failed prerequisite blocks any physical trajectory.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_joint_path_controller_17 import measured
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'supported_step_screen_53'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=False);src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors']
 mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json'
 m=BodyModel(mp);md=read(mp);ine=read(ip);cal=read(cp);mass=np.asarray(ine['mass']);W=mass.sum()*9.81
 ids={s:[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']] for s in ['left','right']};legs=ids['left']+ids['right'];limits={j['name']:np.deg2rad(j['limits']) for j in md['joints'] if j['type']=='PhysicsRevoluteJoint'}
 kp=np.array([200.,150.,150.,200.,160.,160.]*2)
 case=2;at=int(np.argmin(abs(tr['t']-1.)));q0=tr['q'][at,case].astype(float);g=ss[0,10,case,339:342].astype(float)
 assert abs(tr['t'][at]-1)<1e-9 and np.array_equal(q0[:23].astype(np.float32),ss[0,10,case,270:293])
 n,d,origin,h,slope,_,_=measured(m,q0,g,tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],cal);R=level_rotation(g);f=m.fk(q0)
 feet0={s:(R@f[s+'_ankle_roll_link'][:3,3]+origin,R@f[s+'_ankle_roll_link'][:3,:3]) for s in ids};SR=Rotation.from_euler('y',slope,degrees=True).as_matrix();center=np.array([-.4,0,h]);allrows=[]
 def ik(seed,root,feet):
  q=seed.copy()
  for side in ids:
   ii=ids[side];pos,rot=feet[side]
   def residual(x):
    qq=q.copy();qq[ii]=x;F=root@m.fk(qq)[side+'_ankle_roll_link'];return np.r_[F[:3,3]-pos,.2*Rotation.from_matrix(rot.T@F[:3,:3]).as_rotvec()]
   lo=np.array([limits[m.names[j]][0]+.031 for j in ii]);hi=np.array([limits[m.names[j]][1]-.031 for j in ii]);fit=least_squares(residual,np.clip(q[ii],lo,hi),bounds=(lo,hi),max_nfev=50,ftol=1e-9,xtol=1e-9,gtol=1e-9);q[ii]=fit.x
  return q
 def com(q,root):
  f=m.fk(q);return np.average([apply(root@f[b],c) for b,c in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass)
 def sole(q,root,side):
  v=apply(root@m.fk(q)[side+'_ankle_roll_link'],m.vertices[side+'_ankle_roll_link']);return v[v[:,2]<v[:,2].min()+.003]
 def margin(poly,p):
  hu=ConvexHull(poly[:,:2]);return float(np.min(-(hu.equations[:,:2]@p[:2]+hu.equations[:,2])))
 for shift in [-.06,-.10]:
  feet={s:(p.copy(),rot.copy()) for s,(p,rot) in feet0.items()};q=q0.copy();root=transform(R,origin);c0=com(q,root);nodes=[]
  # Each COM target is above a support-sole centre. Root adjustment is solved,
  # rather than assumed equal to the COM shift.
  points={s:sole(q,root,s).mean(0) for s in ids};mid=(points['left']+points['right'])/2
  phases=[('shift_to_left','both',points['left'][:2],None),('right_lift_land','left',points['left'][:2],'right'),('shift_to_right','both',points['right'][:2]+[shift,0],None),('left_lift_land','right',points['right'][:2]+[shift,0],'left'),('recenter','both',mid[:2]+[shift,0],None)]
  for name,support,target,swing in phases:
   start_c=com(q,root)[:2].copy();start_feet={s:(p.copy(),rr.copy()) for s,(p,rr) in feet.items()};start_root=root.copy();last_q=q.copy();phase=[]
   for k,u in enumerate(np.linspace(0,1,31)):
    a=u*u*(3-2*u);wanted=start_c+(target-start_c)*a;feet={s:(p.copy(),rr.copy()) for s,(p,rr) in start_feet.items()}
    if swing:
     # Lift vertically first, translate only when airborne, land vertically.
     lift=.035*(u/.25)**2*(3-2*u/.25) if u<.25 else (.035 if u<=.75 else .035*((1-u)/.25)**2*(3-2*(1-u)/.25))
     v=np.clip((u-.25)/.5,0,1);v=v*v*(3-2*v);feet[swing][0][0]+=shift*v;feet[swing][0][2]+=lift
    root=start_root.copy()
    for _ in range(6):
     q=ik(last_q,root,feet);err=com(q,root)[:2]-wanted;root[:2,3]-=err
     if np.max(abs(err))<1e-7:break
    q=ik(q,root,feet);last_q=q.copy();f=m.fk(q);C=com(q,root)
    pose=max(np.linalg.norm((root@f[s+'_ankle_roll_link'])[:3,3]-feet[s][0]) for s in ids);rot_err=max(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@(root@f[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in ids)
    active=list(ids) if support=='both' or (swing and k in [0,30]) else [support];poly=np.concatenate([sole(q,root,s) for s in active]);cop_margin=margin(poly,C)
    weights={s:0. for s in ids}
    if len(active)==1:weights[active[0]]=1.
    else:
     yl=(root@f['left_ankle_roll_link'])[1,3];yr=(root@f['right_ankle_roll_link'])[1,3];weights['left']=float(np.clip((C[1]-yr)/(yl-yr),0,1));weights['right']=1-weights['left']
    footcenters={s:sole(q,root,s).mean(0) for s in ids};dx=C[0]-sum(weights[s]*footcenters[s][0] for s in ids);contact={s:np.linalg.inv(root@f[s+'_ankle_roll_link'])@np.r_[footcenters[s][0]+dx,footcenters[s][1],footcenters[s][2],1] for s in ids}
    tor=[];eps=1e-5
    for j in legs:
     vals=[]
     for sign in [-1,1]:
      qq=q.copy();qq[j]+=sign*eps;ff=m.fk(qq);potential=sum(mm*9.81*apply(root@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],mass,ine['local_com']));work=sum(W*weights[s]*(root@ff[s+'_ankle_roll_link']@contact[s])[2] for s in ids);vals.append(potential-work)
     tor.append((vals[1]-vals[0])/(2*eps))
    cmd=q.copy();cmd[legs]+=np.array(tor)/kp
    forbidden=[];floor=[]
    for b,v in m.vertices.items():
     w=apply(root@f[b],v)
     if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
     if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
     vv=(w-center)@SR;vv[:,0]-=.4;vv=clipped_vertices(vv)
     if len(vv):forbidden.append(float(vv[:,2].min()))
    phase.append(dict(phase=name,u=float(u),support=active,q=q.tolist(),command=cmd.tolist(),root=np.r_[root[:3,3],Rotation.from_matrix(root[:3,:3]).as_quat()].tolist(),com=C.tolist(),foot_pose_error_m=float(pose),foot_rotation_error_rad=float(rot_err),static_cop_margin_m=cop_margin,floor_clearance_m=min(floor),forbidden_seat_clearance_m=min(forbidden) if forbidden else None,foot_targets={s:p.tolist() for s,(p,rr) in feet.items()},joint_command_margin_rad=float(min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in legs))))
   commands=np.array([x['command'] for x in phase]);arc=float(np.sum(np.max(abs(np.diff(commands,axis=0)),axis=1)));nodes.extend(phase);print(shift,name,'arc',arc,'cop',min(x['static_cop_margin_m'] for x in phase),'IK',max(x['foot_pose_error_m'] for x in phase),flush=True)
  # Discontinuities between phase commands are included, not discarded.
  commands=np.array([x['command'] for x in nodes]);arc=float(np.sum(np.max(abs(np.diff(commands,axis=0)),axis=1)));necessary_time=arc/.15
  gates=dict(full_foot_pose=max(x['foot_pose_error_m'] for x in nodes)<=.003,full_foot_rotation=max(x['foot_rotation_error_rad'] for x in nodes)<=np.deg2rad(.5),static_support=min(x['static_cop_margin_m'] for x in nodes)>=.005,floor=min(x['floor_clearance_m'] for x in nodes)>=.005,seat_clearance=min(x['forbidden_seat_clearance_m'] for x in nodes if x['forbidden_seat_clearance_m'] is not None)>=.003,command_joint_margin=min(x['joint_command_margin_rad'] for x in nodes)>=.03,necessary_time_before15s=necessary_time<=14.)
  row=dict(shift_m=shift,necessary_command_arc_rad=arc,necessary_step_duration_s_at015=necessary_time,gate=gates,all_prerequisite_gates_passed=bool(all(gates.values())),nodes=nodes);allrows.append(row);(OUT/f'shift_{int(abs(shift)*1000)}mm.json').write_text(json.dumps(row,indent=2))
 paths=[Path(__file__),mp,ip,cp,src/'trace.npz',src/'sensor_inputs.npz'];result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_ticks=0,optimizer_updates=0,holdout_used=False,scope='OFFLINE stepping prerequisite for36cm/+20deg using actual1s sensors, frozen inertia and cooked collision model. Explicit35mm vertical lift, airborne horizontal translation and vertical landing; full6joint leg IK and static single/double support. Two bounded rearward foot shifts60/100mm. Original upper body retained. Command arc is a NECESSARY lower bound at0.15rad/s; acceleration/ZMP/friction/self-collision/tracking, actual load transfer and following sit NOT certified. No executable or physical admission regardless of static pass. No feet moved/reset/teleported in simulation.',source_input_sha256={str(p.resolve()):sha(p) for p in paths},rows=allrows);(OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()

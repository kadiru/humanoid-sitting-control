"""Offline contact-transition path screen, never an executable controller.

Unlike the old endpoint-only fixed-pivot screen, retain the active material
point's tangential position at each small rotation while enforcing the finite
seat plane and fixed feet. Compute the minimum assumed seat load needed for
the foot CoP margin along the complete path. Actual load must be sensed before
any later controller may consume this path; no physics or force oracle here.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
from low_seat_joint_path_controller_17 import measured
from low_seat_terminal_contract_42 import terminal_posture_check
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'rolling_contact_screen_44'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 OUT.mkdir(exist_ok=False)
 mp=Path('runs/g1_kinematics_geometry_04/model.json');ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json'
 model=BodyModel(mp);inertia=read(ip);cal=read(cp);meta=read(mp);base=D/'normal_landing3s_06/candidate';trace=np.load(base/'trace.npz');sensors=np.load(base/'sensor_inputs.npz')['sensors'];endpoints=read(D/'terminal_screen_42/results.json')
 limits={j['name']:np.deg2rad(j['limits']) for j in meta['joints'] if j['type']=='PhysicsRevoluteJoint'}
 legs=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']];arms=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
 at=int(np.argmin(abs(trace['t']-1.)));rows=[]
 for case in [0,2,6,10]:
  endpoint=next(r for r in endpoints['rows'] if r['case']==case and r['assumption']=='foot_only_before_contact')['best_other_screens_feasible'];assert endpoint is not None
  q0=trace['q'][at,case].astype(float);cmd0=trace['command'][at,case].astype(float);frame=sensors[0,10,case,270:405];g=frame[69:72];geometry=trace['geometry'][at,case]
  assert np.array_equal(frame[:23],q0[:23].astype(np.float32))
  n,d,origin,h,slope,_,_=measured(model,q0,g,geometry,[trace['estimated_root_x'][at,case],0,0],cal);n=np.asarray(n);n/=np.linalg.norm(n)
  tangent=np.array([n[2],0.,-n[0]]);tangent/=np.linalg.norm(tangent);A=np.stack([tangent,n,[0.,1.,0.]])
  R0=level_rotation(g);fk0=model.fk(q0);feet={s:(R0@fk0[s+'_ankle_roll_link'][:3,3]+origin,R0@fk0[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
  seed=q0.copy();seed[arms]-=.35;seatR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0,h]);vertices=model.vertices['pelvis']
  # One specified rolling direction, not an angle/gain search. Smooth38->0deg
  # over4.5s leaves max nominal angular speed below the existing.25rad/s gate.
  time=np.linspace(0,4.5,181);u=time/4.5;angles=endpoint['pitch_rad']*(1-u*u*(3-2*u));path=[];previous_root=None;previous_R=None;previous_vertex=None
  for k,(t,angle) in enumerate(zip(time,angles)):
   R=Rotation.from_euler('y',angle).as_matrix();rotated=vertices@R.T
   if k==0:x=endpoint['x']
   else:
    along=float(tangent@previous_root[:3]+tangent@(previous_R-R)@previous_vertex)
    target_normal=.001-d-float((rotated@n).min());xyz=np.linalg.solve(A,[along,target_normal,origin[1]]);x=float(xyz[0])
   qi,root,err=solve(model,seed,feet,n,d,x,float(angle),gap=.001,y=origin[1]);T=transform(R,root[:3]);fk=model.fk(qi);pelvis=apply(T,vertices);depth=pelvis@n+d
   patch=depth<depth.min()+.002;local=(pelvis-center)@seatR;contactx=float(pelvis[patch,0].mean())
   foot=np.concatenate([apply(T@fk[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
   bias0,stat0=static_bias(model,inertia,qi,root,contactx,0.);_,statfull=static_bias(model,inertia,qi,root,contactx,1.)
   target_cop=float(sole[:,0].min()+.0055);comx=stat0['com'][0];required=max(0.,(target_cop-comx)/(target_cop-contactx)) if target_cop>contactx else float('inf')
   fraction=required/max(statfull['predicted_seat_bw'],1e-12);bias,stat=static_bias(model,inertia,qi,root,contactx,float(np.clip(fraction,0,1)))
   command=cmd0.copy();command[legs]=qi[legs]+bias;command[arms]+=qi[arms]-q0[arms]
   clear=[];floor=[]
   for name,verts in model.vertices.items():
    world=apply(T@fk[name],verts)
    if name not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append((float(world[:,2].min()),name))
    if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
    v=(world-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
    if len(v):clear.append((float(v[:,2].min()),name))
   footpos=max(float(np.linalg.norm(apply(T@fk[s+'_ankle_roll_link'],[0,0,0])-feet[s][0])) for s in ['left','right'])
   footrot=max(float(np.linalg.norm(Rotation.from_matrix(feet[s][1].T@R@fk[s+'_ankle_roll_link'][:3,:3]).as_rotvec())) for s in ['left','right'])
   residual=0. if k==0 else float(tangent@(root[:3]+R@previous_vertex-previous_root[:3]-previous_R@previous_vertex))
   margin=min(min(command[j]-limits[model.names[j]][0],limits[model.names[j]][1]-command[j]) for j in legs+arms)
   r=dict(time_s=float(t),pitch_deg=float(np.rad2deg(angle)),x=float(x),root=root.tolist(),q=qi.tolist(),command=command.tolist(),clearance_m=min(clear)[0],nearest_body=min(clear)[1],pelvis_gap_m=float(depth.min()),patch_margin_m=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),floor_clearance_m=min(floor)[0],assumed_minimum_vertical_seat_share=required,achievable_static_allocation_share=stat['predicted_seat_bw'],fraction_of_model_allocation=fraction,foot_cop_margin_m=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x'])),foot_position_error_m=footpos,foot_rotation_error_rad=footrot,ik_error=float(err),command_joint_margin_rad=float(margin),previous_contact_material_tangent_residual_m=residual,terminal_check=terminal_posture_check(root))
   r['model_constraints_passed']=bool(r['clearance_m']>=.003 and r['patch_margin_m']>=.015 and r['floor_clearance_m']>=.005 and r['foot_cop_margin_m']>=.005 and r['command_joint_margin_rad']>=.03 and err<=.003 and footpos<=.003 and footrot<=.003 and required<=.65 and fraction<=1. and r['terminal_check']['passed'])
   path.append(r);previous_root=root;previous_R=R;previous_vertex=vertices[int(np.argmin(depth))];seed=qi
  commands=np.array([r['command'] for r in path]);rates=np.abs(np.diff(commands,axis=0))/(time[1]-time[0]);failed=[i for i,r in enumerate(path) if not r['model_constraints_passed']]
  result=dict(case=case,measured_height_m=h,measured_slope_deg=slope,physical_ticks=0,starting_endpoint=endpoint,assumption='Active seat contact is hypothetical; force allocation is static and must never be consumed as a measured load.',nominal_duration_s=4.5,nominal_peak_command_rate_rad_s=float(rates.max()),max_contact_material_tangent_residual_m=max(abs(r['previous_contact_material_tangent_residual_m']) for r in path),first_failed_knot=failed[0] if failed else None,failed_knots=len(failed),whole_path_model_passed=not failed and rates.max()<=.175,minimum_clearance_m=min(r['clearance_m'] for r in path),maximum_required_vertical_seat_share=max(r['assumed_minimum_vertical_seat_share'] for r in path),path=path)
  rows.append(result);(OUT/f'case_{case}.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['path','starting_endpoint']},flush=True)
 paths=[Path(__file__),mp,ip,cp,base/'trace.npz',base/'sensor_inputs.npz',D/'terminal_screen_42/results.json',Path('scripts/low_seat_terminal_contract_42.py')]
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline evolving-contact material-point rolling path, fixed measured feet, fixed previously tested arms, no simulator, optimizer or holdout. Unlike prior endpoint-only rolling, checks each small step and required support allocation. Full dynamic force/contact transition, sensed loading, collision approximation and whole approach are NOT validated. No endpoint teleportation or physical trial authorized by this result.',sources={str(p.resolve()):sha(p) for p in paths},physics_ticks=0,optimizer_updates=0,holdout_used=False,rows=rows)
 (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()

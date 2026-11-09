"""Offline staged approach feasibility; unchanged actuator cap, no physics.
Tests two explicitly bounded earlier starts, then a stationary hover and a normal
landing. Does not certify dynamic balance, force feasibility or physical safety.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices
from low_seat_joint_path_controller_17 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'staged_headroom_screen_35'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 p=D/'normal_landing3s_06/candidate';t=np.load(p/'trace.npz');ss=np.load(p/'sensor_inputs.npz')['sensors']
 mp=Path('runs/g1_kinematics_geometry_04/model.json');ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';ep=D/'early_anchor_clearance_screen_29/results.json'
 m=BodyModel(mp);inertia=read(ip);cal=read(cp);endpoints={v['case']:v['best_tested_feasible'] for v in read(ep)['results']}
 legs=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']];arms=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];rows=[]
 for start in [1.]:
  for case in [0,2,6,10]:
   at=int(np.argmin(abs(t['t']-start)));q=t['q'][at,case].astype(float);cmd=t['command'][at,case].astype(float);frame=ss[0,int(start*10),case,270:405]
   assert np.array_equal(frame[:23],q[:23].astype(np.float32));g=frame[69:72].astype(float);assert t['geometry'][at,case,11] and t['geometry'][at,case,12]<=.5
   n,d,origin,height,slope,_,_=measured(m,q,g,t['geometry'][at,case],[t['estimated_root_x'][at,case],0,0],cal)
   R=level_rotation(g);fk=m.fk(q);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']};seed=q.copy();seed[arms]-=.35
   endpoint=endpoints[case];x=endpoint['x'];pitch=endpoint['pitch'];hover_gap=.02
   hover,hover_root,hover_err=solve(m,seed,feet,n,d,x+(hover_gap-.001)*n[0],pitch,gap=hover_gap,y=origin[1])
   root0=np.r_[origin,Rotation.from_matrix(R).as_quat()];bias0,_=static_bias(m,inertia,q,root0,-.2,0.);preload=cmd[legs]-q[legs]-bias0
   SR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0,height])
   def point(qi,pre):
    f=m.fk(qi);rot=[feet[s][1]@f[s+'_ankle_roll_link'][:3,:3].T for s in ['left','right']];rr=rot[0];pos=[feet[s][0]-rr@f[s+'_ankle_roll_link'][:3,3] for s in ['left','right']];xyz=np.mean(pos,axis=0);T=transform(rr,xyz);root=np.r_[xyz,Rotation.from_matrix(rr).as_quat()]
    pelvis=apply(T,m.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002;loc=(pelvis-center)@SR;sx=float(pelvis[patch,0].mean());bias,stat=static_bias(m,inertia,qi,root,sx,0.)
    c=cmd.copy();c[legs]=qi[legs]+bias+pre*preload;c[arms]=cmd[arms]+qi[arms]-q[arms]
    foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003];clear=[]
    for name,vertices in m.vertices.items():
     if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
     v=(apply(T@f[name],vertices)-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
     if len(v):clear.append((float(v[:,2].min()),name))
    info=dict(root=root.tolist(),gap_m=float(depth.min()),clearance_m=min(clear)[0] if clear else 1.,nearest_body=min(clear)[1] if clear else None,patch_margin_m=float(min(.25-abs(loc[patch,0]).max(),.45-abs(loc[patch,1]).max())),cop_margin_m=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x'])),foot_position_error_m=float(np.linalg.norm(pos[0]-pos[1])),foot_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(rot[0].T@rot[1]).as_rotvec())))
    return c,info
   approach=[point(q+f*(hover-q),1-f) for f in np.linspace(0,1,97)];approach[0]=(cmd,approach[0][1]);landing=[];errors=[hover_err]
   for f in np.linspace(0,1,49):
    gap=hover_gap+f*(.001-hover_gap);qi,root,err=solve(m,seed,feet,n,d,x+(gap-.001)*n[0],pitch,gap=gap,y=origin[1]);errors.append(err);landing.append(point(qi,0.))
   arcs=[float(np.max(abs(np.diff(np.array([v[0] for v in part]),axis=0)),axis=1).sum()) for part in [approach,landing]]
   checks=[v[1] for v in approach+landing];near=[v for v in checks if v['gap_m']<.02]
   s=dict(min_clearance_m=min(v['clearance_m'] for v in checks),min_cop_m=min(v['cop_margin_m'] for v in checks),min_gap_m=min(v['gap_m'] for v in checks),min_near_patch_m=min(v['patch_margin_m'] for v in near),max_foot_position_error_m=max(v['foot_position_error_m'] for v in checks),max_foot_rotation_error_rad=max(v['foot_rotation_error_rad'] for v in checks),max_ik_error=max(errors))
   timings=[]
   for rate in [.15,.175]:
    #0.25s rate ramps,0.5s hover hold,>=1.5s descent,0.5s settling before15s.
    ta=arcs[0]/rate+.25;tl=max(1.5,arcs[1]/rate+.25);finish=start+ta+.5+tl
    timings.append(dict(nominal_rate_rad_s=rate,approach_duration_s=ta,hover_hold_s=.5,landing_duration_s=tl,end_s=finish,settled_by_15s=finish+.5<=15.))
   geom=bool(s['min_clearance_m']>=.003 and s['min_cop_m']>=.005 and s['min_gap_m']>=0 and s['min_near_patch_m']>=.015 and s['max_foot_position_error_m']<.003 and s['max_foot_rotation_error_rad']<.003 and s['max_ik_error']<.003)
   row=dict(case=case,start_s=start,measured_height_m=height,measured_slope_deg=slope,endpoint_parameters_from_development_screen29=dict(x=x,pitch=pitch),hover_gap_m=hover_gap,arm_offset_rad=-.35,command_arcs_rad=arcs,model=s,geometric_screen_pass=geom,timings=timings,screen_pass_original=geom and timings[0]['settled_by_15s'],screen_pass_proposed=geom and timings[1]['settled_by_15s'],approach=[v[1] for v in approach],landing=[v[1] for v in landing])
   rows.append(row);(OUT/f'case_{case}_start{int(start)}.json').write_text(json.dumps(row,indent=2));print(json.dumps({k:v for k,v in row.items() if k not in ['approach','landing']}),flush=True)
 paths=[Path(__file__),mp,ip,cp,ep,p/'trace.npz',p/'sensor_inputs.npz'];out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline staged approach with recorded1s sensors, earlier1s-anchor endpoints29, fixed previously tested-.35rad arms,20mm hover,0.5s hold,1.5s normal descent and0.5s settling. Compare original.15 nominal rate with.175 proposed nominal under separately tested.20 applied cap. Existing3mm clearance/5mm support/15mm patch model screens remain;7mm reserve was not achieved for the two harder geometries in33/34, so this is NOT a robust safety certificate. The hypothesis is that stopping before normal landing reduces tracking lag. No physics or runtime change.',simulations=0,optimizer_updates=0,holdout_used=False,source_input_hashes={str(p.resolve()):sha(p) for p in paths},rows=rows)
 (OUT/'results.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()

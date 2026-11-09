"""Bounded offline endpoint optimization with a path-error buffer for the remaining36cm/+20deg case.

This does not run physics, change current controllers or relax physical criteria.
The previously tested0.35rad arm offset is fixed, not swept; its model COM shift
allows the endpoint solver to explore rearward placement while retaining foot
support. A successful endpoint still needs a whole-path and physical check.
"""
import json,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from build_near_seat_candidates import solve
from plan_seat_load_targets import clipped_vertices
from sagittal_static_bias import static_bias
from low_seat_joint_path_controller_17 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def main():
    p=D/'normal_landing3s_06/candidate';t=np.load(p/'trace.npz');idx=int(np.argmin(abs(t['t']-8.)));i=2
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');inertia=json.loads((p/'static_inertial_model.json').read_text());cal=json.loads((D.parent/'contact_effort_repair_01/plane_calibration.json').read_text())
    q=t['q'][idx,i].astype(float);command=t['command'][idx,i].astype(float);gravity=Rotation.from_quat(t['body'][idx,i,0,3:]).inv().apply([0,0,-1.])
    n,d,origin,height,slope,_,_=measured(m,q,gravity,t['geometry'][idx,i],[t['estimated_root_x'][idx,i],0,0],cal)
    R=level_rotation(gravity);fk=m.fk(q);feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    arms=[m.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    seed=q.copy();seed[arms]-=.35;SR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0,height]);history=[]
    def metrics(qi,root):
        T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(qi);v=apply(T,m.vertices['pelvis']);depth=v@n+d;patch=depth<depth.min()+.002;local=(v-center)@SR
        bias,stat=static_bias(m,inertia,qi,root,float(v[patch,0].mean()),0.)
        foot=np.concatenate([apply(T@f[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=foot[foot[:,2]<foot[:,2].min()+.003]
        cop=float(min(stat['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-stat['foot_cop_x']))
        clear=[]
        for name,verts in m.vertices.items():
            if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
            v=(apply(T@f[name],verts)-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
            if len(v):clear.append((float(v[:,2].min()),name))
        return bias,dict(clearance_m=min(clear)[0],nearest_forbidden_body=min(clear)[1],cop_margin_m=cop,
            patch_margin_m=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),gap_m=float(depth.min()))
    @lru_cache(maxsize=None)
    def endpoint(x,pitch):
        qi,root,error=solve(m,seed,feet,n,d,x,pitch,gap=.001,y=origin[1]);bias,v=metrics(qi,root)
        c=command.copy();c[ids]=qi[ids]+bias;c[arms]-=.35
        v.update(x=x,pitch=pitch,ik_error=error,endpoint_max_command_change_rad=float(abs(c-command).max()))
        history.append(v.copy());return qi,root,c,v
    def objective(x):return endpoint(float(x[0]),float(x[1]))[3]['endpoint_max_command_change_rad']
    def constraints(x):
        v=endpoint(float(x[0]),float(x[1]))[3]
        return np.array([v['clearance_m']-.0033,v['cop_margin_m']-.00525,v['patch_margin_m']-.015,.003-v['ik_error']])
    from types import SimpleNamespace
    prior=json.loads((D/'clearance_endpoint_buffer_screen_19.json').read_text())
    result=SimpleNamespace(x=np.array([prior['endpoint']['x'],prior['endpoint']['pitch']]),success=True,message='Fixed screen19 endpoint replay; no new optimization',nit=0,nfev=0)
    end,root,cmd,values=endpoint(float(result.x[0]),float(result.x[1]));initial_root=np.r_[origin,Rotation.from_matrix(R).as_quat()]
    b0,_=static_bias(m,inertia,q,initial_root,-.2,0.);preload=command[ids]-q[ids]-b0;commands=[];path=[]
    for u in np.linspace(0,1,97):
        qi=q+u*(end-q);f=m.fk(qi);rots=[feet[s][1]@f[s+'_ankle_roll_link'][:3,:3].T for s in ['left','right']];rr=rots[0]
        positions=[feet[s][0]-rr@f[s+'_ankle_roll_link'][:3,3] for s in ['left','right']];xyz=np.mean(positions,axis=0);r=np.r_[xyz,Rotation.from_matrix(rr).as_quat()]
        bias,check=metrics(qi,r);c=command.copy();c[ids]=qi[ids]+bias+(1-u)*preload;c[arms]-=.35*u
        commands.append(c);check.update(fraction=float(u),root=r.tolist(),foot_position_error_m=float(np.linalg.norm(positions[0]-positions[1])),foot_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(rots[0].T@rots[1]).as_rotvec())))
        path.append(check)
    commands[0]=command;arc=float(np.max(abs(np.diff(commands,axis=0)),axis=1).sum());near=[x for x in path if x['gap_m']<.02]
    incoming=(t['command'][idx,i].astype(float)-t['command'][idx-1,i].astype(float))/.1
    leading=np.asarray(commands[1])-np.asarray(commands[0]);leading=leading/np.max(abs(leading))
    initial_arc_speed=float(np.clip(incoming@leading/(leading@leading),0,.15))
    peak=(arc-.125*initial_arc_speed)/6.75
    check=dict(initial_arc_speed_rad_s=initial_arc_speed,nominal_retimed_rate_rad_s=max(initial_arc_speed,peak),cruise_arc_speed_rad_s=peak,incoming_applied_command_rate=incoming.tolist(),min_clearance_m=min(x['clearance_m'] for x in path),min_cop_margin_m=min(x['cop_margin_m'] for x in path),
        minimum_near_patch_margin_m=min(x['patch_margin_m'] for x in near),minimum_gap_m=min(x['gap_m'] for x in path),max_foot_position_error_m=max(x['foot_position_error_m'] for x in path),max_foot_rotation_error_rad=max(x['foot_rotation_error_rad'] for x in path))
    check['passed']=bool(check['nominal_retimed_rate_rad_s']<=.15 and check['min_clearance_m']>=.003 and check['min_cop_margin_m']>=.005 and check['minimum_near_patch_margin_m']>0 and check['minimum_gap_m']>=0 and check['max_foot_position_error_m']<.003 and check['max_foot_rotation_error_rad']<.003)
    out=D/'clearance_velocity_screen_20.json';assert not out.exists()
    out.write_text(json.dumps(dict(scope='Recorded-velocity timing check on the fixed screen19 endpoint, with no further endpoint optimization. Starting applied velocity is projected onto the path tangent and clamped to the unchanged0.15rad/s limit;0.25s blends connect it to cruise speed and zero endpoint speed. This accounts for the nonzero incoming command velocity rather than assuming an initial stop. Fixed previously tested arm offset; objective endpoint command travel, constrained by3.3mm endpoint forbidden clearance and5.25mm endpoint CoP margin after screen18 showed a0.247mm whole-path clearance reduction; the unchanged whole-path screens remain3mm clearance and5mm CoP margin,15mm endpoint patch and3mm IK. This is not a changed physical threshold, runtime controller, causal proof or physical admission.',
        physical_simulations=0,optimizer_ppo_updates=0,fresh_holdout=False,optimizer=dict(success=bool(result.success),message=str(result.message),iterations=int(result.nit),evaluations=int(result.nfev)),
        endpoint=values,whole_path=check,path=path,history=history,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(optimizer_success=bool(result.success),endpoint=values,whole_path=check),indent=2))
if __name__=='__main__':main()

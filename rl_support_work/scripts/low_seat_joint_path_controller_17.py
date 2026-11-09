"""Selected low-seat approach from8s; nominal actuator-rate-aware joint path.

Pre-training research candidate. Simulator truth is not consumed. Only the
three development geometries that passed the rate/clearance model screen are
targeted by measured height<38cm and14<slope<18deg. The20deg housing case keeps
the previous controller and remains unresolved. No physical admission implied.
"""
import json,inspect,textwrap,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform,level_rotation
from humanoid_sitting.contact_aware_reference import measured_level_plane
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from plan_seat_load_targets import clipped_vertices
import low_seat_normal_duration_17 as normal

def measured(model,q,gravity,geometry,reference,cal):
    rn,rd,origin=measured_level_plane(model,q,gravity,geometry,reference)
    cn=np.array(cal['reference_normal']);cd=cal['reference_offset']
    height=.4-(-.4*rn[0]+rd)/rn[2]+(-.4*cn[0]+cd)/cn[2]
    slope=np.arctan2(rn[0],rn[2])-np.arctan2(cn[0],cn[2])
    n=Rotation.from_euler('y',slope).as_matrix()[:,2];d=-n@np.array([-.4,0.,height])
    return n,d,origin,height,float(np.rad2deg(slope)),rn,rd

def eligible(plane):return bool(plane[3]<.38 and 14.<plane[4]<18.)

def plan(model,q,gravity,geometry,reference,command,cal,inertial,config):
    n,d,origin,height,slope,rn,rd=measured(model,q,gravity,geometry,reference,cal)
    assert height<.38 and 14.<slope<18.
    q=np.asarray(q,dtype=float);command=np.asarray(command);R=level_rotation(gravity);fk=model.fk(q)
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    end,endroot,error=solve(model,q,feet,n,d,-.17,.9,gap=.001,y=origin[1]);assert error<.003
    root0=np.r_[origin,Rotation.from_matrix(R).as_quat()];bias0,_=static_bias(model,inertial,q,root0,-.2,0.)
    preload=command[ids]-q[ids]-bias0;seatR=Rotation.from_euler('y',np.deg2rad(slope)).as_matrix();center=np.array([-.4,0.,height])
    def point(fraction,screen=False):
        qi=q+fraction*(end-q);f=model.fk(qi)
        rotations=[feet[s][1]@f[s+'_ankle_roll_link'][:3,:3].T for s in ['left','right']]
        rr=rotations[0];positions=[feet[s][0]-rr@f[s+'_ankle_roll_link'][:3,3] for s in ['left','right']]
        xyz=np.mean(positions,axis=0);T=transform(rr,xyz);root=np.r_[xyz,Rotation.from_matrix(rr).as_quat()]
        pelvis=apply(T,model.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002
        local=(pelvis-center)@seatR;sx=float(pelvis[patch,0].mean())
        b0,s0=static_bias(model,inertial,qi,root,sx,0.)
        cmd=command.copy();cmd[ids]=qi[ids]+b0+(1-fraction)*preload
        info={}
        if screen:
            fp=np.concatenate([apply(T@f[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ['left','right']]);sole=fp[fp[:,2]<fp[:,2].min()+.003]
            clear=[]
            for name,vertices in model.vertices.items():
                if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
                v=(apply(T@f[name],vertices)-center)@seatR;v[:,0]-=.4;v=clipped_vertices(v)
                if len(v):clear.append(float(v[:,2].min()))
            info=dict(gap=float(depth.min()),patch=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max())),
                clearance=min(clear) if clear else 1.,cop=float(min(s0['foot_cop_x']-sole[:,0].min(),sole[:,0].max()-s0['foot_cop_x'])),
                foot_position_error=float(np.linalg.norm(positions[0]-positions[1])),
                foot_rotation_error=float(np.linalg.norm(Rotation.from_matrix(rotations[0].T@rotations[1]).as_rotvec())))
        return cmd,qi,root,b0,s0,sx,info
    grid=np.linspace(0,1,97);dense=[point(x,True) for x in grid];commands=np.array([x[0] for x in dense]);commands[0]=command
    arc=np.r_[0.,np.cumsum(np.max(abs(np.diff(commands[:,ids],axis=0)),axis=1))];peak=float(arc[-1]/6.75)
    checks=[x[-1] for x in dense];near=[x for x in checks if x['gap']<.02]
    screen=dict(peak_nominal_rate_rad_s=peak,minimum_clearance_m=min(x['clearance'] for x in checks),
        minimum_gap_m=min(x['gap'] for x in checks),minimum_near_patch_margin_m=min(x['patch'] for x in near),endpoint_patch_margin_m=checks[-1]['patch'],
        minimum_no_seat_cop_margin_m=min(x['cop'] for x in checks),max_foot_position_error_m=max(x['foot_position_error'] for x in checks),
        max_foot_rotation_error_rad=max(x['foot_rotation_error'] for x in checks),goal_ik_error=error)
    assert peak<=.15 and screen['minimum_clearance_m']>=.003 and screen['minimum_gap_m']>=0 and screen['minimum_near_patch_margin_m']>0 and screen['endpoint_patch_margin_m']>=.015 and screen['minimum_no_seat_cop_margin_m']>=.005 and screen['max_foot_position_error_m']<.003 and screen['max_foot_rotation_error_rad']<.003,screen
    cs=[];qs=[];roots=[];zero=[];full=[];shares=[];statics=[];fractions=[]
    for elapsed in np.linspace(0,7,25):
        if elapsed<.25:distance=.5*elapsed**2/.25
        elif elapsed>6.75:distance=6.75-.5*(7-elapsed)**2/.25
        else:distance=elapsed-.125
        fraction=float(np.interp(distance/6.75*arc[-1],arc,grid));fractions.append(fraction)
        c,qi,root,b0,s0,sx,_=point(fraction)
        # Interpolate the audited dense command so component secant rates stay
        # bounded after reparameterization; static correction uses that basis.
        c=command.copy()
        for j in ids:c[j]=np.interp(fraction,grid,commands[:,j])
        effective_zero=c[ids]-qi[ids]-(1-fraction)*preload
        b1,s1=static_bias(model,inertial,qi,root,sx,1.)
        cs.append(c);qs.append(qi);roots.append(root.tolist());zero.append(effective_zero.tolist());full.append(b1.tolist());shares.append(s1['predicted_seat_bw'])
        statics.append(dict(s1,predicted_seat_bw=0.,torque=(effective_zero*np.array([200,200,160]*2)).tolist()))
    cs=np.array(cs);qs=np.array(qs);cs[0]=command;qs[0]=q
    assert abs(np.diff(cs,axis=0)).max()*24/7<=.150001
    audit=dict(raw_normal=rn.tolist(),raw_offset=rd,normal=n.tolist(),offset=d,origin=origin.tolist(),pitch=float(Rotation.from_matrix(R).as_euler('xyz')[1]),
        gap_initial=checks[0]['gap'],root_path=roots,static_path=statics,target_root=roots[-1],chosen_pitch=.9,contact_gap=.001,ik_error=error,
        first_command_exact=True,landing_config=config,landing_repair_applied=True,positive_candidate_applied=True,
        load_feedback_basis=dict(zero=zero,full=full,max_share=shares,scheduled=zero),
        joint_path_17=dict(start_s=8.,duration_s=7.,end_s=15.,screen=screen,fractions=fractions,
            runtime_inputs=dict(q=q.tolist(),gravity=np.asarray(gravity).tolist(),geometry=np.asarray(geometry).tolist(),reference=np.asarray(reference).tolist(),command=command.tolist())),
        certification='None. Nominal model/rate screen only; physical validation required. Selected approach changes after8s.')
    return cs,qs,audit

def install(harness,config):
    normal.install(harness,config);bridge=harness.parent_harness.traction.bridge
    original=bridge.BridgeEnv.reference
    src=textwrap.dedent(inspect.getsource(original))
    assert src.count('u=(elapsed/5.*24).clamp(0,24);')==1 and src.count('*24/5.*(u<24)')==1
    # Preserve the exact scalar arithmetic on all unselected rows. Replacing
    # scalar division by a tensor divisor could introduce avoidable CUDA ULPs.
    src=src.replace('u=(elapsed/5.*24).clamp(0,24);',"u=(elapsed/5.*24).clamp(0,24)\n    if hasattr(self,'repair_duration'):u=torch.where(self.repair_duration>5.,(elapsed/7.*24).clamp(0,24),u)\n    ")
    src=src.replace('*24/5.*(u<24)',"*24/5.*(u<24)\n    if hasattr(self,'repair_duration'):velocity=torch.where(self.repair_duration>5.,(self.target_paths[ids,lo+1,1]-self.target_paths[ids,lo,1])*24/7.*(u<24),velocity)")
    scope=dict(original.__globals__);exec(src,scope);bridge.BridgeEnv.reference=scope['reference']
    Parent=harness.FeedbackEnv
    class JointPathEnv(Parent):
        def __init__(self,*args,**kwargs):
            self.joint_path_ready=False
            super().__init__(*args,**kwargs)
            self.repair_duration=torch.full((self.num_envs,),5.,device=self.device)
            self.joint_path_ready=True
        def _reset_rows(self,ids):
            if self.joint_path_ready:self.repair_duration[ids]=5.
            return super()._reset_rows(ids)
        def _observe(self,force=False):
            super()._observe(force)
            if not self.joint_path_ready or not self.planning_enabled:return
            pending=(self.near_start<0)&(self.warming==0)&(self.motion_time()>=8)&(self.motion_time()<8.1)
            q=self.robot.data.joint_pos.torch.cpu().numpy();gravity=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            for i in pending.nonzero().flatten().tolist():
                if not self.geometry[i,11] or self.geometry[i,12]>.5:continue
                origin=self.reset_library['root_pose'][self.index[i],int(self.blend[i])].copy();origin[0]=float(self.estimated_root_x[i])
                plane=measured(self.model,q[i],gravity[i],self.geometry[i],origin,self.calibration)
                if not eligible(plane):continue
                c,qs,a=plan(self.model,q[i],gravity[i],self.geometry[i],origin,self.previous[i].cpu().numpy(),self.calibration,bridge.inertial,config)
                self.near_paths[i]=torch.tensor(c,device=self.device,dtype=torch.float32);self.near_q[i]=torch.tensor(qs,device=self.device,dtype=torch.float32)
                self.near_start[i]=self.episode_length_buf[i];self.repair_duration[i]=7.
                self.target_paths[i]=torch.tensor([[r[0],Rotation.from_quat(r[3:]).as_euler('xyz')[1]] for r in a['root_path']],device=self.device);self.target_ready[i]=True
                self.plan_audit.append(dict(case=i,attempt=int(self.attempt[i]),tick=int(self.episode_length_buf[i]),stage=int(self.blend[i]),**a))
            self._augment()
        def close(self):
            super().close()
            source=Path(__file__).parent
            names=[Path(__file__).name,'low_seat_normal_duration_17.py','low_seat_load_duration_17.py','low_seat_contact_duration_17.py']
            for name in names:shutil.copy2(source/name,self.out/'source'/name)
            (self.out/'joint_path_contract_17.json').write_text(json.dumps(dict(locked_prefix_seconds=8.,unselected_reference_unchanged=True,
                selector='measuredheight<38cm and14<slope<18deg',duration_s=7.,no_optimizer=True,unresolved_deep_case='36cm/+20deg unchanged',
                bridge_reference_original_sha256=hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),bridge_reference_replacement=src,
                sources={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}),indent=2))
    harness.FeedbackEnv=JointPathEnv

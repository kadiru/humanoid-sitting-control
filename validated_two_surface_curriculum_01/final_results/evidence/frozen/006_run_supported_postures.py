"""Independent physical validation of arbitrary-angle supported G1 postures."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import run_resting_posture as base
from humanoid_sitting.resting_objective import resting_objective,RestingObjectiveCfg
from humanoid_sitting.supported_surface import observed_support

CONTACTS=[('pelvis_seat','pelvis',True),('left_thigh_seat','left_hip_yaw_link',True),
          ('right_thigh_seat','right_hip_yaw_link',True),('torso_surface','torso_link',True),('back_cover_surface','logo_link',True),
          ('rest_left_floor','left_ankle_roll_link',False),('rest_right_floor','right_ankle_roll_link',False)]
REAR_CAMERA=np.eye(4);REAR_CAMERA[:3,:3]=np.array([[0.,.707106781,-.707106781],[1.,0.,0.],[0.,-.707106781,-.707106781]])
REAR_CAMERA[:3,3]=[-.07,0.,.38]

def write(path,data):base.write_json(path,data)

def reset_surface(env,env_ids):
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.arange(env.num_envs,device=env.device)[env_ids] if isinstance(env_ids,slice) else env_ids
    gs=[env.cfg.geometry_bank[i] for i in base.furniture.indices(env)[ids].cpu().tolist()]
    pos=np.array([g['top_center'] for g in gs])-.05*np.array([g['normal'] for g in gs])
    pose=torch.tensor(np.c_[pos,[g['rotation_xyzw'] for g in gs]],device=env.device,dtype=torch.float32)
    pose[:,:3]+=env.scene.env_origins[ids]
    env.scene['seat'].write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
    env.scene['seat'].write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids),6,device=env.device),env_ids=ids)

def make_cfg(args,geometry):
    cfg=base.make_cfg(args,geometry)
    cfg.events.reset_seats.func=reset_surface
    for sensor,body,rest in CONTACTS:
        setattr(cfg.scene,sensor,base.ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot/{body}',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Seat' if rest else '{ENV_REGEX_NS}/RestFloor'],
            history_length=4,track_friction_forces=True,track_contact_points=True,max_contact_data_count_per_prim=32))
    if getattr(args,'depth_check',False):
        cfg.scene.rear_camera=base.CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/RearSupportCamera',width=128,height=96,
            data_types=['distance_to_image_plane'],update_period=1.e6,
            offset=base.CameraCfg.OffsetCfg(pos=tuple(REAR_CAMERA[:3,3]),rot=tuple(Rotation.from_matrix(REAR_CAMERA[:3,:3]).as_quat()),convention='ros'),
            spawn=base.sim.PinholeCameraCfg(focal_length=15.,horizontal_aperture=26.,clipping_range=(.05,4.)))
    else:cfg.scene.rest_camera=None
    return cfg

def metrics(raw):
    r=raw.scene['robot'];d=r.data;n=raw.num_envs;device=raw.device
    indices=base.furniture.indices(raw).cpu().tolist();gs=[raw.cfg.geometry_bank[i] for i in indices]
    R=torch.tensor(np.array([Rotation.from_quat(g['rotation_xyzw']).as_matrix() for g in gs]),device=device,dtype=torch.float32)
    top=torch.tensor([g['top_center'] for g in gs],device=device)+raw.scene.env_origins
    extent=torch.tensor([[g['depth'],g['width']] for g in gs],device=device)
    force=[];positions=[];velocities=[];wrong=torch.zeros(n,device=device)
    for sensor,body,rest in CONTACTS:
        s=raw.scene[sensor].data;j=r.body_names.index(body)
        normal=s.normal_force_matrix_w.torch.sum((1,2));f=normal+s.friction_force_matrix_w.torch.sum((1,2));force.append(f)
        cp=s.contact_pos_w.torch[:,0,0];cp=torch.where(torch.isfinite(cp),cp,d.body_com_pos_w.torch[:,j]);positions.append(cp)
        v=d.body_com_vel_w.torch[:,j];velocities.append(v[:,:3]+torch.cross(v[:,3:],cp-d.body_com_pos_w.torch[:,j],dim=-1))
        wrong=torch.maximum(wrong,(s.net_normal_forces_w.torch.sum(1)-normal).norm(dim=-1))
    force=torch.stack(force,1);position=torch.stack(positions,1);velocity=torch.stack(velocities,1)
    normals=torch.zeros_like(force);normals[:,:5]=R[:,None,:,2];normals[:,5:,2]=1.
    local=torch.einsum('nki,nij->nkj',position-top[:,None,:],R)
    inside=torch.ones(n,7,dtype=torch.bool,device=device)
    inside[:,:5]=(local[:,:5,0].abs()<extent[:,None,0]/2-.01)&(local[:,:5,1].abs()<extent[:,None,1]/2-.01)&(local[:,:5,2].abs()<.015)
    # Torso support is legal only on the rear trunk, not the chest or side.
    torso=r.body_names.index('torso_link');body_R=torch.tensor(Rotation.from_quat(d.body_quat_w.torch[:,torso].cpu().numpy()).as_matrix(),device=device,dtype=torch.float32)
    for patch in (3,4):
        trunk_local=torch.einsum('ni,nij->nj',position[:,patch]-d.body_pos_w.torch[:,torso],body_R)
        inside[:,patch] &=trunk_local[:,0]<-.035
    slip=(velocity-(velocity*normals).sum(-1,keepdim=True)*normals).norm(dim=-1)
    contacts=raw.scene['all_contacts'];known={body for _,body,_ in CONTACTS}
    bad=[i for i,b in enumerate(contacts.body_names) if b not in known]
    unexpected=contacts.data.net_normal_forces_w_history.torch[:,:,bad].norm(dim=-1).flatten(1).max(1).values
    wrong=torch.maximum(wrong,unexpected)
    ids=[r.joint_names.index(x) for x in base.BODY_JOINT_NAMES]
    lim=d.joint_pos_limits.torch[:,ids];q=d.joint_pos.torch[:,ids]
    margin=((q>lim[...,0]+.005)&(q<lim[...,1]-.005)).all(1)
    weight=d.body_mass.torch.sum(1)*9.81
    peak=torch.stack([raw.scene[s].data.normal_force_matrix_w_history.torch.sum((2,3)).norm(dim=-1).max(1).values for s,_,rest in CONTACTS if rest],1).sum(1)
    tilt=torch.acos((-d.projected_gravity_b.torch[:,2]).clamp(-1,1));rootz=d.root_pos_w.torch[:,2]-raw.scene.env_origins[:,2]
    outside=((force.norm(dim=-1)>.01*weight[:,None])&~inside).any(1)
    flags=dict(fall=(rootz<.28)|(tilt>np.deg2rad(55)),forbidden=(wrong>10)|outside,
        torque=(d.applied_torque.torch.abs()>d.joint_effort_limits.torch+.001).any(1),impact=peak>2*weight,joint_limit=~margin)
    failure=torch.stack(list(flags.values())).any(0)
    m=dict(torque=d.applied_torque.torch,torque_scale=d.joint_effort_limits.torch,joint_velocity=d.joint_vel.torch,
        force_w=force,normal_w=normals,friction=torch.full((n,7),.6,device=device),permitted=torch.ones(n,7,dtype=torch.bool,device=device),
        rest_patch=torch.tensor([[True]*5+[False]*2],device=device).expand(n,-1),inside_patch=inside,slip_m_s=slip,
        bodyweight_N=weight,speed_m_s=d.root_lin_vel_w.torch.norm(dim=-1),angular_speed_rad_s=d.root_ang_vel_w.torch.norm(dim=-1),
        trunk_tilt_rad=tilt,joint_margin_ok=margin,failure=failure)
    _,good,detail=resting_objective(m,torch.ones(n,dtype=torch.bool,device=device))
    detail.update(good=good,wrong_force_N=wrong,peak_force_N=peak,max_slip_m_s=slip.max(1).values,
        foot_load_bw=force[:,5:,2].sum(1)/weight,torso_load_bw=(force[:,3:5]*normals[:,3:5]).sum((1,2))/weight,
        root_height_m=rootz,tilt_deg=torch.rad2deg(tilt))
    return m,detail,flags

def restore(raw,bank,ids=None):
    r=raw.scene['robot'];ids=torch.arange(raw.num_envs,device=raw.device) if ids is None else ids
    tensor=lambda key:torch.as_tensor(bank[key],device=raw.device,dtype=torch.float32)[ids]
    root=tensor('root_pose').clone();root[:,:3]+=raw.scene.env_origins[ids]
    r.reset(ids);r.write_root_pose_to_sim_index(root_pose=root,env_ids=ids)
    r.write_root_velocity_to_sim_index(root_velocity=tensor('root_velocity'),env_ids=ids)
    r.write_joint_state_to_sim_index(position=tensor('q'),velocity=tensor('qd'),env_ids=ids)
    r.set_joint_position_target(tensor('command'),env_ids=ids)
    for sensor in raw.scene.sensors.values():sensor.reset(ids)
    raw.scene.write_data_to_sim()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--bank',required=True)
    p.add_argument('--seconds',type=float,default=24.);p.add_argument('--settle',type=float,default=4.)
    p.add_argument('--depth-check',action='store_true');p.add_argument('--seed',type=int,default=54101)
    base.add_launcher_args(p);args=p.parse_args();args.mode='fixed';args.enable_cameras=args.depth_check
    out=Path(args.output);out.mkdir(exist_ok=False);(out/'source').mkdir()
    paths=[Path(__file__),Path(base.__file__),base.ROOT/'sitting_task/src/humanoid_sitting/supported_surface.py',base.MODEL]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for f in paths:shutil.copy2(f,out/'source'/f.name)
    write(out/'manifest.json',dict(args=vars(args),sources=hashes,dt=.02,label_contract='oriented plane normals/foot floor; pelvis/thigh or rear torso support; unchanged physical limits'))
    geometry=json.loads((Path(args.bank)/'geometry.json').read_text());bank=dict(np.load(Path(args.bank)/'candidates.npz'))
    args.num_envs=len(geometry);cfg=make_cfg(args,geometry)
    write(out/'geometry.json',geometry)
    with base.launch_simulation(cfg,args):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped;raw.reset();restore(raw,bank)
        r=raw.scene['robot'];command=torch.tensor(bank['command'],device=raw.device,dtype=torch.float32)
        good_ticks=torch.zeros(raw.num_envs,device=raw.device,dtype=torch.long);held=good_ticks.clone();flags_all={};traces=[];first=[]
        active=0;depth_records=[];model=base.BodyModel(base.MODEL);seen_failure=set()
        with torch.inference_mode():
            for step in range(round(args.seconds/.02)):
                raw.step(command);assert not raw.reset_buf.any();m,d,flags=metrics(raw)
                if step>=round(args.settle/.02):
                    active+=1;good=d['good'];good_ticks+=good;held=torch.where(good,held+1,torch.zeros_like(held))
                    for key,value in flags.items():
                        if key not in flags_all:flags_all[key]=torch.zeros_like(value)
                        flags_all[key]|=value
                    for i in m['failure'].nonzero().flatten().tolist():
                        if i in seen_failure:continue
                        seen_failure.add(i);contact=raw.scene['all_contacts']
                        peak=contact.data.net_normal_forces_w_history.torch[i].norm(dim=-1).max(0).values
                        first.append(dict(case=i,t=(step+1)*.02,flags={k:bool(v[i]) for k,v in flags.items()},
                            bodies=[dict(body=name,normal_N=float(peak[j])) for j,name in enumerate(contact.body_names) if peak[j]>1],
                            filtered=[dict(body=body,force=m['force_w'][i,j].cpu().tolist(),inside=bool(m['inside_patch'][i,j]),
                                contact=raw.scene[sensor].data.contact_pos_w.torch[i,0,0].cpu().tolist()) for j,(sensor,body,_) in enumerate(CONTACTS)]))
                if step%5==4:
                    traces.append(dict(t=(step+1)*.02,body=r.data.body_pose_w.torch.cpu().numpy().copy(),q=r.data.joint_pos.torch.cpu().numpy().copy(),
                        torque=r.data.applied_torque.torch.cpu().numpy().copy(),good=d['good'].cpu().numpy(),
                        support=d['rest_normal_load_bw'].cpu().numpy(),torso=d['torso_load_bw'].cpu().numpy(),foot=d['foot_load_bw'].cpu().numpy(),
                        wrong=d['wrong_force_N'].cpu().numpy(),slip=d['max_slip_m_s'].cpu().numpy()))
                if args.depth_check and step%50==49:
                    clock=raw.sim.get_physics_step_count();raw.sim.render_context.update_scene_state(clock)
                    for cam_name,T in [('rest_camera',base.T_CAMERA),('rear_camera',REAR_CAMERA)]:
                        cam=raw.scene[cam_name];base.acquire_current_camera(cam);assert raw.sim.get_physics_step_count()==clock
                        depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy();K=cam.data.intrinsic_matrices.torch.cpu().numpy()
                        for i in range(raw.num_envs):
                            measured,diagnostic=observed_support(model,K[i],T,depth[i],r.data.joint_pos.torch[i].cpu().numpy(),r.data.projected_gravity_b.torch[i].cpu().numpy())
                            row=dict(t=(step+1)*.02,case=i,camera=cam_name,valid=measured is not None,diagnostic=diagnostic)
                            if measured is not None:
                                bodyR=Rotation.from_quat(r.data.root_quat_w.torch[i].cpu().numpy()).as_matrix()
                                normal=bodyR.T@np.array(geometry[i]['normal'])
                                top=np.array(geometry[i]['top_center'])+raw.scene.env_origins[i].cpu().numpy()
                                offset=(r.data.root_pos_w.torch[i].cpu().numpy()-top)@np.array(geometry[i]['normal'])
                                row.update(values=measured.tolist(),normal_error_deg=float(np.rad2deg(np.arccos(np.clip(measured[2:5]@normal,-1,1)))),offset_error_m=float(measured[5]-offset))
                            depth_records.append(row)
                if step%100==99:print('SECONDS',(step+1)*.02,'good',int(d['good'].sum()),flush=True)
            bad=torch.stack(list(flags_all.values())).any(0)
            valid=(good_ticks>=.95*active)&(held>=100)&~bad
            root=r.data.root_pose_w.torch.clone();root[:,:3]-=raw.scene.env_origins
            np.savez_compressed(out/'candidates.npz',root_pose=root.cpu().numpy(),root_velocity=r.data.root_vel_w.torch.cpu().numpy(),
                q=r.data.joint_pos.torch.cpu().numpy(),qd=r.data.joint_vel.torch.cpu().numpy(),command=command.cpu().numpy(),valid=valid.cpu().numpy())
            rows=[dict(case=i,geometry=geometry[i],valid=bool(valid[i]),steps=active,good_fraction=float(good_ticks[i])/active,
                final_hold_seconds=float(held[i])*.02,failures={k:bool(v[i]) for k,v in flags_all.items()},
                final_metrics={k:float(v[i]) for k,v in d.items()}) for i in range(raw.num_envs)]
            write(out/'results.json',dict(valid_count=int(valid.sum()),total=raw.num_envs,rows=rows,joint_names=r.joint_names,body_names=r.body_names,
                origins=raw.scene.env_origins.cpu().numpy().tolist()))
            write(out/'depth_validation.json',depth_records)
            write(out/'first_failures.json',first)
            np.savez_compressed(out/'trace.npz',**{k:np.stack([v[k] for v in traces]) for k in traces[0]})
            print('RESULT',int(valid.sum()),raw.num_envs,flush=True)
        raw.close()

if __name__=='__main__':main()

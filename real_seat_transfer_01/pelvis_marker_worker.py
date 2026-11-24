from pathlib import Path
import sys
import numpy as np

def install(original,profile,launcher,options):
    sys.path[:0]=[str(original.parent),str(launcher.parent),str(launcher.parent/'trainer')]
    outdir=Path(options[options.index('--output')+1])
    import seated_arm_posture
    import grip_hold_ppo
    capture=grip_hold_ppo.capture_metrics
    def connect(support):
        import torch
        from scipy.spatial.transform import Rotation
        policy_state=capture(support)
        old=support.metrics
        maps=[]
        captured=set()
        history=[]
        counts_total=[0,0]
        def metrics(raw):
            m,d,flags=old(raw)
            if not maps:
                from surface import Seat
                maps.extend(Seat(g) for g in raw.cfg.geometry_bank)
            # Preserve the policy's original handoff inputs; the mesh lookup is an evaluator.
            m=dict(m); m['inside_patch']=m['inside_patch'].clone();m['normal_w']=m['normal_w'].clone();m['slip_m_s']=m['slip_m_s'].clone()
            robot=raw.scene['robot']; origins=raw.scene.env_origins.cpu().numpy()
            active=m['force_w'].norm(dim=-1)>.01*m['bodyweight_N'][:,None]
            report=np.full((raw.num_envs,5,10),np.nan,np.float32)
            for j,(sensor,body,rest) in enumerate(support.CONTACTS[:5]):
                eligible=np.flatnonzero(active[:,j].cpu().numpy())
                if not len(eligible):continue
                s=raw.scene[sensor]
                data=s.data
                net=data.normal_force_matrix_w.torch[:,0,0].detach().cpu().numpy().copy()
                vals=s.contact_view.get_contact_data(dt=.005)
                def arr(v):
                    return v.numpy() if hasattr(v,'numpy') else np.asarray(v)
                counts=arr(vals[4]).reshape(-1);starts=arr(vals[5]).reshape(-1)
                positions=arr(vals[1]).reshape(-1,3);norms=arr(vals[2]).reshape(-1,3)
                b=robot.body_names.index(body);cp=s.data.contact_pos_w.torch[:,0,0]
                for i in eligible:
                    count=int(counts[i]);k=int(starts[i]);ps=positions[k:k+count]-origins[i]
                    force=net[i]
                    w=np.abs(arr(vals[0]).reshape(-1)[k:k+count])
                    finite=np.isfinite(w)&np.isfinite(ps).all(1)&np.isfinite(norms[k:k+count]).all(1)&(w>1.e-7)
                    w=w[finite];points=ps[finite];unit=norms[k:k+count][finite]
                    total=float(w.sum());denom=float(np.linalg.norm(force))
                    summed=(w[:,None]*unit).sum(0)
                    sign=1. if np.dot(summed,force)>=0 else -1.
                    error=float(np.linalg.norm(sign*summed-force))
                    bw=float(m['bodyweight_N'][i])
                    # No attribution when the manifold does not reproduce the filtered force.
                    # Numerical tolerance only applies to checking the two sensor readouts.
                    resolved=bool(total>0 and error<=.02+.0003*max(total,denom))
                    centerpoint=cp[i].detach().cpu().numpy()-origins[i]
                    if resolved:
                        centerpoint=(points*w[:,None]).sum(0)/total
                    ok,loc,margin,dz=maps[i].lookup(centerpoint[None])
                    good_points,point_normals,_,_=maps[i].lookup(points)
                    if len(w):
                        aligned_points=(unit*sign*point_normals).sum(-1)>=np.cos(np.deg2rad(35))
                        off=float(w[~(good_points&aligned_points)].sum())
                    else:off=0.
                    local=loc
                    upward=force
                    aligned=(float(np.dot(force,loc[0]))/max(denom,1e-9)) if denom>0 else 0.
                    valid=bool(resolved and ok[0] and aligned>=np.cos(np.deg2rad(35)) and off<=.01*bw)
                    report[i,j]=[*centerpoint,error,denom,total,off,bw,float(valid),float(resolved)]
                    counts_total[0]+=1
                    counts_total[1]+=int(resolved)
                    ps=centerpoint[None]
                    key=(i,j)
                    if key not in captured:
                        import json
                        detail=dict(row=int(i),channel=j,asset=maps[i].g['name'],count=count,cp=centerpoint.tolist(),
                            pts=(positions[k:k+count]-origins[i]).tolist(),normals=norms[k:k+count].tolist(),normal_force=upward.tolist(),
                            old_inside=bool(m['inside_patch'][i,j]),mapper_inside=valid,margin=float(margin[0]),height_error=float(dz[0]),dot=float(aligned))
                        with (outdir/'first_contacts.jsonl').open('a') as f:
                            f.write(json.dumps(detail)+'\n')
                        captured.add(key)
                    if valid and j in (3,4):
                        t=robot.body_names.index('torso_link')
                        rot=Rotation.from_quat(robot.data.body_quat_w.torch[i,t].cpu().numpy()).as_matrix()
                        valid=bool((((ps+origins[i]-robot.data.body_pos_w.torch[i,t].cpu().numpy())@rot)[:,0]<-.035).all())
                    m['inside_patch'][i,j]=valid
                    if valid:
                        normal=m['normal_w'].new_tensor(local.mean(0));normal/=normal.norm()
                        m['normal_w'][i,j]=normal
                        center=robot.data.body_com_pos_w.torch[i,b]; vel=robot.data.body_com_vel_w.torch[i,b]
                        point=center.new_tensor(centerpoint+origins[i])
                        v=vel[:3]+torch.cross(vel[3:],point-center,dim=0)
                        m['slip_m_s'][i,j]=(v-(v*normal).sum()*normal).norm()
            if not hasattr(metrics,'floor_trace'):metrics.floor_trace=[]
            floor_log=np.full((raw.num_envs,2,7),np.nan,np.float32)
            for j in (5,6):
                sensor,body,_=support.CONTACTS[j]
                eligible=np.flatnonzero(active[:,j].cpu().numpy())
                if not len(eligible):continue
                s=raw.scene[sensor];data=s.data
                net=data.normal_force_matrix_w.torch[:,0,0].detach().cpu().numpy().copy()
                vals=s.contact_view.get_contact_data(dt=.005)
                def arr(v):return v.numpy() if hasattr(v,'numpy') else np.asarray(v)
                counts=arr(vals[4]).reshape(-1);starts=arr(vals[5]).reshape(-1)
                b=robot.body_names.index(body);center=robot.data.body_com_pos_w.torch[:,b].cpu().numpy();vel=robot.data.body_com_vel_w.torch[:,b].cpu().numpy()
                for i in eligible:
                    k=int(starts[i]);count=int(counts[i]);sl=slice(k,k+count)
                    f=np.abs(arr(vals[0]).reshape(-1)[sl]);points=arr(vals[1]).reshape(-1,3)[sl];norms=arr(vals[2]).reshape(-1,3)[sl]
                    use=np.isfinite(f)&np.isfinite(points).all(1)&np.isfinite(norms).all(1)&(f>1e-7)
                    f=f[use];points=points[use];norms=norms[use]
                    bw=float(m['bodyweight_N'][i]);old_speed=float(m['slip_m_s'][i,j])
                    vector=(f[:,None]*norms).sum(0);error=min(np.linalg.norm(vector-net[i]),np.linalg.norm(vector+net[i]))
                    if not len(f) or error>.02+.0003*max(float(f.sum()),np.linalg.norm(net[i])):
                        floor_log[i,j-5]=[0,error,old_speed,old_speed,old_speed,0,bw]
                        continue
                    normal=m['normal_w'][i,j].cpu().numpy();normal=normal/np.linalg.norm(normal)
                    pv=vel[i,:3]+np.cross(vel[i,3:],points-center[i]);tang=pv-(pv@normal)[:,None]*normal
                    toolspeed=np.linalg.norm(tang,axis=1);order=np.argsort(-toolspeed)
                    # Keep the fastest set of points carrying an aggregate of at least one
                    # percent of body weight; also include the velocity at the force center.
                    cut=np.searchsorted(np.cumsum(f[order]),.01*bw,side='right')
                    point_speed=toolspeed[order[min(cut,len(order)-1)]]
                    mean_speed=float(np.linalg.norm((tang*f[:,None]).sum(0)/f.sum()))
                    speed=max(point_speed,mean_speed)
                    floor_log[i,j-5]=[1,error,old_speed,mean_speed,speed,f[toolspeed>.02].sum(),bw]
                    m['slip_m_s'][i,j]=float(speed)
            metrics.floor_trace.append(floor_log)
            if len(metrics.floor_trace) in (200,400,600,800,1000,1025):
                np.savez_compressed(outdir/'floor_slip_witness.npz',measurements=np.stack(metrics.floor_trace),
                    columns=['conserved','force_error_N','old_speed_m_s','center_speed_m_s','guard_speed_m_s','fast_point_load_N','bodyweight_N'])
            history.append(report)
            if len(history) in (200,400,600,800,1000,1025):
                np.savez_compressed(outdir/'contact_load_geometry.npz',observations=np.stack(history),columns=['x','y','z','force_error_N','normal_net_N','sum_normal_N','outside_load_N','bw_N','inside','force_checked'])
            flags=dict(flags); flags['forbidden']=(d['wrong_force_N']>10)|(active&~m['inside_patch']).any(1)
            m['failure']=torch.stack(list(flags.values())).any(0)
            _,good,new=support.resting_objective(m,torch.ones(raw.num_envs,dtype=torch.bool,device=raw.device))
            d=dict(d,**{})|new;d['good']=good
            return m,d,flags
        support.metrics=metrics
        cfg=support.make_cfg
        def with_buffers(args,geometry):
            config=cfg(args,geometry)
            for key in [entry[0] for entry in support.CONTACTS]+['seat_probe_'+name for name in getattr(config,'armchair_probe_links',())]:
                sensor=getattr(config.scene,key)
                sensor.max_contact_data_count_per_prim=max(256,sensor.max_contact_data_count_per_prim or 0)
            return config
        support.make_cfg=with_buffers
        return policy_state
    grip_hold_ppo.capture_metrics=connect
    import torch
    import evaluate_armchair_pair as pair
    import json
    from chair_contact_motion import ChairContactMotion
    attach=pair.trace_contacts
    settings=json.loads(Path(__file__).with_name('folded_arms.json').read_text())['arms']
    ids=settings['left']['joints']+settings['right']['joints']
    values=settings['left']['angles']+settings['right']['angles']
    def patch(launcher,options):
        original_apply=ChairContactMotion.apply
        def apply(motion,learner,obs,t,nominal):
            before=original_apply(motion,learner,obs,t,nominal)
            env=learner.env
            if not hasattr(motion,'arm_handoff'):
                motion.arm_handoff=before.new_zeros(env.num_envs)
                motion.front_log=[]
            if t<40:return before
            lap=getattr(env,'supported_lap',None)
            if lap is not None:
                motion.arm_handoff=torch.clamp(motion.arm_handoff+.02*lap.started.float(),max=1.)
            entry=float(np.clip((t-40)/75.,0.,1.))
            weight=(entry*(1-motion.arm_handoff))[:,None]
            out=before.clone();out[:,ids]=torch.lerp(before[:,ids],out.new_tensor(values).expand(env.num_envs,-1),weight)
            if t==40:
                from route_depth import identify
                db=np.load(Path(__file__).with_name('confirmed_routes.npz'))
                picked=[identify([s[1][i] for s in env.standing_samples],db['reference'],db['methods']) for i in range(len(obs))]
                motion.method=torch.tensor([x[0] for x in picked],device=before.device)
                (env.out/'camera_method.json').write_text(json.dumps(picked))
                from depth_target import support_patch
                lib=np.load(Path(__file__).with_name('height_library.npz'))
                hs=lib['height']; height=[]; valid=[]; active=[]
                for i in range(len(obs)):
                    scan=[support_patch(sample[1][i],sample[2][i],sample[3][i],sample[4][i],sample[8],env.model,learner.contact_feedback)
                          for sample in env.standing_samples]
                    found=[m[0] for m in scan if m is not None]
                    h=float(np.median(found)) if found else 0.
                    valid.append(len(found));height.append(h)
                    active.append(len(found)>=3 and np.ptp(found)<.035 and hs[0]-.01<=h<=hs[-1]+.01)
                hh=np.clip(height,hs[0],hs[-1]);idx=np.clip(np.searchsorted(hs,hh)-1,0,len(hs)-2)
                ratio=((hh-hs[idx])/(hs[idx+1]-hs[idx]))[:,None,None]
                from pelvis_marker_motion import StanceTransfer
                motion.stance=StanceTransfer(picked,env,before)
                motion.height_paths=before.new_tensor((1-ratio)*lib['desired'][idx]+ratio*lib['desired'][idx+1])
                motion.leg_ids=lib['legs'].tolist()
                motion.use_height=torch.as_tensor(active,device=obs.device)
                acquired=Path(__file__).with_name('feedback_donors.npz')
                data=np.load(acquired);motion.donor_rows=np.flatnonzero(np.array([p[0] for p in picked])>=3)
                if len(motion.donor_rows):
                    keys=[picked[j][0]-3 for j in motion.donor_rows]
                    motion.dem_q=before.new_tensor(data['q'][keys]);motion.dem_v=before.new_tensor(data['v'][keys]);motion.dem_cmd=before.new_tensor(data['desired'][keys])
                    motion.dem_latch=data['latch'][keys];motion.dem_delta=torch.zeros_like(before[motion.donor_rows])
                    motion.dem_arms=np.array(ids)
                motion.clocks=np.full(len(obs),float(t))
                motion.height_log=[]
                (env.out/'seen_seat.json').write_text(json.dumps(dict(estimated_height_m=height,usable_frames=valid,used=active),default=lambda x:x.item()))
            phase=motion.clocks
            ii=np.floor(phase).astype(int)
            envrows=np.arange(len(obs))
            v0=motion.height_paths[envrows,ii];v1=motion.height_paths[envrows,np.minimum(ii+1,999)]
            path=torch.lerp(v0,v1,before.new_tensor(phase-ii)[:,None])
            blend=min(1.,(t-40)/80.)
            joints=motion.leg_ids
            out[:,joints]=torch.where((motion.use_height & (motion.method==2))[:,None],torch.lerp(out[:,joints],path[:,joints],blend),out[:,joints])
            bw=env.robot.data.body_mass.torch.sum(1)*9.81
            seat=env.raw.scene['pelvis_seat'].data.normal_force_matrix_w.torch.sum((1,2))[:,2]/bw
            load=seat.detach().cpu().numpy()
            # Give the feet time to unload while the demonstrated transfer continues.
            rate=np.where((phase<700)&(load>.03),np.clip(1.-.8*load,.35,1.),1.)
            motion.height_log.append(np.column_stack((phase,load)))
            phase[:]=np.minimum(999,phase+rate)
            if t==999:np.savez_compressed(env.out/'retargeted_transfer.npz',phase_and_load=np.array(motion.height_log))
            motion.front_log.append(weight[:,0].cpu().numpy())
            if t==999:np.savez_compressed(env.out/'arm_clearance.npz',preload_blend=motion.front_log)
            result=torch.where((motion.method!=0)[:,None],out,before)
            rows=motion.donor_rows
            if len(rows):
                target=motion.dem_cmd[:,t].clone()
                for j,row in enumerate(rows):
                    start=int(motion.dem_latch[j])
                    if t>=start and (lap is None or not bool(lap.started[row])):
                        target[j,motion.dem_arms]=motion.dem_cmd[j,max(0,start-1),motion.dem_arms]
                measured_q=env.robot.data.joint_pos.torch[rows,:23];measured_v=env.robot.data.joint_vel.torch[rows,:23]
                gain=((seat[rows]-.03)/.13).clamp(0.,1.)[:,None]
                correction=(.75*(motion.dem_q[:,t-1]-measured_q)+.035*(motion.dem_v[:,t-1]-measured_v)).clamp(-.045,.045)*gain
                correction[:,motion.dem_arms]=0.
                motion.dem_delta+=torch.clamp(correction-motion.dem_delta,-.0015,.0015)
                result[rows]=target+motion.dem_delta
            return motion.stance.apply(env,t,result,load)
        ChairContactMotion.apply=apply
        attach(launcher,options)
    pair.trace_contacts=patch
    seated_arm_posture.install(launcher,options,profile)
if __name__=='__main__':
    assert sys.argv[2]=='--arm-worker'
    install(Path(sys.argv[1]),Path(sys.argv[3]),Path(sys.argv[4]),sys.argv[5:])






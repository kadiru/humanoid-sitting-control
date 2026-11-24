"""Refine the seated posture after load transfer with both soles as anchors."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from rescan_sitting import points, match, shoe_frame, smooth

def efficient_target(reference, model, sample, demo, setback, solver, robot):
    depth,k,joints,gravity,mount=sample
    observed=match(demo['cloud'],points(depth,k,mount,joints,gravity,model))
    if observed is None:
        return None
    match_se2,_=observed
    local,mid,basis=shoe_frame(model,joints,gravity)
    chair_axes=demo['frame']@Rotation.from_euler('z',match_se2[2]).as_matrix()
    shoe_origin=demo['origin']+demo['frame']@np.r_[match_se2[:2],0.]
    pelvis_R=chair_axes@basis.T
    landed=shoe_origin+(local[:,:3,3]-mid)@pelvis_R.T
    landed_R=pelvis_R@local[:,:3,:3]
    shoe_shift=landed-reference['start_feet'][:,:3]
    shoe_turn=landed_R@Rotation.from_quat(reference['start_feet'][:,3:]).as_matrix().transpose(0,2,1)
    roots=reference['hold_pose']
    rot=Rotation.from_quat(roots[0,3:]).as_matrix()
    base=roots[0,:3].copy()
    new_p=base+np.array([setback,0.,.001])
    oldq=reference['q']
    newq=oldq.copy()
    for side,(chain,ids) in enumerate(zip(solver.chains,solver.legs)):
        target=np.eye(4)
        target[:3,:3]=rot.T@shoe_turn[side]@Rotation.from_quat(roots[1+side,3:]).as_matrix()
        target[:3,3]=rot.T@(roots[1+side,:3]+shoe_shift[side]-new_p)
        newq[ids],error=solver.solve(chain,oldq[ids],target,ids)
        if error>.001:
            return None
    old=robot.state(oldq,rot,base)
    new=robot.state(newq,rot,new_p)
    oldcm=np.average(old[1],axis=0,weights=robot.mass)
    newcm=np.average(new[1],axis=0,weights=robot.mass)
    f=reference['floor_force']
    frac=f[:,2]/f[:,2].sum()
    old_cop=roots[1:,:3].copy()
    old_cop[:,2]=0.
    old_cop[:,:2]+=np.clip(oldcm[:2]-frac@old_cop[:,:2],[-.058,-.024],[.082,.024])
    new_cop=old_cop.copy()
    new_cop[:,:2]+=shoe_shift[:,:2]+np.clip((newcm-oldcm)[:2]-frac@shoe_shift[:,:2],-.035,.035)
    # The difference keeps the reference's measured static bias and changes the
    # gravity and floor-reaction moments induced by the staggered soles.
    new_torque=robot.forces(new,np.zeros(3),new_cop,f)-robot.forces(old,np.zeros(3),old_cop,f)
    kp=np.array([200,200,200,150,150,40,40,150,150,40,40,200,200,40,40,160,160,40,40,160,160,40,40])
    return reference['command']+newq[:23]-oldq[:23]+np.clip(new_torque[:23]/kp,-.10,.10)


def attach(config, inherited):
    from chair_contact_motion import ChairContactMotion
    chooser=ChairContactMotion.apply
    spec=config.get('rest_refinement')
    if not spec:
        return inherited()
    archive=Path(spec['inputs'])
    with np.load(archive,allow_pickle=False) as z:
        saved={k:z[k] for k in z.files}
    with np.load(config['rescan_library'],allow_pickle=False) as z:
        demo={k.removeprefix('Basic_'):z[k] for k in z.files if k.startswith('Basic_')}

    def apply(motion, learner, obs, tick, nominal):
        from rearstep_model import Robot
        import stance23_planner as solver
        cmd=chooser(motion,learner,obs,tick,nominal)
        env=learner.env
        if tick==0:
            motion.rest_refinements={}
            schedule=config['templates_by_row']
            for row in motion.rescan_paths:
                name=schedule[row]
                if name not in ('Anora','Basic'):
                    continue
                state=dict(name=name,checks=0,enable_tick=None,drop=0.,rows=[],target=None)
                if name=='Basic':
                    _,d,k,q,g,_,_,_,mount=env.standing_samples[-1]
                    goal=efficient_target(saved,env.model,(d[row],k[row],q[row],g[row],mount),demo,
                                          spec['basic_setback_m'],solver,Robot())
                    if goal is None:
                        continue
                    state['target']=cmd.new_tensor(goal)
                motion.rest_refinements[row]=state
        if tick<spec['start_tick']:
            return cmd
        d=env.robot.data
        for row,state in motion.rest_refinements.items():
            weight=float(d.body_mass.torch[row].sum())*9.81
            seat=env.raw.scene['pelvis_seat'].data.normal_force_matrix_w.torch[row].sum((0,1)).norm().item()/weight
            foot=np.array([(env.raw.scene['rest_'+side+'_floor'].data.normal_force_matrix_w.torch[row].sum((0,1))[2].item())/weight
                           for side in ('left','right')])
            gyro=float(d.root_ang_vel_b.torch[row].norm())
            threshold=.32 if state['name']=='Basic' else .052
            if seat>threshold and foot.min()>.12 and gyro<.16:
                state['checks']+=1
            else:
                state['checks']=0
            if state['enable_tick'] is None and state['checks']>=8:
                state['enable_tick']=tick
            if state['enable_tick'] is None:
                continue
            age=tick-state['enable_tick']
            if state['name']=='Basic':
                # Full transfer increased seat shear beyond the friction
                # gate. Keep a share of the original, verified posture.
                w=float(spec['basic_share']*smooth(age/spec['basic_blend_ticks']))
                cmd[row]=(1-w)*cmd[row]+w*state['target']
            else:
                w=float(smooth(age/spec['blend_ticks']))
                joints=d.joint_pos.torch[row].cpu().numpy()
                gravity=d.projected_gravity_b.torch[row].cpu().numpy()
                feet,_,basis=shoe_frame(env.model,joints,gravity)
                rotate=Rotation.from_euler('xy',w*np.asarray(spec['anora_roll_pitch_rad'])).as_matrix()
                # Rotate around the region already carrying the load, then
                # press into the surface by a bounded amount.
                pin=np.array([.0024,.026,-.1492])
                if age>28:
                    requested=.00155*(spec['anora_load_target']-seat)
                    state['drop']=float(np.clip(state['drop']+np.clip(requested,-.00007,.00007),
                                                0.,spec['anora_drop_limit_m']))
                pose=basis@rotate@basis.T
                down=-basis[:,2]*state['drop']
                shift=pin-pose@pin+down
                for side,(chain,ids) in enumerate(zip(solver.chains,solver.legs)):
                    goal=np.eye(4)
                    goal[:3,:3]=pose.T@feet[side,:3,:3]
                    goal[:3,3]=pose.T@(feet[side,:3,3]-shift)
                    q_new,error=solver.solve(chain,joints[ids].copy(),goal,ids)
                    if error<.001:
                        cmd[row,ids]+=cmd.new_tensor(np.clip(q_new-joints[ids],-.09,.09))
            state['rows'].append([tick,seat,*foot,gyro,w,state['drop']])
        if tick==999:
            np.savez_compressed(env.out/'rest_refinement.npz',
                                **{str(row):np.asarray(r['rows']).reshape(-1,7) for row,r in motion.rest_refinements.items()})
        return cmd

    ChairContactMotion.apply=apply
    return inherited()

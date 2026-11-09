"""One finite GPU batch: 3 physical presses + 1 empty-space control per layout.

Diagnostic spheres are NOT the robot and receive no sitting success score.
Geometry is loaded from the exact same USD used by the robot test adapter.
"""
import argparse
import time
import numpy as np
import torch
from io_utils import ASSETS, HERE, OUT, read, write, sha
from surface_map import SurfaceMap
from contact_adapter import raw_points
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);add_launcher_args(p);a=p.parse_args()
    a.device='cuda:0'; out=OUT/'probes';assert str(out)==a.output;out.mkdir(exist_ok=False)
    cases=read(ASSETS/'cases.json');maps=[SurfaceMap(c) for c in cases]
    planned=read(HERE/'probe_points.json');assert len(cases)==14
    cfg=sim_utils.SimulationCfg(dt=.005,device='cuda:0',gravity=(0.,0.,0.))
    started=time.monotonic();records=[];probes=[];seats=[];sensors=[]
    origins=np.array([[0.,i*8.,0.] for i in range(14)])
    with launch_simulation(cfg,a):
        sim=SimulationContext(cfg)
        for i,c in enumerate(cases):
            root=f'/World/Test_{i:02d}'
            sim_utils.create_prim(root,'Xform')
            seats.append(RigidObject(RigidObjectCfg(prim_path=root+'/Seat',
                spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS/'scenes'/c['id']/'furniture.usda'),activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(origins[i])))))
            probes.append(RigidObject(RigidObjectCfg(prim_path=root+'/Probe',
                spawn=sim_utils.SphereCfg(radius=.025,rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),mass_props=sim_utils.MassCfg(mass=1.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.6),activate_contact_sensors=True),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(origins[i]+[1.,0.,2.])))))
            sensors.append(ContactSensor(ContactSensorCfg(prim_path=root+'/Probe',filter_prim_paths_expr=[root+'/Seat'],
                track_friction_forces=True,track_contact_points=True,history_length=4,max_contact_data_count_per_prim=128)))
        sim.reset();initial=sim.get_physics_step_count();assert initial<=1000
        from pxr import UsdGeom,UsdPhysics
        stage=sim_utils.get_current_stage();inventory=[]
        for i,c in enumerate(cases):
            root=stage.GetPrimAtPath(f'/World/Test_{i:02d}/Seat')
            from pxr import Usd
            colliders=[q for q in Usd.PrimRange(root) if q.HasAPI(UsdPhysics.CollisionAPI)]
            assert len(colliders)==1
            q=colliders[0];mesh=UsdGeom.Mesh(q);z=np.load(ASSETS/'scenes'/c['id']/'mesh.npz')
            np.testing.assert_allclose(np.asarray(mesh.GetPointsAttr().Get()),z['vertices'],rtol=1e-7,atol=1e-7)
            np.testing.assert_array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),z['faces'])
            assert UsdPhysics.CollisionAPI(q).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.MeshCollisionAPI(q).GetApproximationAttr().Get()=='none'
            pose=seats[i].data.root_pose_w.torch[0].cpu().numpy()
            np.testing.assert_allclose(pose,np.r_[origins[i],[0.,0.,0.,1.]],atol=1e-6)
            inventory.append(dict(case=c['id'],collider_path=str(q.GetPath()),triangle_count=len(z['faces']),
                approximation='none',root_pose=pose.tolist(),mesh_sha256=sha(ASSETS/'scenes'/c['id']/'furniture.usda')))
        write(out/'runtime_collision_inventory.json',inventory)
        summary=[]
        try:
            for round_id in range(4):
                targets=[];normals=[]
                for i,c in enumerate(cases):
                    if round_id<3:spec=planned[c['id']][round_id]
                    else:spec=dict(point=[c['bounds'][1][0]+.4,0.,c['bounds'][1][2]+.1],normal=[0.,0.,1.])
                    targets.append(np.array(spec['point'])+origins[i]);normals.append(np.array(spec['normal']))
                    pos=targets[-1]+.085*normals[-1]
                    probe=probes[i];probe.reset()
                    probe.write_root_pose_to_sim_index(root_pose=torch.tensor([np.r_[pos,[0.,0.,0.,1.]]],device='cuda:0',dtype=torch.float32))
                    probe.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device='cuda:0'))
                    sensors[i].reset()
                targets=np.array(targets);normals=np.array(normals);round_records=[]
                for tick in range(1200):
                    assert time.monotonic()-started<3600
                    t=(tick+1)*.005
                    if t<=.5:offset=.085
                    elif t<1.5:offset=.085-.1*(t-.5)
                    elif t<=4.5:offset=-.015
                    else:offset=min(.085,-.015+.1*(t-4.5))
                    forces=[];old_v=[]
                    for i,probe in enumerate(probes):
                        pos=probe.data.root_pos_w.torch[0];vel=probe.data.root_lin_vel_w.torch[0]
                        target=torch.as_tensor(targets[i]+offset*normals[i],device='cuda:0',dtype=torch.float32)
                        f=(250*(target-pos)-30*vel).clamp(-15,15)
                        forces.append(f.cpu().numpy());old_v.append(vel.cpu().numpy().copy())
                        probe.instantaneous_wrench_composer.add_forces_and_torques_index(forces=f[None,None,:],is_global=True)
                        probe.write_data_to_sim()
                    sim.step(render=False)
                    for obj in probes+seats:obj.update(.005)
                    for sensor in sensors:sensor.update(.005,force_recompute=True)
                    if tick%4==3:
                        frame=[]
                        for i,(probe,sensor) in enumerate(zip(probes,sensors)):
                            normal=sensor.data.normal_force_matrix_w.torch.sum((1,2))[0].cpu().numpy()
                            friction=sensor.data.friction_force_matrix_w.torch.sum((1,2))[0].cpu().numpy()
                            net=sensor.data.net_normal_forces_w.torch.sum(1)[0].cpu().numpy()
                            cp=raw_points(sensor)-origins[i];labels=maps[i].map_points(cp)
                            pos=probe.data.root_pos_w.torch[0].cpu().numpy()-origins[i]
                            v=probe.data.root_lin_vel_w.torch[0].cpu().numpy()
                            residual=(v-old_v[i])/.005-forces[i]-normal-friction
                            row=dict(case=cases[i]['id'],round=round_id,t=t,position=pos.tolist(),velocity=v.tolist(),
                                normal_force=normal.tolist(),friction_force=friction.tolist(),force=(normal+friction).tolist(),
                                applied_force=forces[i].tolist(),acceleration=((v-old_v[i])/.005).tolist(),balance_residual_N=float(np.linalg.norm(residual)),
                                filter_error_N=float(np.linalg.norm(net-normal)),points=cp.tolist(),labels=labels)
                            frame.append(row)
                        round_records.append(frame)
                records.append(round_records)
                for i,c in enumerate(cases):
                    steady=[f[i] for f in round_records if 3.<=f[i]['t']<=4.5]
                    load=np.array([np.dot(x['force'],normals[i]) for x in steady])
                    centroid_offset=np.array([(np.array(x['position'])-(targets[i]-origins[i]))@normals[i] for x in steady])
                    correct=all(x['labels'] and all(l['valid'] for l in x['labels']) for x in steady)
                    if round_id<3:
                        checks=dict(sustained_contact=bool((load>1).all()),force_balance=bool(np.median([x['balance_residual_N'] for x in steady])<1.),
                            filters=bool(max(x['filter_error_N'] for x in steady)<.1),target_mapping=correct,
                            no_through_surface=bool(centroid_offset.min()>=.025-.015),load_magnitude=bool(8<float(load.mean())<12))
                    else:checks=dict(empty_space=bool(max(np.linalg.norm(x['force']) for x in steady)<.1))
                    summary.append(dict(case=c['id'],round=round_id,kind='surface_press' if round_id<3 else 'empty_space',
                        passed=all(checks.values()),checks=checks,mean_normal_force_N=float(load.mean()),
                        minimum_center_offset_m=float(centroid_offset.min())))
                write(out/'partial_summary.json',summary)
            write(out/'measurements.json',records)
            allowed=[c['id'] for c in cases if all(x['passed'] for x in summary if x['case']==c['id'])]
            write(out/'result.json',dict(complete=True,checks=summary,validated_layouts=allowed,all_passed=len(allowed)==14,
                simulation_steps=sim.get_physics_step_count(),initialization_steps=initial,probe_steps=4800,
                diagnostic_only=True,robot_success_score=None,gravity_disabled_for_calibrated_force_test=True))
            assert sim.get_physics_step_count()<=initial+4800
        finally:
            if not (out/'measurements.json').exists():write(out/'partial_measurements.json',records)

if __name__=='__main__':main()

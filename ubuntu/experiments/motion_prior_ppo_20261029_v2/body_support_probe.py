"""Controlled load tests with replicas of actual G1 pelvis/thigh/torso colliders.

These are force-controlled fixtures, not free-standing robot successes. They
separate collision/load capacity from the policy's ability to reach a surface.
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import ROOT,read,write
ASSETS=ROOT/'experiments/furniture_test_20261028_v1'
CHECK=ROOT/'experiments/furniture_validation_20261028_v2'
sys.path.insert(1,str(CHECK))
from surface_map import SurfaceMap
from contact_adapter import raw_points
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.assets import RigidObject,RigidObjectCfg
from isaaclab.sensors import ContactSensor,ContactSensorCfg
import isaaclab.sim as sim
from isaaclab.sim import SimulationContext

def part_prim(stage,path,shapes,position):
    from pxr import UsdGeom,UsdPhysics,Gf,Vt,UsdShade,PhysxSchema
    root=UsdGeom.Xform.Define(stage,path)
    root.AddTranslateOp().Set(Gf.Vec3d(*map(float,position)))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0.)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(10.)
    material=UsdShade.Material(stage.GetPrimAtPath('/World/ProbeMaterial'))
    for i,shape in enumerate(shapes):
        mesh=UsdGeom.Mesh.Define(stage,path+f'/Shape_{i}')
        v=np.asarray(shape['vertices'],np.float32);f=np.asarray(shape['faces'],np.int32)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v));mesh.CreateFaceVertexCountsAttr([3]*len(f))
        mesh.CreateFaceVertexIndicesAttr(f.ravel().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,materialPurpose='physics')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--inventory',required=True);add_launcher_args(p);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False);cases=read(ASSETS/'cases.json');plans=read(CHECK/'probe_points.json')
    inventory=read(Path(a.inventory))['shapes']
    bybody={b:[s for s in inventory if s['body']==b] for b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link']}
    assert all(bybody.values())
    model=read(ROOT/'runs/g1_kinematics_geometry_04/model.json');weight=float(sum(model['body_mass'])*9.81);load=.5*weight
    cfg=sim.SimulationCfg(dt=.005,device='cuda:0',gravity=(0.,0.,0.));a.device='cuda:0';started=time.monotonic()
    origins=np.array([[0.,i*8.,0.] for i in range(14)]);records=[];summaries=[]
    with launch_simulation(cfg,a):
        context=SimulationContext(cfg);stage=sim.get_current_stage()
        from pxr import UsdShade,UsdPhysics
        material=UsdShade.Material.Define(stage,'/World/ProbeMaterial');api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(.8);api.CreateDynamicFrictionAttr(.6);api.CreateRestitutionAttr(0.)
        seats=[];probes={};sensors={};geometry={};maps=[SurfaceMap(c) for c in cases]
        for i,c in enumerate(cases):
            root=f'/World/Test_{i:02d}';sim.create_prim(root,'Xform')
            seats.append(RigidObject(RigidObjectCfg(prim_path=root+'/Seat',spawn=sim.UsdFileCfg(usd_path=c['mesh_usd'],activate_contact_sensors=True),init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(origins[i])))))
            n=np.asarray(c['target_normal']);rotation=Rotation.align_vectors(n[None],np.array([[0.,0.,1.]]))[0]
            for j in range(3):
                body=['pelvis','left_hip_yaw_link','right_hip_yaw_link'][j] if c['task']=='sitting' else 'torso_link'
                shapes=bybody[body];vertices=np.concatenate([np.asarray(s['vertices']) for s in shapes])
                extent=-float((rotation.apply(vertices)@n).min())
                path=root+f'/BodyProbe_{j}';park=origins[i]+[4.,j*2.,3.]
                part_prim(stage,path,shapes,park)
                probes[i,j]=RigidObject(RigidObjectCfg(prim_path=path,spawn=None,init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(park))))
                sensors[i,j]=ContactSensor(ContactSensorCfg(prim_path=path,filter_prim_paths_expr=[root+'/Seat'],track_friction_forces=True,
                    track_contact_points=True,history_length=4,max_contact_data_count_per_prim=512))
                geometry[i,j]=dict(body=body,rotation=rotation.as_quat(),extent=extent,shapes=shapes)
        context.reset();initial=context.get_physics_step_count();assert initial<=1000
        write(out/'manifest.json',dict(load_N=load,load_bodyweight_fraction=.5,fixture_mass_kg=10.,
            gravity_disabled=True,orientation_servo=True,scope='Replicas of actual G1 collider shapes under controlled force; not autonomous or whole-robot stability.',
            bodies=bybody,source_inventory=a.inventory))
        for j in range(3):
            targets=[];normals=[];extents=[];goals=[]
            for i,c in enumerate(cases):
                spec=plans[c['id']][j];target=origins[i]+spec['point'];n=np.asarray(spec['normal']);g=geometry[i,j]
                targets.append(target);normals.append(n);extents.append(g['extent']);goals.append(g['rotation'])
                probe=probes[i,j];probe.reset();pose=np.r_[target+(g['extent']+.10)*n,g['rotation']]
                probe.write_root_pose_to_sim_index(root_pose=torch.tensor(pose[None],device='cuda:0',dtype=torch.float32))
                probe.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device='cuda:0'));sensors[i,j].reset()
            frame_records=[]
            for tick in range(1200):
                assert time.monotonic()-started<3600
                t=(tick+1)*.005;fraction=np.clip((t-.5)/1.,0,1) if t<=4.5 else np.clip(1-(t-4.5),0,1)
                forces=[];old=[]
                for i,c in enumerate(cases):
                    probe=probes[i,j];pos=probe.data.root_pos_w.torch[0];v=probe.data.root_lin_vel_w.torch[0]
                    offset=extents[i]+.10-fraction*(.10+load/2500)
                    target=torch.tensor(targets[i]+offset*normals[i],device='cuda:0',dtype=torch.float32)
                    f=(2500*(target-pos)-316.227766*v).clamp(-2*load,2*load)
                    q=probe.data.root_quat_w.torch[0].cpu().numpy();omega=probe.data.root_ang_vel_w.torch[0]
                    error=(Rotation.from_quat(goals[i])*Rotation.from_quat(q).inv()).as_rotvec()
                    torque=(100*torch.tensor(error,device='cuda:0',dtype=torch.float32)-10*omega).clamp(-30,30)
                    forces.append(f.cpu().numpy().copy());old.append(v.cpu().numpy().copy())
                    probe.instantaneous_wrench_composer.add_forces_and_torques_index(forces=f[None,None],torques=torque[None,None],is_global=True)
                    probe.write_data_to_sim()
                context.step(render=False)
                for probe in probes.values():probe.update(.005)
                for seat in seats:seat.update(.005)
                for i in range(14):sensors[i,j].update(.005,force_recompute=True)
                if tick%4==3:
                    frame=[]
                    for i,c in enumerate(cases):
                        probe=probes[i,j];sensor=sensors[i,j]
                        normal=sensor.data.normal_force_matrix_w.torch.sum((1,2))[0].cpu().numpy()
                        friction=sensor.data.friction_force_matrix_w.torch.sum((1,2))[0].cpu().numpy();force=normal+friction
                        net=sensor.data.net_normal_forces_w.torch.sum(1)[0].cpu().numpy();v=probe.data.root_lin_vel_w.torch[0].cpu().numpy()
                        ps=raw_points(sensor)-origins[i];labels=maps[i].map_points(ps)
                        frame.append(dict(case=c['id'],body=geometry[i,j]['body'],round=j,t=t,force=force.tolist(),normal_force=normal.tolist(),friction_force=friction.tolist(),net_normal_force=net.tolist(),applied_force=forces[i].tolist(),
                            acceleration=((v-old[i])/.005).tolist(),force_balance_residual_N=float(np.linalg.norm(10*(v-old[i])/.005-forces[i]-force)),
                            filter_error_N=float(np.linalg.norm(normal-net)),points=ps.tolist(),labels=labels,pose=probe.data.root_pose_w.torch[0].cpu().tolist()))
                    frame_records.append(frame)
            records.append(frame_records)
            for i,c in enumerate(cases):
                steady=[f[i] for f in frame_records if 3<=f[i]['t']<=4.5]
                measured=np.array([np.dot(x['force'],normals[i]) for x in steady])
                checks=dict(sustained_contact=bool((measured>.1*load).all()),load_capacity=bool(.8*load<measured.mean()<1.2*load),
                    force_balance=bool(np.median([x['force_balance_residual_N'] for x in steady])<.05*load),
                    filter_consistency=bool(max(x['filter_error_N'] for x in steady)<.05*load))
                mapped=all(x['labels'] and all(y['valid'] for y in x['labels']) for x in steady)
                summaries.append(dict(case=c['id'],body=geometry[i,j]['body'],round=j,physical_support=all(checks.values()),
                    intended_contact_patch=mapped,checks=checks,mean_load_N=float(measured.mean()),target_load_N=load))
                # Park before another body replica approaches the same surface.
                probe=probes[i,j];probe.write_root_pose_to_sim_index(root_pose=torch.tensor([[*(origins[i]+[4.,j*2.,3.]),0,0,0,1]],device='cuda:0',dtype=torch.float32))
                probe.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1,6),device='cuda:0'))
            write(out/'partial_summary.json',summaries)
        write(out/'measurements.json',records)
        write(out/'result.json',dict(complete=True,checks=summaries,physical_steps=context.get_physics_step_count(),
            initial_physics_steps=initial,load_N=load,robot_success_claim=False,body_collider_load_fixture=True))
        assert context.get_physics_step_count()<=initial+3600

if __name__=='__main__':main()

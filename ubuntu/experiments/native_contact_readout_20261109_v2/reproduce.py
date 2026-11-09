"""Two fresh one-joint fixtures; isolate contact with unchanged solver settings.

Not a robot trial, sensor substitute, or permission to change production physics.
"""
import os,argparse,json,math,time
from pathlib import Path
import numpy as np
import torch
from isaaclab.app import add_launcher_args,launch_simulation
from isaaclab.sim import SimulationCfg,SimulationContext
from isaaclab_physx.physics import PhysxCfg
H=Path(__file__).resolve().parent;R=H.parent.parent;OUT=R/'outputs'/H.name/'fixture'
parser=argparse.ArgumentParser();add_launcher_args(parser);args=parser.parse_args()
assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
assert torch.cuda.device_count()==1 and '4090' in torch.cuda.get_device_name(0)
config=SimulationCfg(dt=.005,device='cuda:0',render_interval=4)
config.physics=PhysxCfg(solver_type=1,enable_external_forces_every_iteration=True,enable_stabilization=False,
                       solve_articulation_contact_last=False)

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    with launch_simulation(config,args):
        from pxr import UsdGeom,UsdPhysics,UsdShade,PhysxSchema,Gf,Sdf
        import omni.physics.tensors as tensors
        from isaaclab.sim.utils.stage import get_current_stage
        sim=SimulationContext(config);stage=get_current_stage()
        def material(path,static,dynamic):
            mat=UsdShade.Material.Define(stage,path);api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(static);api.CreateDynamicFrictionAttr(dynamic);api.CreateRestitutionAttr(0.)
            return mat
        floor_material=material('/World/FloorMaterial',.8,.6);robot_material=material('/World/LinkMaterial',.5,.5)
        def cube(path,size,translation,mat,collision=True):
            c=UsdGeom.Cube.Define(stage,path);c.CreateSizeAttr(1.)
            c.AddTranslateOp().Set(Gf.Vec3d(*translation));c.AddScaleOp().Set(Gf.Vec3d(*size))
            if collision:UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat,materialPurpose='physics')
            return c
        fixture_paths=[]
        for i in range(2):
            # Fixture0 has no floor; fixture1 has the only additional contact.
            x=2.*i;prefix=f'/World/Probe_{i}';UsdGeom.Xform.Define(stage,prefix)
            base=UsdGeom.Xform.Define(stage,prefix+'/Base');base.AddTranslateOp().Set(Gf.Vec3d(x,0.,.5))
            bp=base.GetPrim();UsdPhysics.RigidBodyAPI.Apply(bp);UsdPhysics.MassAPI.Apply(bp).CreateMassAttr(1.)
            UsdPhysics.ArticulationRootAPI.Apply(bp)
            art=PhysxSchema.PhysxArticulationAPI.Apply(bp)
            art.CreateEnabledSelfCollisionsAttr(False);art.CreateSolverPositionIterationCountAttr(8)
            art.CreateSolverVelocityIterationCountAttr(4);art.CreateSleepThresholdAttr(0.)
            world=UsdPhysics.FixedJoint.Define(stage,prefix+'/Fixed');world.CreateBody1Rel().SetTargets([bp.GetPath()])
            world.CreateLocalPos0Attr(Gf.Vec3f(x,0.,.5))
            link=UsdGeom.Xform.Define(stage,prefix+'/Link');link.AddTranslateOp().Set(Gf.Vec3d(x,0.,.5))
            link.AddOrientOp().Set(Gf.Quatf(math.cos(.3),Gf.Vec3f(0,math.sin(.3),0)))
            lp=link.GetPrim();UsdPhysics.RigidBodyAPI.Apply(lp)
            mass=UsdPhysics.MassAPI.Apply(lp);mass.CreateMassAttr(2.);mass.CreateCenterOfMassAttr(Gf.Vec3f(0.,0.,-.275))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(2*(.04**2+.55**2)/12,2*(.04**2+.55**2)/12,2*(2*.04**2)/12))
            rb=PhysxSchema.PhysxRigidBodyAPI.Apply(lp);rb.CreateLinearDampingAttr(0.);rb.CreateAngularDampingAttr(0.)
            cube(prefix+'/Link/Shape',(.04,.04,.55),(0,0,-.275),robot_material)
            joint=UsdPhysics.RevoluteJoint.Define(stage,prefix+'/Hinge');jp=joint.GetPrim()
            joint.CreateBody0Rel().SetTargets([bp.GetPath()]);joint.CreateBody1Rel().SetTargets([lp.GetPath()])
            joint.CreateAxisAttr('Y');joint.CreateLowerLimitAttr(-90.);joint.CreateUpperLimitAttr(90.)
            jp.AddAppliedSchema('PhysicsJointStateAPI:angular')
            jp.CreateAttribute('state:angular:physics:position',Sdf.ValueTypeNames.Float).Set(math.degrees(.6))
            jp.CreateAttribute('state:angular:physics:velocity',Sdf.ValueTypeNames.Float).Set(0.)
            drive=UsdPhysics.DriveAPI.Apply(jp,'angular');drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(160.);drive.CreateDampingAttr(4.);drive.CreateMaxForceAttr(40.)
            drive.CreateTargetPositionAttr(math.degrees(.6));drive.CreateTargetVelocityAttr(0.)
            PhysxSchema.PhysxJointAPI.Apply(jp).CreateJointFrictionAttr(0.)
            if i: cube(prefix+'/Floor',(1.,1.,.02),(x,0.,-.01),floor_material)
            fixture_paths.append(str(bp.GetPath()))
        stage.GetRootLayer().Export(str(OUT/'fixture.usda'))
        sim.reset()
        view=tensors.create_simulation_view('torch');view.set_subspace_roots('/')
        art=view.create_articulation_view('/World/Probe_*/Base')
        assert art.count==2 and art.max_dofs==1,(art.count,art.max_dofs)
        idx=torch.arange(2,device='cuda:0',dtype=torch.int32)
        # Explicit fresh state after the runtime's initialization step. Neither
        # fixture is loaded here; lower-corner clearance is positive at0.6rad.
        initial=torch.full((2,1),.6,device='cuda:0')
        art.set_dof_positions(initial,idx);art.set_dof_velocities(torch.zeros_like(initial),idx)
        q0=art.get_dof_positions().clone();assert torch.max(abs(q0-.6)).item()<1e-4,q0
        (OUT/'initialization.json').write_text(json.dumps(dict(physics_steps=sim.get_physics_step_count(),q=q0.cpu().tolist(),qd=art.get_dof_velocities().cpu().tolist(),fresh_reset=True))+'\n')
        target=q0.clone();records=[]
        for control in range(500):
            # Same20ms,.1filter,.003rad increment as the robot; no target jumps.
            target=target+torch.clamp(.1*(torch.zeros_like(target)-target),-.003,.003)
            art.set_dof_position_targets(target,idx)
            for substep in range(4):
                sim.step(render=False)
                records.append([art.get_dof_positions().clone().cpu().numpy(),
                    art.get_dof_velocities().clone().cpu().numpy(),target.cpu().numpy().copy()])
        a=np.asarray(records);np.savez_compressed(OUT/'native_states.npz',q=a[:,0,:,0],qd=a[:,1,:,0],target=a[:,2,:,0],dt=.005)
        out=[]
        for i,label in enumerate(('free','floor_contact')):
            q=a[-1000:,0,i,0];v=a[-1000:,1,i,0]
            out.append(dict(fixture=label,angle_range_rad=float(np.ptp(q)),angle_change_rad=float(q[-1]-q[0]),
              mean_native_velocity_rad_s=float(v.mean()),max_abs_native_velocity_rad_s=float(abs(v).max()),
              native_velocity_integral_rad=float(v.sum()*.005),mean_angle_rad=float(q.mean()),
              minimum_corner_height_m=float((.5-.55*np.cos(q)-.02*abs(np.sin(q))).min())))
        result=dict(fixtures=out,physics_steps=sim.get_physics_step_count(),active_steps=2000,wall_s=time.monotonic()-started,
          fixture_parameters=dict(mass_kg=2.,length_m=.55,hinge_height_m=.5,initial_angle_rad=.6,drive_Kp=160.,drive_Kd=4.,max_effort_Nm=40.,dt=.005,position_iterations=8,velocity_iterations=4),
          changed_robot_configuration=False,robot_cycles=0,source_readings_not_corrected=True,
          conclusion='A controlled readout reproduction only; does not establish a robot repair or authorize parameter changes.')
        (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

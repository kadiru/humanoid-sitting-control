"""Import the original G1 into isolated CPU PhysX and audit its native state.

This is model/runtime admission, not a successful task or sensor repair.
No stepping, trajectory replay, tuning, or camera is performed here.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from isolated_runtime import start


def run(contract, out, probe=None):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdUtils, Gf, UsdShade
    import omni.physx
    import omni.physics.tensors as tensors
    c = json.loads(contract.read_text())
    for file, digest in c['input_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest, file
    properties = json.loads(Path(c['properties']).read_text())
    names = properties['joint_names']
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, 'Z')
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    scene = UsdPhysics.Scene.Define(stage, '/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    pa = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    pa.CreateEnableGPUDynamicsAttr(False)
    pa.CreateBroadphaseTypeAttr('MBP')
    pa.CreateSolverTypeAttr('TGS')
    pa.CreateEnableExternalForcesEveryIterationAttr(True)
    pa.CreateEnableStabilizationAttr(False)
    pa.CreateFrictionTypeAttr('patch')
    pa.CreateFrictionCorrelationDistanceAttr(.025)
    pa.CreateFrictionOffsetThresholdAttr(.04)
    robot = stage.DefinePrim('/Robot', 'Xform')
    robot.GetReferences().AddReference(c['usd'])
    # Make nested instance proxies editable before applying runtime overrides.
    for prim in list(stage.Traverse()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    root_paths = []
    bodies = []
    joints = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root_paths.append(str(prim.GetPath()))
            a = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            a.CreateEnabledSelfCollisionsAttr(False)
            a.CreateSolverPositionIterationCountAttr(8)
            a.CreateSolverVelocityIterationCountAttr(4)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(prim.GetName())
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
            a = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            a.CreateDisableGravityAttr(False)
            a.CreateLinearDampingAttr(0.)
            a.CreateAngularDampingAttr(0.)
            a.CreateMaxLinearVelocityAttr(1000.)
            # Match G1_CFG's documented degrees/second field verbatim.
            a.CreateMaxAngularVelocityAttr(1000.)
            a.CreateMaxDepenetrationVelocityAttr(1.)
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints[prim.GetName()] = prim
    assert set(joints) == set(names), (set(names)-set(joints), set(joints)-set(names))
    assert len(root_paths) == 1 and len(bodies) == 44, (root_paths, len(bodies))
    native = properties['native']
    expected = {'stiffnesses': native['get_dof_stiffnesses'][0],
                'dampings': native['get_dof_dampings'][0],
                'max_forces': [], 'armatures': []}
    for i, name in enumerate(names):
        joint = joints[name]
        drive = UsdPhysics.DriveAPI.Apply(joint, 'angular')
        drive.CreateTypeAttr('force')
        # USD angular drive gains are per degree; tensor gains are per radian.
        drive.CreateStiffnessAttr(float(expected['stiffnesses'][i]*np.pi/180))
        drive.CreateDampingAttr(float(expected['dampings'][i]*np.pi/180))
        force = 20. if 'ankle' in name else 300.
        armature = .001 if any(x in name for x in
            ('_zero_', '_one_', '_two_', '_three_', '_four_', '_five_', '_six_')) else .01
        drive.CreateMaxForceAttr(force)
        a = PhysxSchema.PhysxJointAPI.Apply(joint)
        a.CreateArmatureAttr(armature)
        a.CreateJointFrictionAttr(0.)
        expected['max_forces'].append(force)
        expected['armatures'].append(armature)
    floor = UsdGeom.Cube.Define(stage, '/RestFloor')
    floor.CreateSizeAttr(1.)
    floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.05))
    floor.AddScaleOp().Set(Gf.Vec3f(5., 5., .1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    rb = UsdPhysics.RigidBodyAPI.Apply(floor.GetPrim())
    rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(floor.GetPrim()).CreateMassAttr(1000.)
    material = UsdShade.Material.Define(stage, '/FloorMaterial')
    pm = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    pm.CreateStaticFrictionAttr(.8)
    pm.CreateDynamicFrictionAttr(.6)
    pm.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(
        material, materialPurpose='physics')
    if 'support' in c:
        support = c['support']
        surface = UsdGeom.Cube.Define(stage, '/Support')
        surface.CreateSizeAttr(1.)
        surface.AddTranslateOp().Set(Gf.Vec3d(*support['position']))
        x,y,z,w = support['quaternion_xyzw']
        surface.AddOrientOp().Set(Gf.Quatf(w,Gf.Vec3f(x,y,z)))
        surface.AddScaleOp().Set(Gf.Vec3f(*support['size']))
        UsdPhysics.CollisionAPI.Apply(surface.GetPrim())
        body = UsdPhysics.RigidBodyAPI.Apply(surface.GetPrim())
        body.CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(surface.GetPrim()).CreateMassAttr(1000.)
        UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material,materialPurpose='physics')
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(stage).ToLongInt()
    stage.Export(str(out/'stage.usda'))
    simulation = omni.physx.get_physx_simulation_interface()
    assert simulation.attach_stage(stage_id)
    omni.physx.get_physx_interface().force_load_physics_from_usd()
    # Explicit tensor initialization does not advance time.
    view = tensors.create_simulation_view('numpy', stage_id)
    articulation = view.create_articulation_view(root_paths[0])
    assert articulation.count == 1, 'G1_NATIVE_ARTICULATION_MISSING'
    found_names = list(articulation.shared_metatype.dof_names)
    indices = np.array([found_names.index(n) for n in names])
    checks = {}
    actual = {}
    for name, value in expected.items():
        data = getattr(articulation, 'get_dof_'+name)()[0, indices]
        actual[name] = data.tolist()
        checks[name] = bool(np.allclose(data, value, rtol=2e-6, atol=1e-7))
    bank = np.load(c['reset'])
    ids = np.array([0], dtype=np.uint32)
    reorder = np.array([names.index(n) for n in found_names])
    for field, method in [('q', 'set_dof_positions'), ('qd', 'set_dof_velocities'),
                          ('command', 'set_dof_position_targets')]:
        data = np.ascontiguousarray(bank[field][:, 0, reorder], dtype=np.float32)
        getattr(articulation, method)(data, ids)
    articulation.set_root_transforms(np.ascontiguousarray(bank['root_pose'][:, 0]), ids)
    articulation.set_root_velocities(np.ascontiguousarray(bank['root_velocity'][:, 0]), ids)
    view.update_articulations_kinematic()
    q = articulation.get_dof_positions()[0, indices].copy()
    checks['reset_q'] = bool(np.max(np.abs(q-bank['q'][0, 0])) < 1e-6)
    checks['links44'] = len(articulation.shared_metatype.link_names) == 44
    result = dict(native_checks=checks, native_properties=actual,
        native_joint_names=found_names, native_link_names=list(articulation.shared_metatype.link_names),
        model_import_passed=all(checks.values()), root_paths=root_paths,
        physics_steps=0, complete_cycles=0, sensor_repair=False,
        scope='Native CPU model import only; collision/dynamic/camera parity not established')
    stage.Export(str(out/'stage.usda'))
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    if probe is not None:
        assert result['model_import_passed'], 'MODEL_IMPORT_REJECTED'
        probe(stage, simulation, view, articulation, names, indices, c, out)
    simulation.detach_stage()
    cache.Erase(stage)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('contract', type=Path)
    parser.add_argument('output', type=Path)
    a = parser.parse_args()
    a.output.mkdir(exist_ok=False)
    started = time.monotonic()
    app = start()
    try:
        result = run(a.contract, a.output)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        (a.output/'failure.json').write_text(json.dumps(
            {'error': repr(exc), 'physics_steps': 0, 'sensor_repair': False}))
        raise
    finally:
        import omni.physx
        omni.physx.get_physx_simulation_interface().detach_stage()
        (a.output/'resources.json').write_text(json.dumps(
            {'wall_s': time.monotonic()-started, 'GPU_devices_exposed': False}))
        app.shutdown()


if __name__ == '__main__':
    main()

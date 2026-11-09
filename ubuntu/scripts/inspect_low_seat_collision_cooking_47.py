"""Zero-tick read-only collision-cooking diagnostic on a separate asset copy.

No environment, reset, physics step, policy or trajectory is instantiated.
Use the installed cooking API exactly as its shipped interface test does,
with synchronous requests. All edits are confined to an anonymous USD session
layer. The original model, controller and downloaded USD remain unchanged.
"""
import argparse, json, hashlib, datetime
from pathlib import Path
import numpy as np
from isaaclab.app import add_launcher_args, launch_simulation
from humanoid_sitting.tasks.config import SittingEnvCfg

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
OUT=D/'cooked_contact_47'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();add_launcher_args(p);args=p.parse_args()
    OUT.mkdir(exist_ok=False)
    cfg=SittingEnvCfg();cfg.scene.num_envs=1;cfg.sim.device=args.device
    asset=D/'asset_contact_47/g1.usd';modelpath=R/'runs/g1_kinematics_geometry_04/model.json';meta=json.loads(modelpath.read_text())
    before={str(x):sha(x) for x in [asset,modelpath,Path(__file__)]}
    names=['pelvis','left_hip_pitch_link','right_hip_pitch_link','left_hip_yaw_link','right_hip_yaw_link']
    with launch_simulation(cfg,args):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, PhysicsSchemaTools, PhysxSchema, Gf
        import omni.timeline
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        timeline=omni.timeline.get_timeline_interface()
        assert not timeline.is_playing()
        time_before=float(timeline.get_current_time())
        stage=Usd.Stage.Open(str(asset));assert stage
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdPhysics.Scene.Define(stage,'/diagnosticPhysicsScene')
        cache=UsdUtils.StageCache.Get();sid=cache.Insert(stage).ToLongInt();cook=get_physx_cooking_interface();rows=[]
        for bodyname in names:
            body=stage.GetPrimAtPath('/g1/'+bodyname);prim=stage.GetPrimAtPath('/g1/'+bodyname+'/collisions/mesh_0');assert body and prim
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='convexHull'
            rel=UsdGeom.XformCache().ComputeRelativeTransform(prim,body)[0]
            raw=np.array([list(rel.Transform(Gf.Vec3d(*[float(x) for x in v]))) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
            frozen=np.asarray(next(x for x in meta['collisions'] if x['body']==bodyname)['points'])
            assert raw.shape==frozen.shape and np.array_equal(raw,frozen),(bodyname,raw.shape,frozen.shape)
            received=[]
            def ready(result,convexes):
                received.append((result,convexes))
            cook.request_convex_collision_representation(stage_id=sid,collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),run_asynchronously=False,on_result=ready)
            assert len(received)==1,(bodyname,'Missing synchronous cooking callback')
            result,convexes=received[0];assert result==PhysxCollisionRepresentationResult.RESULT_VALID,(bodyname,str(result))
            output=[]
            for convex in convexes:
                points=np.array([list(rel.Transform(Gf.Vec3d(*[float(x) for x in v]))) for v in convex.vertices])
                output.append(dict(vertices=points.tolist(),indices=list(convex.indices),polygons=[dict(plane=list(x.plane),num_vertices=x.num_vertices,index_base=x.index_base) for x in convex.polygons]))
            physics=PhysxSchema.PhysxCollisionAPI(prim)
            row=dict(body=bodyname,asset_path=str(prim.GetPath()),raw_points=len(raw),raw_points_exact_frozen_model=True,convex_count=len(output),cooked_convexes=output,contact_offset=physics.GetContactOffsetAttr().Get(),rest_offset=physics.GetRestOffsetAttr().Get(),collision_enabled=True,approximation='convexHull')
            rows.append(row);(OUT/(bodyname+'.json')).write_text(json.dumps(row,indent=2));print(bodyname,'raw',len(raw),'cooked',[len(x['vertices']) for x in output],flush=True)
        assert not timeline.is_playing() and float(timeline.get_current_time())==time_before
        result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Standalone synchronous cooking query of the configured USD copy; raw points match frozen model exactly. No environment/reset/step/policy. Cooked representations are queries, not live contact-force validation.',physics_steps=0,control_ticks=0,settling_ticks=0,optimizer_updates=0,holdout_used=False,timeline_before=time_before,timeline_after=float(timeline.get_current_time()),timeline_stopped=True,inputs=before,rows=rows)
        cache.Erase(stage)
    for path,digest in before.items():assert sha(path)==digest
    (OUT/'results.json').write_text(json.dumps(result,indent=2));(OUT/'SHA256.json').write_text(json.dumps({x.name:sha(x) for x in OUT.iterdir() if x.is_file()},indent=2))

if __name__=='__main__':main()

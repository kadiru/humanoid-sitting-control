"""One complete20s autonomous trial, original action map and pass thresholds.

The sitting checkpoint is the requested imitation-PPO final checkpoint; leaning
uses the last completed leaning policy and its own original action decoder.
No teaching, nudges, new optimizer steps, or ground-truth actor observations.
"""
import argparse
import sys
import time
import numpy as np
import torch
from io_utils import ROOT, HERE, ASSETS, TRAIN, PPO, OUT, read, write, sha

def collision_inventory(raw, out):
    from pxr import Usd,UsdGeom,UsdPhysics
    import trimesh
    import isaaclab.sim as sim
    stage=sim.get_current_stage();cache=UsdGeom.XformCache()
    bodies={name:stage.GetPrimAtPath('/World/envs/env_0/Robot/'+name) for name in raw.scene['robot'].body_names}
    shapes=[]
    for body,root in bodies.items():
        if not root.IsValid():continue
        for prim in Usd.PrimRange(root):
            if prim!=root and prim.HasAPI(UsdPhysics.RigidBodyAPI):continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Mesh):continue
            g=UsdGeom.Mesh(prim);v=np.asarray(g.GetPointsAttr().Get(),float)
            counts=np.asarray(g.GetFaceVertexCountsAttr().Get());indices=np.asarray(g.GetFaceVertexIndicesAttr().Get())
            faces=[];start=0
            for n in counts:
                poly=indices[start:start+n];start+=n
                faces.extend([[int(poly[0]),int(poly[j]),int(poly[j+1])] for j in range(1,n-1)])
            transform=np.asarray(cache.ComputeRelativeTransform(prim,root)[0])
            v=(np.c_[v,np.ones(len(v))]@transform)[:,:3]
            approx=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else 'none'
            mesh=trimesh.Trimesh(v,np.array(faces),process=False)
            if approx=='convexHull':mesh=trimesh.convex.convex_hull(mesh.vertices)
            shapes.append(dict(body=body,path=str(prim.GetPath()),approximation=approx,vertices=mesh.vertices.tolist(),faces=mesh.faces.tolist()))
    assert shapes,'NO_RUNTIME_ROBOT_COLLISION_GEOMETRY'
    write(out/'robot_collision_shapes.json',dict(shapes=shapes,scope='Runtime USD collider shapes; convexHull displayed geometrically. Not a bit-exact PhysX cooking certification.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    from isaaclab.app import add_launcher_args
    add_launcher_args(p);a=p.parse_args()
    c=next(c for c in read(ASSETS/'cases.json') if c['id']==a.case)
    assert a.case in read(OUT/'probes/result.json')['validated_layouts'],'COLLISION_VALIDATION_REQUIRED'
    folder=TRAIN if c['task']=='sitting' else ROOT/'experiments/full_body_contact_trainer_4090_step_v5'
    sys.path.insert(0,str(folder));sys.path.insert(1,str(ROOT/'scripts'));sys.path.insert(2,str(ASSETS))
    from environment import FullBodyEnv
    from policy import Learner
    import run_resting_posture as base
    import run_supported_postures as support
    from scene_adapter import make_cfg
    from contact_adapter import MeshMetrics
    a.device='cuda:0';a.num_envs=1;a.seed=64301;a.mode='evaluate';a.depth_check=True;a.enable_cameras=True
    out=OUT/('robot_'+c['id']);assert str(out)==a.output;out.mkdir(exist_ok=False)
    bankroot=ROOT/'outputs/full_body_contact_imitation_v4/dataset'/c['task']
    metadata=read(bankroot/'metadata.json');metadata['cases']=1;metadata['ids']=[c['id']]
    with np.load(bankroot/'bank.npz') as z:
        bank={k:z[k][:1].copy() for k in ['reset_q','reset_qd','reset_command','reset_root_pose','reset_root_velocity','phase_times','admitted']}
        bank['q_post']=z['q_post'][:,:1].copy();bank['reference']=np.zeros(1,int)
    ck=PPO/'train_sitting_64111/update_256.pt' if c['task']=='sitting' else ROOT/'outputs/full_body_contact_imitation_v5/train_leaning_64111/update_256.pt'
    assert sha(ck)==read(OUT/'policy_bindings.json')[str(ck)]
    learner=Learner(np.zeros(336),np.ones(336),64301,device='cuda:0');learner.load(ck)
    adapter=MeshMetrics(c,support.metrics,support);support.metrics=adapter
    cfg=make_cfg(a,c);started=time.monotonic()
    with base.launch_simulation(cfg,a):
        raw=base.gym.make('Humanoid-Sit-Learned-G1',cfg=cfg).unwrapped
        env=FullBodyEnv(raw,a,out,bank,metadata)
        write(out/'startup.json',env.startup)
        write(out/'manifest.json',dict(case=c,checkpoint=str(ck),checkpoint_sha256=sha(ck),source=str(folder),
            action_limits_unchanged=True,success_thresholds_unchanged=True,task_policy=c['task'],
            scope='new furniture generalization; excluded from protected72 development comparison'))
        try:
            collision_inventory(raw,out)
            with torch.no_grad():
                for tick in range(1000):
                    assert time.monotonic()-started<3600
                    latent=learner.choose(env.observation(),stochastic=False)[0]
                    env.step(latent,record=True)
            env.save_records(out/'trajectory.npz')
            write(out/'contact_mapping.json',dict(validated=adapter.validated,frames=adapter.frames,
                thresholds=dict(surface_distance_m=.015,perimeter_inset_m=.01),multiple_normal_policy='mark score unvalidated'))
        finally:env.close()
    from audit import physical
    with np.load(out/'trajectory.npz') as z:d={k:z[k] for k in z.files}
    independent=physical(d);warm=bool(env.startup['warm_failure'][0])
    for key in ['failure','legal','no_slip']:np.testing.assert_array_equal(independent[key],d[key])
    for key in ['alive','good']:np.testing.assert_array_equal(independent[key]&~np.array(warm),d[key])
    row=read(out/'results.json')['episodes'][0]
    assert row['steps']==1000 and row['duration_s']==20 and row['full_standing_start']
    assert row['stable']==bool(independent['stable'][0] and not warm)
    assert row['clean']==bool(independent['clean'][0] and not warm)
    write(out/'independent_audit.json',dict(passed=True,complete_trials=1,full_seconds=20,
        contact_mapping_validated=adapter.validated,clean=row['clean'] if adapter.validated else None,
        stable=row['stable'] if adapter.validated else None,physical_failures=row['physical_failures'],
        final_hold_s=row['final_hold_s'],slip_ticks=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],
        score_not_promotable_reason=None if adapter.validated else 'mixed or missing contact normals',
        success_thresholds_changed=False,action_limits_changed=False,
        trajectory_sha256=sha(out/'trajectory.npz')))

if __name__=='__main__':main()

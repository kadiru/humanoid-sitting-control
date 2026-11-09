"""Export the actual simulation joint frames and collision points for offline kinematics."""
import json,sys
from pathlib import Path
import numpy as np
from isaaclab.app import add_launcher_args,launch_simulation
import argparse
from humanoid_sitting.tasks.config import SittingEnvCfg
p=argparse.ArgumentParser();p.add_argument('--output',required=True);add_launcher_args(p);a=p.parse_args();sys.argv=[sys.argv[0]]
cfg=SittingEnvCfg();cfg.scene.num_envs=1;cfg.sim.device=a.device
with launch_simulation(cfg,a):
 import gymnasium as gym
 import humanoid_sitting.tasks
 from pxr import Usd,UsdGeom,UsdPhysics
 import omni.usd
 env=gym.make('Humanoid-Sit-Fixed-G1',cfg=cfg).unwrapped;env.reset();stage=omni.usd.get_context().get_stage();robot=env.scene['robot'];from scipy.spatial.transform import Rotation
 local_com=Rotation.from_quat(robot.data.body_pose_w.torch[0,:,3:].cpu().numpy()).inv().apply((robot.data.body_com_pos_w.torch[0]-robot.data.body_pos_w.torch[0]).cpu().numpy())
 result={'body_com_local':local_com.tolist(),'joints':[],'collisions':[],'body_names':robot.body_names,'joint_names':robot.joint_names,'default_q':robot.data.default_joint_pos.torch[0].cpu().tolist(),'body_mass':robot.data.body_mass.torch[0].cpu().tolist(),'initial_body_pose':robot.data.body_pose_w.torch[0].cpu().tolist()}
 def quat(q):return [*q.GetImaginary(),q.GetReal()]
 for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0/Robot'),Usd.TraverseInstanceProxies()):
  if prim.IsA(UsdPhysics.Joint):
   joint=UsdPhysics.Joint(prim)
   result['joints'].append(dict(name=prim.GetName(),type=prim.GetTypeName(),body0=[str(x).split('/')[-1] for x in joint.GetBody0Rel().GetTargets()],body1=[str(x).split('/')[-1] for x in joint.GetBody1Rel().GetTargets()],p0=list(joint.GetLocalPos0Attr().Get()),p1=list(joint.GetLocalPos1Attr().Get()),q0=quat(joint.GetLocalRot0Attr().Get()),q1=quat(joint.GetLocalRot1Attr().Get()),axis=prim.GetAttribute('physics:axis').Get(),limits=[prim.GetAttribute('physics:lowerLimit').Get(),prim.GetAttribute('physics:upperLimit').Get()]))
  if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Boundable):
   body=prim
   while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
   if not body:continue
   mat=UsdGeom.XformCache().ComputeRelativeTransform(prim,body)[0]
   if prim.IsA(UsdGeom.Mesh):
    local_points=UsdGeom.Mesh(prim).GetPointsAttr().Get()
   elif prim.IsA(UsdGeom.Cylinder):
    cyl=UsdGeom.Cylinder(prim);radius=cyl.GetRadiusAttr().Get();height=cyl.GetHeightAttr().Get();axis='XYZ'.index(cyl.GetAxisAttr().Get());others=[i for i in range(3) if i!=axis];local_points=[]
    for z in (-height/2,height/2):
     for angle in np.linspace(0,2*np.pi,64,endpoint=False):
      v=np.zeros(3);v[axis]=z;v[others[0]]=radius*np.cos(angle);v[others[1]]=radius*np.sin(angle);local_points.append(v)
   elif prim.IsA(UsdGeom.Cube):
    half=UsdGeom.Cube(prim).GetSizeAttr().Get()/2;local_points=[np.array(bits)*half for bits in __import__('itertools').product((-1,1),repeat=3)]
   else:
    extent=UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim),Usd.TimeCode.Default());local_points=[np.array([extent[bits[j]][j] for j in range(3)]) for bits in __import__('itertools').product((0,1),repeat=3)]
   from pxr import Gf
   points=[list(mat.Transform(Gf.Vec3d(*[float(x) for x in point]))) for point in local_points]
   result['collisions'].append(dict(body=body.GetName(),path=str(prim.GetPath()),type=prim.GetTypeName(),cube_size=UsdGeom.Cube(prim).GetSizeAttr().Get() if prim.IsA(UsdGeom.Cube) else None,points=points))
 out=Path(a.output);out.mkdir(exist_ok=False,parents=True);(out/'model.json').write_text(json.dumps(result,indent=2)+'\n');print('EXPORTED',out,len(result['joints']),len(result['collisions']));env.close()

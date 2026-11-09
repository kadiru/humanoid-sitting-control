"""Capture-aligned audit data. World poses never enter the recognizer or policy."""
import numpy as np
from scipy.spatial.transform import Rotation
from paths import write

class Observer:
    def __init__(self,intent):self.intent=intent;self.last=None
    def __call__(self,model,K,T,depth,q,gravity,prior=None):
        from perception import observe
        value,info=observe(model,K,T,depth,q,gravity,self.intent,prior)
        self.last=(value,info)
        return value,info

def save(env,observer,folder,index):
    import run_supported_postures as support
    data=env.raw.scene['rear_camera'].data;robot=env.robot.data
    origin=env.raw.scene.env_origins[0].cpu().numpy()
    # Root physical pose is synchronized to the image acquisition above. The
    # camera metadata's Fabric-parent cache may still contain the nominal pose.
    R=Rotation.from_quat(robot.root_quat_w.torch[0].cpu().numpy()).as_matrix()
    root=robot.root_pos_w.torch[0].cpu().numpy()-origin
    T=np.asarray(support.REAR_CAMERA)
    position=root+R@T[:3,3];camera_R=R@T[:3,:3]
    value,info=observer.last
    np.savez_compressed(folder/f'frame_{index}.npz',
        depth=data.output['distance_to_image_plane'].torch[0].cpu().numpy().squeeze(),
        rgb=data.output['rgb'].torch[0].cpu().numpy(),K=data.intrinsic_matrices.torch[0].cpu().numpy(),
        camera_position=position,camera_rotation=camera_R,
        stale_metadata_position=data.pos_w.torch[0].cpu().numpy()-origin,
        stale_metadata_quaternion=data.quat_w_ros.torch[0].cpu().numpy(),
        root_position=root,root_rotation=R,q=robot.joint_pos.torch[0].cpu().numpy(),
        gravity=robot.projected_gravity_b.torch[0].cpu().numpy(),
        geometry=np.zeros(13) if value is None else value,
        selected_points=np.asarray(info.get('selected_points',[])).reshape(-1,3),
        selected_pixels=np.asarray(info.get('selected_pixels',[]),int).reshape(-1,2),
        physics_step=np.array(env.raw.sim.get_physics_step_count()),clock_s=np.array(env.t*.02))
    write(folder/f'frame_{index}.json',{k:v for k,v in info.items() if k not in ['selected_points','selected_pixels']})

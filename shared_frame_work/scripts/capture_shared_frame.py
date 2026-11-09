"""Synchronized moving RGB-D benchmark; frozen sensor controller generates motion.

Only record_truth() reads privileged poses, and its output is stored separately.
No teacher, PPO, online training, or perception feedback control is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
import gymnasium as gym
from scipy.spatial.transform import Rotation
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim
from humanoid_sitting.tasks.student import DeploymentEnvCfg
from humanoid_sitting.deployment import DeploymentStudent
from humanoid_sitting.sim_deployment_sensor import SimSensorInterface
from humanoid_sitting.soft_landing import ContactEstimator, SoftLandingController
import humanoid_sitting.tasks
from seat_depth_estimator import estimate_body_frame


def record_truth(raw):
    r = raw.scene['robot']
    return dict(root_pose=r.data.root_pose_w.torch.cpu().numpy().copy(),
                body_pose=r.data.body_pose_w.torch.cpu().numpy().copy(),
                seat_pose=raw.scene['seat'].data.root_pose_w.torch.cpu().numpy().copy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=16.)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--case-start', type=int, default=0)
    parser.add_argument('--geometry-bank', help='Simulator-only JSON geometry bank; never passed to the observer')
    parser.add_argument('--track-online', action='store_true', help='Update the new observer during simulation; no control feedback')
    parser.add_argument('--camera-profile', choices=['original', 'rear_down'], default='original')
    add_launcher_args(parser)
    a = parser.parse_args(); a.enable_cameras = True
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    assert shutil.disk_usage(out).free > 30*2**30
    root = Path(__file__).resolve().parents[1]
    source_dir = out/'source'; source_dir.mkdir()
    for path in (Path(__file__), root/'sitting_task/src/humanoid_sitting/shared_frame.py', root/'scripts/run_shared_frame.py'):
        shutil.copy2(path, source_dir/path.name)
    bank = [dict(family=family, shape=shape, depth=d, width=w, height=h, x=-.40, y=y,
                 yaw=np.deg2rad(yaw)) for family, shape, d, w, h, y, yaw in [
        ('chair', 'rect', .52, .88, .30, .008, 3.),
        ('bench', 'rect', .48, 1.02, .31, -.008, -4.),
        ('box', 'rect', .54, .84, .29, .012, 5.),
        ('rigid_couch', 'rect', .50, 1.06, .30, -.010, -3.),
        ('bench', 'chamfered', .53, .96, .305, .004, 4.),
        ('box', 'ellipse', .51, .92, .295, -.004, -5.)]]
    if a.geometry_bank:
        bank_path = Path(a.geometry_bank)
        bank = json.loads(bank_path.read_text())
        if not isinstance(bank, list) or not bank:
            parser.error('Geometry bank must be a nonempty list')
        for g in bank:
            if g.get('family') not in ('chair','bench','box','rigid_couch') or g.get('shape') not in ('rect','chamfered','ellipse'):
                parser.error('Unsupported fixture family/shape')
            if any(not np.isfinite(g.get(k, np.nan)) for k in ('depth','width','height','x','y','yaw')):
                parser.error('Geometry dimensions and poses must be finite')
            if min(g['depth'],g['width'],g['height']) <= 0:
                parser.error('Geometry dimensions must be positive')
        shutil.copy2(bank_path, source_dir/'geometry_bank.json')
    bank = bank[a.case_start:a.case_start+a.count]
    if not bank:
        parser.error('Requested case range is empty')
    cfg = DeploymentEnvCfg(geometry_stage=2, geometry_seed=41017, geometry_count=len(bank),
        geometry_bank=bank, include_families=True, arm_clearance=True)
    cfg.scene.num_envs = len(bank); cfg.scene.env_spacing = 4.; cfg.sim.device = a.device
    for name in ('time_out', 'fallen', 'pelvis_floor', 'foot_seat', 'seat_impact'):
        if hasattr(cfg.terminations, name):
            setattr(cfg.terminations, name, None)
    f = np.array([-1., 0., -1.7]); f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1]); right /= np.linalg.norm(right)
    controller_quat = Rotation.from_matrix(np.stack((right, np.cross(f, right), f), axis=1)).as_quat()
    controller_mount = np.array([.05, .18, .55])
    if a.camera_profile == 'rear_down':
        f = np.array([0., 0., -1.]); right = np.array([0., 1., 0.])
    quat = Rotation.from_matrix(np.stack((right, np.cross(f, right), f), axis=1)).as_quat()
    mount = np.array([.05, .18, .55])
    if a.camera_profile == 'rear_down':
        mount = np.array([-.22, 0., .55])
    cfg.scene.seat_camera = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/SeatCamera',
        width=320, height=240, data_types=['rgb', 'distance_to_image_plane'], update_period=0.,
        offset=CameraCfg.OffsetCfg(pos=tuple(mount), rot=tuple(quat), convention='ros'),
        spawn=sim.PinholeCameraCfg(focal_length=18., horizontal_aperture=24., clipping_range=(.05, 4.)))
    if a.camera_profile != 'original':
        # Keep the frozen motion generator's original startup geometry input.
        # The experimental camera is an additional observer, not policy input.
        cfg.scene.controller_camera = CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/ControllerCamera',
            width=320, height=240, data_types=['distance_to_image_plane'], update_period=0.,
            offset=CameraCfg.OffsetCfg(pos=tuple(controller_mount), rot=tuple(controller_quat), convention='ros'),
            spawn=sim.PinholeCameraCfg(focal_length=18., horizontal_aperture=24., clipping_range=(.05, 4.)))
    student_path = root/'runs/deployment_status_mask_05/student.pt'
    contact_path = root/'runs/contact_estimator_02/contact.pt'
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (student_path, contact_path)}
    metadata = dict(arguments=vars(a), geometry_bank=bank, mount_position=mount.tolist(),
        camera_profile=a.camera_profile,
        controller_geometry_camera='original fixed camera; experimental view has no control authority',
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_dir.iterdir()},
        mount_quaternion_xyzw=quat.tolist(), frame_period_s=.1,
        runtime_inputs=['depth', 'intrinsics', 'fixed extrinsics', 'q', 'gyro', 'projected gravity'],
        truth_file='truth.npz: evaluation only', controller='unchanged sensor05 + contact02 actions mode',
        checkpoint_sha256=hashes, proposed_mount_not_hardware_verified=True,
        source_trajectory='free floating dynamics, original actuators/gravity, no pose teleports after reset')
    (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    online = None; online_rows = []
    if a.track_online:
        from humanoid_sitting.shared_frame import BodyModel, SharedFrameTracker, SensorFrame, transform
        from run_shared_frame import serial
        model = BodyModel(root/'runs/g1_kinematics_geometry_04/model.json')
        online = [SharedFrameTracker(model, transform(Rotation.from_quat(quat).as_matrix(), mount)) for _ in bank]
    with launch_simulation(cfg, a):
        env = gym.make('Humanoid-Sit-Learned-G1', cfg=cfg).unwrapped
        env.reset(); robot = env.scene['robot']; camera = env.scene['seat_camera']
        initial_q = robot.data.joint_pos.torch.clone()
        for _ in range(5):
            robot.set_joint_position_target(initial_q)
            env.scene.write_data_to_sim(); env.sim.step(); env.scene.update(.005)
        camera.update(.005, force_recompute=True)
        K = camera.data.intrinsic_matrices.torch.cpu().numpy()
        control_camera = camera if a.camera_profile == 'original' else env.scene['controller_camera']
        control_camera.update(.005, force_recompute=True)
        depth = control_camera.data.output['distance_to_image_plane'].torch.cpu().numpy()
        control_K = control_camera.data.intrinsic_matrices.torch.cpu().numpy()
        gravity = robot.data.projected_gravity_b.torch.cpu().numpy()
        estimates = [dict(valid=True, **estimate_body_frame(depth[i], control_K[i], controller_mount, controller_quat, gravity[i]))
                     for i in range(len(bank))]
        sensor = SimSensorInterface(robot, estimates, env.device, delay_steps=1, seed=41017)
        student, _ = DeploymentStudent.load(student_path, env.device)
        controller = SoftLandingController(student, ContactEstimator.load(contact_path, env.device), mode='actions')
        previous = robot.data.joint_pos.torch.clone(); state = None
        gyro_previous = robot.data.root_ang_vel_b.torch.cpu().numpy().copy()
        delta = np.tile(np.eye(3), (len(bank), 1, 1))
        sensed = {k: [] for k in ('depth', 'rgb', 'q', 'gravity', 'delta_rotation', 'stamp')}
        truth = {k: [] for k in ('root_pose', 'body_pose', 'seat_pose')}
        for step in range(round(a.seconds/.02)+1):
            if step % 5 == 0:
                camera.update(.02, force_recompute=True)
                sensed['depth'].append(camera.data.output['distance_to_image_plane'].torch.cpu().numpy().squeeze(-1).astype(np.float32))
                sensed['rgb'].append(camera.data.output['rgb'].torch.cpu().numpy()[..., :3].copy())
                sensed['q'].append(robot.data.joint_pos.torch.cpu().numpy().copy())
                sensed['gravity'].append(robot.data.projected_gravity_b.torch.cpu().numpy().copy())
                sensed['delta_rotation'].append(delta.copy())
                sensed['stamp'].append(step*.02)
                if online is not None:
                    for i, observer in enumerate(online):
                        stamp = step*.02
                        packet = SensorFrame(stamp, stamp, sensed['depth'][-1][i], K[i], sensed['q'][-1][i],
                            sensed['gravity'][-1][i], delta[i], stamp, stamp)
                        state_estimate = observer.update(packet)
                        online_rows.append(serial({key: val for key, val in state_estimate.items()
                            if key not in ('cloud', 'skeleton', 'proxy_world', 'seat', 'floor')} | dict(case=i)))
                delta[:] = np.eye(3)
                for k, v in record_truth(env).items():
                    truth[k].append(v)
                if step % 50 == 0:
                    print('CAPTURE', step*.02, 'seconds', flush=True)
            if step == round(a.seconds/.02):
                break
            with torch.inference_mode():
                observation = sensor.read(previous)
                targets, state = controller(observation[:, None, :], state)
                action = previous.clone()
                action[:, sensor.ids] = targets[:, 0].clamp(min=previous[:, sensor.ids]-.05, max=previous[:, sensor.ids]+.05)
                previous = action.clone()
                _, _, terminated, truncated, _ = env.step(action)
                assert not (terminated | truncated).any(), 'Unexpected reset'
            gyro = robot.data.root_ang_vel_b.torch.cpu().numpy().copy()
            delta = delta @ Rotation.from_rotvec(.5*(gyro_previous+gyro)*.02).as_matrix()
            gyro_previous = gyro
        metadata.update(joint_names=robot.joint_names, body_names=robot.body_names,
                        frames=len(sensed['stamp']), bytes_budget=600*2**20)
        (out/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
        calibration = np.eye(4)
        calibration[:3, :3] = Rotation.from_quat(quat).as_matrix(); calibration[:3, 3] = mount
        np.savez_compressed(out/'sensors.npz', **{k: np.asarray(v) for k, v in sensed.items()}, K=K,
                            T_P_C=calibration)
        np.savez_compressed(out/'truth.npz', **{k: np.asarray(v) for k, v in truth.items()})
        if online is not None:
            (out/'online_estimates.json').write_text(json.dumps(online_rows, indent=2, allow_nan=False)+'\n')
        print('CAPTURE_COMPLETE', str(out), flush=True)
        env.close()


if __name__ == '__main__':
    main()

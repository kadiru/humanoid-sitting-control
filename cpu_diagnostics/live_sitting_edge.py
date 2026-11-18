"""Observe the seat edge during the frozen standing-start sitting evaluation."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from run_repaired_sitting_ppo import HERE

SLAB_DEPTH_M = .5
CAMERA_ROUTE_TICK = 40
CAMERA_HANDOFF_CHECK_TICK = 119
CONTACT_FORE_AFT_M = -.16
CAMERA_FEATURE_SPREAD = np.array([.020, .010, 1.0])


def camera_route_feature(edge, height, pitch):
    # Express height at a nominal pelvis contact point behind the feet.
    contact_height = height-CONTACT_FORE_AFT_M*np.tan(np.deg2rad(pitch))
    return np.array([edge,contact_height,pitch])


def camera_route_measurement(history):
    history = np.asarray(history)
    if len(history)<4 or np.any(np.ptp(history,axis=0)>CAMERA_FEATURE_SPREAD):
        return None
    return camera_route_feature(*np.median(history,axis=0))


def choose_camera_route(feature, library, allow_extension=True):
    distances = np.square((library['features']-feature)/library['scales']).sum(1)
    eligible = np.flatnonzero(distances<=library['radius'])
    if 'baseline_count' in library:
        baseline = eligible[eligible<int(library['baseline_count'])]
        if baseline.size or not allow_extension:
            eligible = baseline
    if not eligible.size:
        return -1, float(distances.min())
    index = int(eligible[distances[eligible].argmin()])
    return index, float(distances[index])


def interpolate_touchdown(feature, library):
    samples = library['neighborhood_features']
    distances = np.square((samples-feature)/library['neighborhood_scales']).sum(1)
    eligible = np.flatnonzero(distances<=library['neighborhood_radius'])
    if not eligible.size:
        return None
    closest = int(eligible[distances[eligible].argmin()])
    group = library['neighborhood_groups']==library['neighborhood_groups'][closest]
    members = np.flatnonzero(group & (distances<=library['neighborhood_radius']))
    # Exact measured matches recover the verified path. Between matches,
    # interpolate trajectories from the same donor and adjustment rule.
    if distances[closest]<=1e-12:
        return closest,float(distances[closest]),library['neighborhood_paths'][closest], \
            library['neighborhood_handoff'][closest]
    weights = 1/np.square(distances[members])
    weights /= weights.sum()
    trajectory = np.einsum('n,ntj->tj',weights,library['neighborhood_paths'][members]).astype(np.float32)
    handoff = np.einsum('n,nj->j',weights,library['neighborhood_handoff'][members]).astype(np.float32)
    return closest,float(distances[closest]),trajectory,handoff


def unit(v):
    return v/np.linalg.norm(v)


def estimate_edge(model, depth, intrinsics, pelvis_from_camera, joints, gravity, plane):
    """Measure the visible front edge in the foot frame using depth and encoders."""
    if not plane[11] or plane[12]!=0:
        return None
    fk = model.fk(joints)
    feet = np.stack([fk[f'{name}_ankle_roll_link'][:3,3] for name in ('left','right')])
    center = feet.mean(0)
    up = unit(-gravity)
    forward = unit(np.cross(feet[0]-feet[1],up))
    normal = plane[2:5].astype(float)
    if normal@up<0:
        normal = -normal
    if normal@up<.68:
        return None
    seat_forward = unit(forward-(forward@normal)*normal)
    sideways = np.cross(normal,seat_forward)
    v,u = np.indices(depth.shape)
    valid = np.isfinite(depth)&(depth>.08)&(depth<3.)
    d = depth[valid]
    if len(d)<40:
        return None
    pix = np.c_[v[valid],u[valid]]
    k = intrinsics
    points = np.c_[(pix[:,1]-k[0,2])*d/k[0,0],(pix[:,0]-k[1,2])*d/k[1,1],d]
    points = points@pelvis_from_camera[:3,:3].T+pelvis_from_camera[:3,3]
    inlier = np.abs(points@plane[2:5]+plane[5]) < .003
    p, pix = points[inlier], pix[inlier]
    keep = model.self_mask(p,fk,margin=.009)
    p, pix = p[keep], pix[keep]
    relative = p-center
    x,y = relative@seat_forward, relative@sideways
    fronts, backs = [], []
    # Occluded or frame-clipped strips cannot certify a visible edge.
    for edge in np.linspace(-.16,.12,8):
        band = (y>=edge)&(y<edge+.04)
        if band.sum()<12:
            continue
        indices = np.flatnonzero(band)
        for samples,idx in ((fronts,indices[np.argmax(x[indices])]),(backs,indices[np.argmin(x[indices])])):
            py,px = pix[idx]
            if min(py,px,depth.shape[0]-1-py,depth.shape[1]-1-px)>=2:
                samples.append(idx)
    near = float(np.median(x[fronts])) if len(fronts)>=3 and np.ptp(x[fronts])<=.025 else None
    far = float(np.median(x[backs])) if len(backs)>=3 and np.ptp(x[backs])<=.025 else None
    if far is None:
        # In this camera pose a crop can look like the near edge. Require a
        # full slab span before treating the two measured boundaries as exact.
        return None
    inferred = far+SLAB_DEPTH_M
    direct = near is not None and abs(near-inferred)<.008
    front = near if direct else inferred
    selected = fronts if direct else backs
    spread = float(np.ptp(x[selected]))
    height = -((center@normal)+plane[5]*np.sign(normal@plane[2:5]))/(normal@up)
    pitch = float(np.rad2deg(np.arctan2(normal@forward,normal@up)))
    return dict(edge=front,span=spread,normal=normal,forward=seat_forward,
                feet_center=center,up=up,samples=len(selected),height=float(height),
                pitch=pitch,direct=direct)


def worker():
    sys.path.insert(0,str(HERE/'trainer'))
    import run_training
    import run_supported_postures as support
    from environment import FullBodyEnv
    from policy import Learner, command
    import torch

    def load_library(option):
        if option not in sys.argv:
            return None
        i = sys.argv.index(option)
        path = Path(sys.argv[i+1])
        del sys.argv[i:i+2]
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k] for k in z.files}

    payload = load_library('--adapt-library')
    touchdown = load_library('--touchdown-library')
    camera_touchdown = load_library('--camera-touchdown')
    feedback = None
    feedback_protected = None
    feedback_routes = None
    seat_plane_approach = '--seat-plane-approach' in sys.argv
    preload_braking = '--preload-braking' in sys.argv
    velocity_braking = '--preload-velocity-brake' in sys.argv
    housing_clearance = '--housing-clearance' in sys.argv
    loaded_hold = '--loaded-clearance-hold' in sys.argv
    shear_feedback = '--seat-shear-feedback' in sys.argv
    normal_transfer = '--normal-load-transfer' in sys.argv
    follow_descent = '--follow-normal-descent' in sys.argv
    verify_contact = '--verify-grip-contact' in sys.argv
    if verify_contact:
        assert follow_descent
        sys.argv.remove('--verify-grip-contact')
    if follow_descent:
        assert normal_transfer
        sys.argv.remove('--follow-normal-descent')
    if normal_transfer:
        assert shear_feedback
        sys.argv.remove('--normal-load-transfer')
    if shear_feedback:
        assert loaded_hold
        sys.argv.remove('--seat-shear-feedback')
        from seat_shear_feedback import SeatShearFeedback
    if loaded_hold:
        assert velocity_braking and seat_plane_approach and not housing_clearance
        sys.argv.remove('--loaded-clearance-hold')
        from loaded_clearance_hold import LoadedClearanceHold
    if housing_clearance:
        assert velocity_braking and seat_plane_approach
        sys.argv.remove('--housing-clearance')
        from housing_clearance import HousingClearance, seat_top_plane
    if seat_plane_approach:
        sys.argv.remove('--seat-plane-approach')
        from seat_plane_approach import MEASURE_UNTIL, ROLL_START, SENSOR_FRAME, UPDATE_INTERVAL, SeatPlaneApproach
    if preload_braking:
        assert seat_plane_approach
        sys.argv.remove('--preload-braking')
        from preload_brake import FIRST_TICK as BRAKE_FIRST_TICK, GAP_INTERVAL, PreloadBrake
    if velocity_braking:
        assert seat_plane_approach and not preload_braking
        sys.argv.remove('--preload-velocity-brake')
        from pelvis_velocity_brake import FIRST_TICK as VELOCITY_FIRST_TICK, INTERVAL, PelvisVelocityBrake
    if '--touchdown-feedback' in sys.argv:
        i = sys.argv.index('--touchdown-feedback')
        feedback = json.loads(Path(sys.argv[i+1]).read_text())
        del sys.argv[i:i+2]
        feedback_protected = load_library('--feedback-protected')
        feedback_routes = load_library('--feedback-routes')
        assert camera_touchdown is not None
        from touchdown_feedback import FIRST_TICK, LAST_TICK, TouchdownFeedback
    if touchdown is not None:
        assert payload is not None
    if camera_touchdown is not None:
        assert touchdown is not None

    class ObservedEnv(FullBodyEnv):
        def __init__(self,*args,**kwargs):
            self.edge_frames = []
            self.camera_samples = []
            self.touch = []
            self.live = None
            super().__init__(*args,**kwargs)
            LiveLearner.env = self

        def _observe(self,force=False):
            count = self.camera_count
            super()._observe(force)
            if self.camera_count==count:
                return
            cam = self.raw.scene['rear_camera'].data
            d = self.robot.data
            depth = cam.output['distance_to_image_plane'].torch.cpu().numpy().squeeze(-1)
            k = cam.intrinsic_matrices.torch.cpu().numpy()
            q = d.joint_pos.torch.cpu().numpy()
            gravity = d.projected_gravity_b.torch.cpu().numpy()
            tick = int(self.episode_length_buf[0])
            rs = Rotation.from_quat(d.root_quat_w.torch.cpu().numpy()).as_matrix()
            root = (d.root_pos_w.torch-self.raw.scene.env_origins).cpu().numpy()
            rows = []
            live = []
            if housing_clearance and tick <= MEASURE_UNTIL:
                self.clearance_planes = np.stack([seat_top_plane(self.model,depth[i],k[i],support.REAR_CAMERA,
                    q[i],gravity[i],self.geometry[i]) for i in range(self.num_envs)])
            for i in range(self.num_envs):
                measured = estimate_edge(self.model,depth[i],k[i],support.REAR_CAMERA,q[i],gravity[i],self.geometry[i])
                live.append(measured)
                if measured is None:
                    rows.append([np.nan]*7)
                    continue
                # Simulation coordinates are used only for this recorded audit.
                spec = self.cfg.geometry_bank[i]
                rotation = Rotation.from_quat(spec['rotation_xyzw']).as_matrix()
                actual = np.asarray(spec['top_center'])+rotation[:,0]*spec['depth']/2
                foot_edge = ((actual-root[i])@rs[i]-measured['feet_center'])@measured['forward']
                rows.append([measured['edge'],foot_edge,measured['span'],measured['samples'],
                             measured['height'],measured['pitch'],float(measured['direct'])])
            self.edge_frames.append((tick,rows))
            self.live = (tick,live)
            if tick in (0,200,400,500,600,700):
                self.camera_samples.append((tick,depth.copy(),k.copy(),q.copy(),gravity.copy(),
                                            self.geometry.copy(),rs,root))

        def step(self,*args,**kwargs):
            out = super().step(*args,**kwargs)
            if kwargs.get('record'):
                cp = self.raw.scene['pelvis_seat'].data.contact_pos_w.torch[:,0,0]
                local = (cp-self.raw.scene.env_origins).cpu().numpy()
                self.touch.append(local.copy())
            return out

        def close(self):
            ticks, rows = zip(*self.edge_frames)
            np.savez_compressed(self.out/'edge_audit.npz',ticks=ticks,values=np.asarray(rows),
                                pelvis_contact_world_local=np.asarray(self.touch),
                                channels=['estimated_edge_m','true_edge_m','strip_spread_m',
                                          'strips','estimated_height_m','estimated_pitch_deg','direct_edge'])
            if self.camera_samples:
                np.savez_compressed(self.out/'edge_depth_samples.npz',**{
                    name:np.asarray(values) for name,values in zip(
                        ('tick','depth','intrinsics','q','gravity','plane','root_rotation','root_position'),
                        zip(*self.camera_samples))})
            super().close()

    class LiveLearner(Learner):
        env = None

        def initialize_routes(self,obs):
            self.paths = torch.as_tensor(payload['paths'],device=obs.device)
            signature = obs[:,336:].cpu().numpy()
            error = np.square((signature[:,None]-payload['protected_signatures'][None])/payload['scale']).mean(-1)
            self.protected = error.min(1) <= 5e-7
            self.selected = np.full(len(obs),-1,int)
            self.selection_tick = np.full(len(obs),-1,int)
            self.features = [[] for _ in range(len(obs))]
            self.last_frame = -1
            self.debt = np.zeros(len(obs))

        def live_selection(self,tick,base):
            recent = self.env.live
            if recent is None or recent[0]==self.last_frame:
                return
            self.last_frame, frames = recent
            if tick>310:
                return
            for i,measured in enumerate(frames):
                if measured is None or self.protected[i]:
                    continue
                self.features[i].append([measured['edge'],measured['height'],measured['pitch']])
                history = np.asarray(self.features[i][-5:])
                if tick<40 or len(history)<4:
                    continue
                if np.ptp(history[:,0])>.012 or np.ptp(history[:,1])>.01 or np.ptp(history[:,2])>2.:
                    continue
                latest = np.median(history,axis=0)
                distance = np.square((payload['features']-latest)/[.012,.018,2.5]).sum(1)
                distance += .18*np.square(payload['peak_slip']/.02)
                best = int(distance.argmin())
                if distance[best]>1.:
                    continue
                current = self.selected[i]
                if current==best or (current>=0 and distance[current]<=distance[best]+.50):
                    continue
                candidate = payload['paths'][best,tick]
                previous = base[i] if current<0 else payload['paths'][current,tick]
                if np.max(np.abs(candidate-previous))>.065:
                    continue
                self.selected[i], self.selection_tick[i] = best,tick

        @torch.no_grad()
        def choose(self,obs,stochastic=True):
            result = self.live_choose(obs,stochastic)
            if touchdown is None:
                return result
            tick = int(self.env.episode_length_buf[0])
            if tick==0:
                signature = obs[:,336:].cpu().numpy()
                error = np.square((signature[:,None]-touchdown['signatures'])/touchdown['scale']).mean(-1)
                index = error.argmin(1)
                distance = error[np.arange(len(obs)),index]
                self.touchdown_index = np.where(distance<touchdown['radius'][index],index,-1)
                self.touchdown_distance = distance
                self.touchdown_active = torch.as_tensor(self.touchdown_index>=0,device=obs.device)
                self.touchdown_paths = torch.as_tensor(touchdown['paths'][index],device=obs.device)
            if tick==999:
                np.savez_compressed(self.env.out/'touchdown_selection.npz',selected=self.touchdown_index,
                                    distance=self.touchdown_distance,ids=touchdown['ids'])
            desired = result[0]
            if self.touchdown_active.any():
                delta = self.actor(self.normalize(obs))*self.residual_mask(obs)
                desired = torch.where(self.touchdown_active[:,None],self.touchdown_paths[:,tick]+delta,desired)
            if camera_touchdown is not None:
                desired = self.choose_camera_touchdown(obs,tick,desired)
            if feedback is not None:
                desired = self.apply_touchdown_feedback(obs,tick,desired)
            if seat_plane_approach:
                desired = self.apply_seat_plane_approach(obs,tick,desired)
            if preload_braking:
                desired = self.apply_preload_brake(obs,tick,desired)
            if velocity_braking:
                desired = self.apply_velocity_brake(obs,tick,desired)
            if loaded_hold:
                desired = self.apply_loaded_hold(obs,tick,desired)
            if shear_feedback:
                desired = self.apply_seat_shear(obs,tick,desired)
            return (desired,*result[1:])

        def apply_seat_shear(self,obs,tick,desired):
            if tick == MEASURE_UNTIL:
                self.shear_feedback = SeatShearFeedback(self.seat_plane,normal_transfer,follow_descent,verify_contact)
            if tick < VELOCITY_FIRST_TICK:
                return desired
            guard = self.shear_feedback
            brake = self.velocity_brake
            if brake.active.size and np.any(brake.started >= 0):
                sensor = self.env.raw.scene['pelvis_seat'].data
                normal = sensor.normal_force_matrix_w.torch.sum((1,2)).cpu().numpy()
                shear = sensor.friction_force_matrix_w.torch.sum((1,2)).cpu().numpy()
                data = self.env.robot.data
                rotation = Rotation.from_quat(data.root_quat_w.torch.cpu().numpy()).inv()
                weight = (data.body_mass.torch.sum(1)*9.81).cpu().numpy()
                braked = np.zeros(len(obs),dtype=bool)
                braked[brake.active] = brake.started >= 0
                point_w = sensor.contact_pos_w.torch[:,0,0].cpu().numpy() if verify_contact else None
                guard.update(tick,rotation.apply(normal),rotation.apply(shear),weight,braked,
                             self.loaded_hold,obs[:,SENSOR_FRAME].cpu().numpy(),desired.cpu().numpy(),
                             self.env.previous[:,self.env.ids].cpu().numpy(),brake,point_w)
            if tick == 999:
                guard.save(self.env.out/'seat_shear_feedback.npz')
            if not normal_transfer:
                return desired + torch.as_tensor(guard.delta,device=obs.device)
            rows = torch.as_tensor(np.flatnonzero(guard.latched),device=obs.device)
            if not len(rows):
                return desired
            legs = torch.as_tensor(self.seat_plane.legs,device=obs.device)
            corrected = desired.clone()
            target = torch.as_tensor((guard.anchor+guard.delta)[guard.latched],device=obs.device)
            corrected[rows[:,None],legs] = target[:,legs]
            return corrected

        def apply_loaded_hold(self,obs,tick,desired):
            if tick == MEASURE_UNTIL:
                self.loaded_hold = LoadedClearanceHold(self.seat_plane,self.env.model,self.env.metadata)
            if tick < ROLL_START:
                return desired
            guard = self.loaded_hold
            sensor = self.env.raw.scene['pelvis_seat'].data
            force = sensor.normal_force_matrix_w.torch.sum((1,2)).norm(dim=-1)
            weight = self.env.robot.data.body_mass.torch.sum(1)*9.81
            load = (force/weight).cpu().numpy()
            if np.any((load >= .05) & (guard.first_load < 0)):
                prior = self.env.previous[:,self.env.ids].cpu().numpy()
                braked = np.zeros(len(load),dtype=bool)
                braked[self.velocity_brake.active] = self.velocity_brake.started >= 0
                guard.observe(tick,load,obs[:,SENSOR_FRAME].cpu().numpy(),prior,braked)
            if tick == 999:
                guard.save(self.env.out/'loaded_clearance_hold.npz')
            if not guard.enabled.any():
                return desired
            rows = torch.as_tensor(np.flatnonzero(guard.enabled),device=obs.device)
            legs = torch.as_tensor(self.seat_plane.legs,device=obs.device)
            corrected = desired.clone()
            corrected[rows[:,None],legs] = torch.as_tensor(guard.target[guard.enabled],device=obs.device)
            return corrected

        def apply_velocity_brake(self,obs,tick,desired):
            if tick == MEASURE_UNTIL:
                self.velocity_brake = PelvisVelocityBrake(self.seat_plane)
            if tick < VELOCITY_FIRST_TICK:
                return desired
            brake = self.velocity_brake
            if tick == 999:
                brake.save(self.env.out/'velocity_brake.npz')
            if not brake.active.size:
                return desired
            rows = brake.active
            if tick % INTERVAL == 0:
                delta = brake.update(tick,obs[rows,SENSOR_FRAME].cpu().numpy(),desired[rows].cpu().numpy())
                self.velocity_delta = torch.as_tensor(delta.copy(),device=obs.device)
            corrected = desired.clone()
            corrected[rows] += self.velocity_delta
            return corrected

        def apply_preload_brake(self,obs,tick,desired):
            if tick == MEASURE_UNTIL:
                self.preload_brake = PreloadBrake(self.seat_plane)
            if tick < BRAKE_FIRST_TICK - 1:
                return desired
            brake = self.preload_brake
            if brake.active.size:
                rows = brake.active
                if tick == BRAKE_FIRST_TICK - 1:
                    self.brake_rows = torch.as_tensor(rows,device=obs.device)
                    self.brake_columns = torch.arange(len(rows),device=obs.device)
                    self.brake_history = torch.empty((1000,len(rows),desired.shape[-1]),
                                                     dtype=desired.dtype,device=obs.device)
                self.brake_history[tick] = desired[self.brake_rows]
                if tick < BRAKE_FIRST_TICK:
                    return desired
                frame = obs[rows,SENSOR_FRAME].cpu().numpy() if tick % GAP_INTERVAL == 0 else None
                clock = brake.update(tick,frame)
                index = torch.as_tensor(np.floor(clock),dtype=torch.long,device=obs.device)
                mix = torch.as_tensor(clock - np.floor(clock),dtype=desired.dtype,device=obs.device)
                corrected = desired.clone()
                corrected[self.brake_rows] = torch.lerp(self.brake_history[index,self.brake_columns],
                    self.brake_history[(index+1).clamp_max(tick),self.brake_columns],mix[:,None])
                desired = corrected
            if tick == 999:
                brake.save(self.env.out/'preload_brake.npz')
            return desired

        def apply_seat_plane_approach(self,obs,tick,desired):
            if tick == 0:
                approach_class = HousingClearance if housing_clearance else SeatPlaneApproach
                self.seat_plane = approach_class(self.env.model,self.env.metadata,len(obs))
            correction = self.seat_plane
            if tick <= MEASURE_UNTIL:
                frame_tick = self.env.live[0]
                if frame_tick == tick:
                    planes = self.env.clearance_planes if housing_clearance else self.env.geometry
                    correction.observe(frame_tick,planes,obs[:,SENSOR_FRAME].cpu().numpy())
            if tick == MEASURE_UNTIL:
                correction.freeze()
            if tick == 999:
                correction.save(self.env.out/'seat_plane_approach.npz')
            if tick < ROLL_START or not correction.active.size:
                return desired
            if tick % UPDATE_INTERVAL == 0:
                rows = correction.active
                correction.update(tick,desired[rows].cpu().numpy(),obs[rows,SENSOR_FRAME].cpu().numpy())
                self.seat_plane_delta = torch.as_tensor(correction.delta.copy(),device=obs.device)
            return desired + self.seat_plane_delta

        def apply_touchdown_feedback(self,obs,tick,desired):
            if tick==0:
                self.contact_feedback = TouchdownFeedback(self.env.model,self.env.metadata,feedback,
                                                          feedback_protected,feedback_routes)
                self.contact_feedback.reset(obs[:,336:].cpu().numpy())
            if tick==CAMERA_ROUTE_TICK:
                self.feedback_seat = np.full((len(obs),3),np.nan)
                for row,history in enumerate(self.camera_history):
                    if camera_route_measurement(history) is not None:
                        self.feedback_seat[row] = np.median(history,axis=0)
            if tick==999:
                self.contact_feedback.save(self.env.out/'touchdown_feedback.npz')
            if FIRST_TICK<=tick<=LAST_TICK and self.contact_feedback.active.size:
                delta = self.contact_feedback.step(tick,obs.cpu().numpy(),self.feedback_seat,
                                                   self.camera_index>=0,self.adjusted_approach)
                return desired+torch.as_tensor(delta,device=obs.device)
            return desired

        def choose_camera_touchdown(self,obs,tick,desired):
            if tick==0:
                signature = obs[:,336:].cpu().numpy()
                protected = ((signature[:,None]-camera_touchdown['protected'][None])/
                             camera_touchdown['signature_scale'])
                self.camera_protected = np.square(protected).mean(-1).min(1)<=5e-7
                if 'extension_protected' in camera_touchdown:
                    extension = ((signature[:,None]-camera_touchdown['extension_protected'][None])/
                                 camera_touchdown['signature_scale'])
                    self.extension_protected = np.square(extension).mean(-1).min(1)<=5e-7
                else:
                    self.extension_protected = np.zeros(len(obs),bool)
                self.has_neighborhoods = 'neighborhood_ids' in camera_touchdown
                self.feedback_route_ids = feedback_routes is not None
                self.feedback_route_start = len(camera_touchdown['ids'])
                if self.has_neighborhoods:
                    self.feedback_route_start += len(camera_touchdown['neighborhood_ids'])
                    normalized = ((signature[:,None]-camera_touchdown['neighborhood_protected'][None])/
                                  camera_touchdown['signature_scale'])
                    self.neighborhood_protected = np.square(normalized).mean(-1).min(1)<=5e-7
                self.camera_history = [[] for _ in range(len(obs))]
                self.camera_seen_tick = -1
                self.camera_index = np.full(len(obs),-1,int)
                self.camera_distance = np.full(len(obs),np.inf)
                self.camera_features = np.full((len(obs),3),np.nan)
                self.camera_handoff_error = np.full(len(obs),np.nan)
                self.adjusted_approach = np.zeros(len(obs),bool)
            if tick<=CAMERA_ROUTE_TICK and self.env.live is not None:
                observed_tick, measurements = self.env.live
                if observed_tick!=self.camera_seen_tick:
                    self.camera_seen_tick = observed_tick
                    for i,measured in enumerate(measurements):
                        if measured is not None and measured['samples']>=4 and measured['span']<=.012:
                            self.camera_history[i].append([measured['edge'],measured['height'],measured['pitch']])
            if tick==CAMERA_ROUTE_TICK:
                selected_paths = np.empty((len(obs),1000,23),dtype=np.float32)
                selected_paths.fill(0.)
                for i,history in enumerate(self.camera_history):
                    if self.camera_protected[i] or self.touchdown_index[i]>=0:
                        continue
                    feature = camera_route_measurement(history)
                    if feature is None:
                        continue
                    self.camera_features[i] = feature
                    options = self.camera_options(feature,i)
                    for index,distance,path,handoff in options:
                        self.camera_distance[i] = distance
                        error = self.camera_handoff(i,tick,path,handoff,obs.device)
                        self.camera_handoff_error[i] = error
                        if error>.002:
                            continue
                        self.camera_index[i] = index
                        self.adjusted_approach[i] = self.feedback_route_ids and index>=self.feedback_route_start
                        selected_paths[i] = path
                        break
                self.camera_active = torch.as_tensor(self.camera_index>=0,device=obs.device)
                self.camera_paths = torch.as_tensor(selected_paths,device=obs.device)
            if tick==999:
                ids = camera_touchdown['ids']
                if self.has_neighborhoods:
                    ids = np.concatenate((ids,camera_touchdown['neighborhood_ids']))
                if self.feedback_route_ids:
                    ids = np.concatenate((ids,feedback_routes['ids']))
                np.savez_compressed(self.env.out/'camera_touchdown_selection.npz',
                                    selected=self.camera_index, ids=ids,
                                    features=self.camera_features, distance=self.camera_distance,
                                    handoff_error=self.camera_handoff_error, protected=self.camera_protected,
                                    extension_protected=self.extension_protected)
            if tick<CAMERA_ROUTE_TICK or not self.camera_active.any():
                return desired
            delta = self.actor(self.normalize(obs))*self.residual_mask(obs)
            return torch.where(self.camera_active[:,None],self.camera_paths[:,tick]+delta,desired)

        def camera_options(self,feature,row):
            if self.feedback_route_ids:
                option = self.contact_feedback.approach(row,feature)
                if option is not None:
                    index,distance,path,handoff = option
                    yield self.feedback_route_start+index,distance,path,handoff
            if self.has_neighborhoods and not self.neighborhood_protected[row]:
                nearby = interpolate_touchdown(feature,camera_touchdown)
                if nearby is not None:
                    index,distance,path,handoff = nearby
                    yield len(camera_touchdown['ids'])+index,distance,path,handoff
            index,distance = choose_camera_route(feature,camera_touchdown,
                                                  allow_extension=not self.extension_protected[row])
            self.camera_distance[row] = distance
            if index>=0:
                yield index,distance,camera_touchdown['paths'][index],camera_touchdown['handoff'][index]

        def camera_handoff(self,row,tick,path,handoff,device):
            actions = torch.as_tensor(path[tick:CAMERA_HANDOFF_CHECK_TICK+1],device=device)
            restore = torch.as_tensor(handoff,device=device)
            predicted = self.env.previous[row,:23]
            for target in actions:
                predicted = command(predicted,target,self.env.limit_lower[row],self.env.limit_upper[row])
            return float((predicted-restore).abs().max())

        def live_choose(self,obs,stochastic=True):
            result = super().choose(obs,stochastic)
            if payload is None:
                return result
            assert not stochastic and self.env.args.mode=='evaluate'
            tick = int(self.env.episode_length_buf[0])
            if tick==0:
                self.initialize_routes(obs)
            if tick<=310:
                self.live_selection(tick,self.prior(obs).cpu().numpy())
            routes = self.selected>=0
            if not routes.any():
                return result
            # Slow the reference by at most 100 ms in the last 25 mm of
            # clearance if the current predicted contact is close to the edge.
            self.debt = np.maximum(0.,self.debt-.12)
            if 400<=tick<695 and self.env.live is not None:
                last, frames = self.env.live
                if tick-last<=10:
                    for i,measured in enumerate(frames):
                        if not routes[i] or measured is None:
                            continue
                        gap = self.env.geometry[i,0]
                        if not .005<gap<.025:
                            continue
                        pelvis = self.env.model.vertices['pelvis']
                        distance = pelvis@measured['normal']
                        cap = pelvis[distance<distance.min()+.004]
                        inset = measured['edge']-((cap-measured['feet_center'])@measured['forward']).max()
                        if inset<.018:
                            self.debt[i] = min(5.,self.debt[i]+.35)
            clock = np.maximum(0.,tick-self.debt)
            idx = torch.as_tensor(np.maximum(self.selected,0),device=obs.device)
            t = torch.as_tensor(clock,dtype=torch.long,device=obs.device).clamp_max(999)
            mix = torch.as_tensor(clock-np.floor(clock),dtype=obs.dtype,device=obs.device)[:,None]
            prior = torch.lerp(self.paths[idx,t],self.paths[idx,(t+1).clamp_max(999)],mix)
            amount = torch.as_tensor(np.clip((tick-self.selection_tick)/30,0,1)*routes,
                                     dtype=obs.dtype,device=obs.device)[:,None]
            delta = self.actor(self.normalize(obs))*self.residual_mask(obs)
            desired = torch.where(amount>0,torch.lerp(result[0],prior+delta,amount),result[0])
            if tick==999:
                np.savez_compressed(self.env.out/'live_selection.npz',selected=self.selected,
                                    selection_tick=self.selection_tick,protected=self.protected,
                                    donor_ids=payload['ids'])
            return (desired,*result[1:])

    run_training.FullBodyEnv = ObservedEnv
    if payload is not None:
        run_training.Learner = LiveLearner
    run_training.main()


if __name__ == '__main__':
    worker()

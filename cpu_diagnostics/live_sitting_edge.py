"""Observe the seat edge during the frozen standing-start sitting evaluation."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from run_repaired_sitting_ppo import HERE

SLAB_DEPTH_M = .5

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
    from policy import Learner
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
    if touchdown is not None:
        assert payload is not None

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
            if not self.touchdown_active.any():
                return result
            delta = self.actor(self.normalize(obs))*self.residual_mask(obs)
            desired = torch.where(self.touchdown_active[:,None],self.touchdown_paths[:,tick]+delta,result[0])
            return (desired,*result[1:])

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

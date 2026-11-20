"""Predict hip clearance and seat-patch travel in the stance frame."""
import numpy as np
from scipy.spatial.transform import Rotation

from scan_landing import scan_distance

SAMPLE = 5
TICKS = np.r_[np.arange(250, 1000, SAMPLE), 999]


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


class StanceHull:
    def __init__(self, model, joint_names):
        self.chains = []
        self.vertices = {}
        # Retain extreme vertices in directions around the calibrated convex hull.
        rays = np.array([[x,y,z] for x in range(-2,3) for y in range(-2,3)
                         for z in range(-2,3) if (x,y,z) != (0,0,0)])
        for name in ('pelvis','left_hip_pitch_link','right_hip_pitch_link'):
            hull = model.vertices[name]
            self.vertices[name] = hull[np.unique((hull@rays.T).argmin(axis=0))]
        for side in ('left','right'):
            name = side+'_ankle_roll_link'
            joints = []
            while name != 'pelvis':
                j = next(j for j in model.joints if j['body1']==[name])
                joints.append((joint_names.index(j['name']), *model.frames[j['name']],
                               np.eye(3)['XYZ'.index(j['axis'])]))
                name = j['body0'][0]
            self.chains.append(list(reversed(joints)))

    def sample(self, frames, scan):
        """Return gap to the finite front edge and a fixed pelvis-point position."""
        n = len(frames)
        feet, hips = [], []
        for chain in self.chains:
            pose = np.broadcast_to(np.eye(4),(n,4,4))
            for k,(joint,a,b,axis) in enumerate(chain):
                turn = np.tile(np.eye(4),(n,1,1))
                turn[:,:3,:3] = Rotation.from_rotvec(frames[:,joint,None]*axis).as_matrix()
                pose = pose@a@turn@b
                if k==0:hips.append(pose)
            feet.append(pose[:,:3,3])
        up = unit(-frames[:,69:72])
        forward = unit(np.cross(feet[0]-feet[1],up))
        side = np.cross(up,forward)
        center = .5*(feet[0]+feet[1])
        normal = unit(up + np.tan(np.deg2rad(scan[2]))*forward + scan[3]*side)
        tangent = unit(forward-(forward*normal).sum(1)[:,None]*normal)
        def points(name, pose=None):
            v = self.vertices[name]
            body = np.broadcast_to(v,(n,*v.shape)) if pose is None else v@pose[:,:3,:3].transpose(0,2,1)+pose[:,None,:3,3]
            rel = body-center[:,None,:]
            height = np.einsum('nvi,ni->nv',rel-scan[1]*up[:,None,:],normal)
            outside = np.einsum('nvi,ni->nv',rel,tangent)-scan[0]
            dist = np.where(outside<=0,height,np.hypot(np.maximum(height,0),outside))
            return rel,dist
        gaps = [points(s+'_hip_pitch_link',h)[1].min(1) for s,h in zip(('left','right'),hips)]
        rel,dist = points('pelvis')
        closest = dist.argmin(1)
        # Save the selected material point in body coordinates as well as the
        # stance transform. Its speed must exclude changes in the closest vertex.
        vertex = self.vertices['pelvis'][closest]
        axes = np.stack((forward,side,up),axis=1)
        normal_stance = np.einsum('nij,nj->ni',axes,normal)
        return np.column_stack((gaps[0],gaps[1],dist.min(1))), vertex, axes, center, normal_stance


def signals(hull, frames, scan, ticks):
    gap,vertex,axes,center,normal = hull.sample(frames,scan)
    material = vertex[:-1]
    pos0 = np.einsum('nij,nj->ni',axes[:-1],material-center[:-1])
    pos1 = np.einsum('nij,nj->ni',axes[1:],material-center[1:])
    velocity = (pos1-pos0)/(np.diff(ticks)[:,None]*.02)
    along = velocity - (velocity*normal[1:]).sum(1)[:,None]*normal[1:]
    return np.column_stack((gap,np.r_[0.,np.linalg.norm(along,axis=1)]))


def warns(series, kind):
    # Allow one control-sample interval of approaching motion when estimating
    # clearance. A moving material point only threatens grip near the seat.
    gap = series[:,:2].min(axis=1)
    advance = np.r_[0.,np.minimum(np.diff(gap),0.)]
    if kind==0:
        return gap+2*advance < .022
    return (series[:,2]<.024) & (series[:,3]>.012)


def risk(observed, motion, data):
    if motion<0 or not np.isfinite(observed).all():return None
    which = np.flatnonzero(data['risk_ids']==motion)
    if not len(which):return None
    distance=scan_distance(observed,data['risk_scans'][which])
    index=which[distance.argmin()]
    good=data['safe_scans'][data['safe_ids']==motion]
    if distance.min()>1.4 or distance.min()+.06 >= scan_distance(observed,good).min(initial=np.inf):
        return None
    return int(index)


def choose(observed, original, data):
    """Screen the nominal motion and rank alternative complete recordings."""
    episode=risk(observed,original,data)
    if episode is None:return original
    matches=scan_distance(observed,data['tested_scans'])
    if len(matches):
        j=matches.argmin()
        if (matches[j]<.80 and data['tested_from'][j]==original and
                matches[j]+.06<scan_distance(observed,data['safe_scans'][data['safe_ids']==original]).min(initial=np.inf)):
            return int(data['tested_to'][j])
    options=alternatives(observed,original,data)
    order=int(data['alternative_rank'])
    return options[order] if len(options)>order else original


def alternatives(observed,original,data):
    mask=data['sweep_ids']!=original
    ids=data['sweep_ids'][mask]
    f=data['sweep_scans'][mask]
    distances=scan_distance(observed,f)
    local=distances<=2.6
    if not local.any():return []
    ids=ids[local];d=distances[local];f=f[local]
    # Evaluate a conservative bound on the change of plane height across the
    # rear 18-32 cm of both legs. Both observed slopes influence hip clearance.
    ds=np.tan(np.deg2rad(observed[2]))-np.tan(np.deg2rad(f[:,2]))
    rise=observed[1]-f[:,1]+np.maximum(.18*ds,.32*ds)+.11*abs(observed[3]-f[:,3])
    trace=data['sweep_signals'][mask][local]
    edge_risk=np.maximum(observed[0]-f[:,0],0)
    shell=trace[:,:,:2].min(axis=(1,2))-rise-.25*edge_risk
    contact=trace[:,:,2]-rise[:,None]
    velocity=trace[:,:,3]
    grip=np.max(np.where(contact<.012,velocity,0.),axis=1)
    cost=d+abs(rise)/.012+1.3*np.maximum(.009-shell,0.)/.006+np.maximum(grip-.022,0.)/.035
    order=np.argsort(cost,kind='stable')
    picks=[]
    for k in order:
        p=int(ids[k])
        if p in picks or risk(observed,p,data) is not None:continue
        if any(np.abs(data['paths'][p,350:850:20]-data['paths'][j,350:850:20]).max()<.025 for j in picks):continue
        picks.append(p)
        if len(picks)==6:break
    return picks


class LiveGuard:
    def __init__(self, model, metadata, descriptors, selected, original, library):
        self.hull=StanceHull(model,metadata['joint_names'])
        self.scans=descriptors
        self.rows=[i for i,(f,p) in enumerate(zip(descriptors,selected,strict=True))
                   if p>=0 and np.isfinite(f).all() and np.any(library['sweep_ids']==p)]
        self.active=set(i for i in self.rows if selected[i]!=original[i])
        self.expected={}
        for i in self.rows:
            ids=np.flatnonzero(library['sweep_ids']==selected[i])
            j=ids[scan_distance(descriptors[i],library['sweep_scans'][ids]).argmin()]
            self.expected[i]=library['sweep_signals'][j]
        self.last={}
        self.delay=np.zeros(len(selected),dtype=int)
        self.trace=[]

    def update(self,tick,frames):
        if tick<250 or tick%SAMPLE and tick!=999:return
        j=np.searchsorted(TICKS,tick)
        for i in self.rows:
            f=frames[i]
            before,oldtick=self.last.get(i,(f,tick-SAMPLE))
            self.last[i]=(f.copy(),tick)
            pair=signals(self.hull,np.stack((before,f)),self.scans[i],np.array([oldtick,tick]))
            shell,cap,speed=pair[-1,:2].min(),pair[-1,2],pair[-1,3]
            expected=self.expected[i][j]
            housing=bool(warns(pair,0)[-1] and shell<expected[:2].min()-.006)
            grip=bool(warns(pair,1)[-1] and speed>expected[3]+.018)
            if i in self.active and (housing or grip):
                self.delay[i]=min(25,self.delay[i]+2)
            self.trace.append((tick,i,shell,cap,speed,housing,grip,self.delay[i]))

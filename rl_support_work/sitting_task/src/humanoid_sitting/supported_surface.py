"""General rigid support-plane math; no simulator state or imports."""
import numpy as np
from scipy.spatial.transform import Rotation
from .shared_frame import apply

def surface_frame(pitch_deg,yaw_deg=0.):
    return Rotation.from_euler('ZY',[yaw_deg,pitch_deg],degrees=True).as_matrix()

def signed_clearance(points,normal,point):
    return (np.asarray(points)-np.asarray(point))@np.asarray(normal)

def tangential_velocity(linear,angular,point,com,normal):
    v=linear+np.cross(angular,point-com)
    return v-(v*normal).sum(-1,keepdims=True)*normal

def surface_coordinates(points,center,rotation):
    return (np.asarray(points)-center)@rotation

def fit_observed_planes(points,normals,min_points=40):
    """Cluster measured normal/distance pairs; fit arbitrary slope, not only z."""
    points=np.asarray(points);normals=np.asarray(normals)
    if len(points)<min_points:return []
    normals=normals.copy()
    # Face each plane toward the sensor/pelvis; orientation is not a truth label.
    flip=(normals*points).sum(1)>0;normals[flip]*=-1
    remaining=np.ones(len(points),bool);planes=[]
    for _ in range(8):
        ids=np.flatnonzero(remaining)
        if len(ids)<min_points:break
        keys=np.round(np.c_[normals[ids]/.12,(-normals[ids]*points[ids]).sum(1)/.035]).astype(int)
        _,inverse,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        group=ids[inverse==count.argmax()];n=np.median(normals[group],axis=0);n/=np.linalg.norm(n)
        offset=-np.median(points[group]@n)
        use=remaining&(abs(points@n+offset)<.014)&(abs(normals@n)>.97)
        selected=points[use]
        if len(selected)<min_points:remaining[group]=False;continue
        center=selected.mean(0);_,_,vectors=np.linalg.svd(selected-center,full_matrices=False)
        n=vectors[-1]
        if np.dot(center,n)>0:n=-n
        offset=-center@n;residual=selected@n+offset
        remaining[use]=False
        if np.sqrt(np.mean(residual**2))>.008:continue
        u=vectors[0];v=np.cross(n,u);uv=(selected-center)@np.stack((u,v),1)
        span=np.ptp(uv,axis=0)
        if min(span)<.06:continue
        planes.append(dict(normal=n,offset=float(offset),center=center,points=selected,
            span=span,rms=float(np.sqrt(np.mean(residual**2))),area=float(np.prod(span))))
    return planes

def observed_support(model,K,T_P_C,depth,q,gravity,prior=None):
    """Use calibrated depth, FK and IMU only. Floor identity comes from foot FK."""
    d=np.asarray(depth).squeeze();h,w=d.shape;v,u=np.mgrid[:h,:w]
    valid=np.isfinite(d)&(d>.08)&(d<3.)
    z=np.where(valid,d,0)
    C=np.stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z),-1)
    P=apply(T_P_C,C.reshape(-1,3)).reshape(h,w,3);poses=model.fk(q)
    valid &=model.self_mask(P.reshape(-1,3),poses,margin=.009).reshape(h,w)
    dx=P[1:-1,2:]-P[1:-1,:-2];dy=P[2:,1:-1]-P[:-2,1:-1]
    n=np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    ok &=(length>1e-8)&(np.linalg.norm(dx,axis=-1)<.15)&(np.linalg.norm(dy,axis=-1)<.15)
    planes=fit_observed_planes(P[1:-1,1:-1][ok],n[ok]/length[ok,None])
    up=-np.asarray(gravity);up=up/np.linalg.norm(up)
    soles=np.concatenate([apply(poses[s+'_ankle_roll_link'],model.vertices[s+'_ankle_roll_link']) for s in ('left','right')])
    floor=float(np.min(soles@up))
    candidates=[x for x in planes if not (abs(x['normal']@up)>.97 and abs(x['center']@up-floor)<.055)]
    candidates=[x for x in candidates if x['area']>.015 and x['center'][0]<.1 and np.linalg.norm(x['center'])<1.5]
    if not candidates:return None,dict(planes=len(planes),candidate_count=0)
    # Associate only to an earlier sensor measurement, never a truth plane.
    if prior is not None and prior[11] and prior[12]<.5:
        matched=[x for x in candidates if x['normal']@prior[2:5]>.94 and abs(x['offset']-prior[5])<.08]
        if not matched:return None,dict(planes=len(planes),candidate_count=len(candidates),association_rejected=True)
        candidates=matched
    # Largest visible support patch; no expected height, slope, pose or family.
    p=max(candidates,key=lambda x:len(x['points']))
    pelvis=float((model.vertices['pelvis']@p['normal']+p['offset']).min())
    rear=np.concatenate([apply(poses[b],model.vertices[b]) for b in ('torso_link','logo_link')])
    torso=float((rear@p['normal']+p['offset']).min())
    return np.array([pelvis,torso,*p['normal'],p['offset'],*p['center'],*p['span'],1.,0.],np.float32),dict(planes=len(planes),candidate_count=len(candidates),rms=p['rms'])

class SupportTracker:
    """Propagate a measured plane using encoders and IMU between depth frames.

    Both feet are assumed stationary over the short (<=0.5s) bridge. Disagreement
    invalidates the bridge. Common-mode foot slip remains unobservable here.
    No simulator pose, surface metadata or contact label is accepted.
    """
    def __init__(self,model):
        self.model=model;self.value=None;self.feet=None;self.gravity=None
    def reset(self):self.value=None;self.feet=None;self.gravity=None
    def update(self,q,gravity,gyro,dt=.02,measurement=None):
        fk=self.model.fk(q)
        feet=np.array([fk[s+'_ankle_roll_link'][:3,3] for s in ('left','right')])
        gravity=np.asarray(gravity);gravity=gravity/np.linalg.norm(gravity)
        if self.value is not None and self.feet is not None:
            R=Rotation.from_rotvec(-np.asarray(gyro)*dt).as_matrix()
            pred=R@self.gravity;v=np.cross(pred,gravity);c=np.clip(pred@gravity,-1,1)
            V=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            R=(np.eye(3)+V+V@V/max(1e-8,1+c))@R
            shifts=feet-self.feet@R.T;t=shifts.mean(0)
            value=self.value.copy();value[2:5]=R@value[2:5];value[5]-=value[2:5]@t;value[6:9]=R@value[6:9]+t
            value[0]=np.min(self.model.vertices['pelvis']@value[2:5]+value[5])
            torso=np.concatenate([apply(fk[b],self.model.vertices[b]) for b in ('torso_link','logo_link')])
            value[1]=np.min(torso@value[2:5]+value[5]);value[12]+=dt
            value[11]=float(bool(value[11]) and value[12]<=.5 and np.linalg.norm(shifts[0]-shifts[1])<.02)
            self.value=value
        if measurement is not None:self.value=measurement.copy()
        self.feet=feet;self.gravity=gravity
        if self.value is None:
            empty=np.zeros(13,np.float32);empty[12]=1.;return empty
        return self.value.copy()

"""Rear-depth registration and a learned correction in the standing foot frame."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
from torch import nn

UNITS = np.array([.04, .04, .105])
LEG = np.array([[0,3,7,11,15,19],[1,4,8,12,16,20]])
N_TICKS = 825


def points(depth, k, mount, q, gravity, model):
    from seat_plane_approach import stance_axes
    poses = model.fk(q)
    feet = np.stack([poses[s+'_ankle_roll_link'] for s in ('left','right')])
    basis = stance_axes(feet, gravity)
    origin = feet[:,:3,3].mean(0)
    v,u = np.indices(depth.shape)
    use = np.isfinite(depth) & (depth>.075) & (depth<1.45)
    z = depth[use]
    xyz = np.stack(((u[use]+.5-k[0,2])*z/k[0,0],
                    (v[use]+.5-k[1,2])*z/k[1,1], z),axis=1)
    xyz = xyz@mount[:3,:3].T+mount[:3,3]
    foot = (xyz-origin)@basis
    keep = ((foot[:,0]<-.095)&(foot[:,0]>-.90)&(abs(foot[:,1])<.58)
            &(foot[:,2]>.19)&(foot[:,2]<.95))
    outside = model.self_mask(xyz[keep],poses,margin=.006)
    cloud = foot[keep][outside]
    if len(cloud)==0:
        return cloud
    _,idx = np.unique(np.rint(cloud/.008).astype(int),axis=0,return_index=True)
    return cloud[idx]


def transform(x, error):
    R = Rotation.from_euler('z',error[2]).as_matrix()
    return x@R.T+np.r_[error[:2],0.]


def register(ref, seen):
    if min(len(ref),len(seen))<45:
        return np.zeros(7), False
    tree = cKDTree(ref)
    best=None
    for angle in (-.12,0,.12):
        pose = np.array([0.,0.,angle])
        for _ in range(14):
            moved=transform(seen,pose)
            d,index=tree.query(moved)
            keep=d<=min(.09,np.quantile(d,.76))
            if keep.sum()<30:
                break
            a=moved[keep,:2]
            b=ref[index[keep],:2]
            A,B=a.mean(0),b.mean(0)
            cross=(a-A).T@(b-B)
            yaw=np.arctan2(cross[0,1]-cross[1,0],np.trace(cross))
            rot=Rotation.from_euler('z',yaw).as_matrix()[:2,:2]
            pose[:2]=rot@(pose[:2]-A)+B
            pose[2]+=yaw
        d,_=tree.query(transform(seen,pose))
        score=np.mean(np.minimum(d,.08)**2)
        if best is None or score<best[0]:
            best=score,pose,np.mean(d<.027),float(np.median(d))
    score,pose,inliers,median=best
    features=np.r_[pose/UNITS,inliers,median/.02,len(seen)/max(len(ref),1),np.sqrt(score)/.035]
    valid=bool(inliers>.35 and median<.045 and np.max(abs(pose/UNITS))<3.)
    return features,valid


class Alignment(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta=nn.Sequential(nn.Linear(7,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,3))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(self,features):
        return features[:,:3]+.45*self.delta(features).tanh()


def correction(pose):
    R=Rotation.from_euler('z',-pose[2]).as_matrix()
    return np.r_[(-R@np.r_[pose[:2],0.])[:2],-pose[2]]

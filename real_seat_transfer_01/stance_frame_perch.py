"""Express the perch reference and balance feedback in the supporting sole's frame."""
import numpy as np
from scipy.spatial.transform import Rotation
from rim_perch_goals import P,robot,legs,chains
from stance23_planner import forward

def features(q):
    right,_=forward(chains[1],q[legs[1]])
    R=right[:3,:3].T;root=-R@right[:3,3];st=robot.state(q,R,root)
    com=np.average(st[1],axis=0,weights=robot.mass)
    return np.r_[com[:2],R[2,:2]]
def build():
    sweep=np.load(P/'perch_command_sweep.npz')['path']
    joints=[1,4,8,12,2,5,6,9,10]
    knots=np.unique(np.r_[np.arange(0,len(sweep),8),len(sweep)-1])
    mats=[];feats=[];rot=[]
    for k in knots:
        q=np.zeros(37);q[:23]=sweep[k,23:46]
        f=features(q)
        J=[]
        for n in joints:
            h=q.copy();h[n]+=.001
            J.append((features(h)-f)/.001)
        J=np.stack(J,axis=1);weights=np.diag([1.,1.,.065,.065])
        W=weights@J
        reg=np.diag([.055,.055,.12,.073,.12,.115,.115,.14,.14])
        mats.append(np.linalg.solve(W.T@W+reg@reg,W.T@weights));feats.append(f)
        tf,_=forward(chains[1],q[legs[1]]);rot.append(tf[:3,:3])
    np.savez_compressed(P/'stance_frame_perch.npz',knots=knots,joints=joints,gains=mats,features=feats,sole_to_root=rot)
if __name__=='__main__':build()

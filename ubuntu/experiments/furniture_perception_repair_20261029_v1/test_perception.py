"""CPU contracts for camera centers, connected patches and above-pelvis tops."""
import sys
import numpy as np
from paths import ROOT,HERE,write
sys.path.insert(0,str(ROOT/'sitting_task/src'))
from perception import cloud,patches,observe

def main():
    T=np.eye(4);K=np.array([[100.,0,2.],[0,100.,2.],[0,0,1.]])
    P,valid=cloud(np.ones((4,4)),K,T)
    np.testing.assert_allclose(P[0,0],[-.015,-.015,1],atol=1e-15)
    # Two separated coplanar supports must never become one phantom broad seat.
    y,x=np.mgrid[:30,:70];P=np.stack((x*.01,y*.01,np.zeros(x.shape)),axis=-1)
    N=np.zeros_like(P);N[...,2]=1;mask=((x<25)|(x>44))
    ps=patches(P,N,mask,np.array([0.,0.,1.]))
    assert len(ps)==2 and all(p['area']<.08 for p in ps)
    assert all(np.ptp(p['pixels'][:,1])<30 for p in ps)
    class Model:
        vertices={k:np.array([[-.05,-.05,-.02],[.05,.05,.02]]) for k in ['left_ankle_roll_link','right_ankle_roll_link','pelvis','torso_link','logo_link']}
        def fk(self,q):
            result={k:np.eye(4) for k in self.vertices}
            for k in ['left_ankle_roll_link','right_ankle_roll_link']:result[k][2,3]=-.7
            return result
        def self_mask(self,p,fk,margin):return np.ones(len(p),bool)
    # A horizontal tabletop above the pelvis remains visible from the higher
    # camera; orienting toward the pelvis incorrectly flips its normal down.
    T=np.eye(4);T[:3,:3]=[[0,2**-.5,-2**-.5],[1,0,0],[0,-2**-.5,-2**-.5]];T[:3,3]=[-.07,0,.38]
    K=np.array([[73.84615,0,64],[0,73.84615,48],[0,0,1.]])
    v,u=np.mgrid[:96,:128];C=np.stack(((u+.5-64)/K[0,0],(v+.5-48)/K[1,1],np.ones(u.shape)),-1)
    rays=C@T[:3,:3].T;depth=(.2-T[2,3])/rays[...,2]
    point=T[:3,3]+rays*depth[...,None]
    depth=np.where((depth>.08)&(point[...,0]<-.2)&(point[...,0]>-.6)&(abs(point[...,1])<.3),depth,np.inf)
    value,info=observe(Model(),K,T,depth,np.zeros(23),np.array([0.,0.,-1.]),'top')
    assert value is not None
    np.testing.assert_allclose(value[2:5],[0,0,1],atol=1e-6);assert abs(value[5]+.2)<1e-6
    write(HERE/'cpu_contract_checks.json',dict(passed=True,pixel_centers=True,separate_coplanar_arms=True,above_pelvis_top=True,simulation_steps=0))
    print('Camera-center, disconnected-patch and high-tabletop tests passed.')
if __name__=='__main__':main()

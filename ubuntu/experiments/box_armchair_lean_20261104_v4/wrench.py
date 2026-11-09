"""Sagittal contact-force observer from paired motor torques and foot-anchored FK.
No contact sensor, world pose or scene identifier enters feedback.
"""
import numpy as np
from humanoid_sitting.shared_frame import apply

class WrenchObserver:
    def __init__(self,plan,nodes):
        self.plan=plan
        rt,poses=plan.pose(np.array(nodes[-1]['q']))
        points=np.concatenate([apply(poses[b],v) for b,v in plan.trunk.items()])
        marker=points[(apply(rt,points)@plan.n).argmin()]
        self.tables=[]
        for node in nodes:
            values=[]
            q=np.array(node['q'])
            for delta in np.eye(3)*.0002:
                rows=[]
                for x in [q+delta,q-delta]:
                    root,f=plan.pose(x)
                    com=apply(root,np.average([apply(f[b],c) for b,c in plan.local],weights=plan.mass,axis=0))
                    pt=apply(root,marker)
                    rows.append(np.r_[plan.weight*com[2],pt@plan.n,pt@plan.up])
                values.append((rows[0]-rows[1])/.0004)
            self.tables.append(np.array(values))
        self.bias=np.zeros(3)
    def estimate(self,i,torques):
        tab=self.tables[i]
        residual=tab[:,0]-torques-self.bias
        return np.linalg.lstsq(tab[:,1:],residual,rcond=None)[0]

"""Exact finite lower-envelope rolling of the actual pelvis collider vertices."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

class Rolling:
    def __init__(self,vertices,normal,angles,start,limit):
        vertices=np.asarray(vertices,float);self.v=vertices[ConvexHull(vertices).vertices]
        self.n=np.asarray(normal,float);self.angles=np.asarray(angles,float).copy();self.start=float(start);self.limit=float(limit)
        assert start<=limit
        rx=Rotation.from_euler('x',angles[0]).as_matrix();rz=Rotation.from_euler('z',angles[2]).as_matrix()
        v=self.v@rx.T;n=rz.T@self.n
        a=n[0]*v[:,0]+n[2]*v[:,2];b=n[0]*v[:,2]-n[2]*v[:,0];c=n[1]*v[:,1]
        self.coeff=np.c_[a,b,c]
        i,j=np.triu_indices(len(v),1);A=a[i]-a[j];B=b[i]-b[j];C=c[i]-c[j]
        mag=np.hypot(A,B);valid=(mag>1e-13)&(abs(C)<=mag)
        phase=np.arctan2(B[valid],A[valid]);delta=np.arccos(np.clip(-C[valid]/mag[valid],-1,1))
        candidates=np.r_[start,limit,*[phase+sign*delta+shift for sign in [-1,1] for shift in [-2*np.pi,0,2*np.pi]]]
        candidates=np.unique(candidates[(candidates>=start)&(candidates<=limit)])
        if len(candidates)>100000:raise ValueError('ROLLING_EVENT_RESOURCE_CAP')
        segments=[]
        for left,right in zip(candidates,candidates[1:]):
            mid=(left+right)/2;idx=int(np.argmin(self.coeff@np.array([np.cos(mid),np.sin(mid),1.])))
            if segments and segments[-1][2]==idx:segments[-1][1]=float(right)
            else:segments.append([float(left),float(right),idx])
        if len(segments)>64:raise ValueError('ROLLING_ACTIVE_EVENT_CAP')
        self.segments=segments
    def rotation(self,theta):
        a=self.angles.copy();a[1]=theta;return Rotation.from_euler('xyz',a).as_matrix()
    def translation(self,theta,initial):
        if theta<self.start-1e-12 or theta>self.limit+1e-12:raise ValueError('ROLLING_ANGLE_OUTSIDE_FROZEN_INTERVAL')
        p=np.asarray(initial,float).copy()
        for lo,hi,idx in self.segments:
            if theta<=lo:break
            end=min(theta,hi);v=self.v[idx]
            p+=self.rotation(lo)@v-self.rotation(end)@v
            if theta<=hi:break
        return p
    def audit(self,initial,offset):
        gap=[];fixed=[];continuity=[]
        for lo,hi,idx in self.segments:
            pa=self.translation(lo,initial);ra=self.rotation(lo);point=pa+ra@self.v[idx]
            for theta in [lo,(lo+hi)/2,hi]:
                p=self.translation(theta,initial);r=self.rotation(theta)
                fixed.append(float(np.linalg.norm(p+r@self.v[idx]-point)))
                gap.append(float(((self.v@r.T+p)@self.n+offset).min()))
            if hi<self.limit:
                continuity.append(float(abs((self.v@self.rotation(hi).T+self.translation(hi,initial))@self.n+offset).min()))
        return dict(passed=min(gap)>=-1e-9 and max(abs(np.asarray(gap)))<=1e-9 and max(fixed)<=1e-9,
          minimum_body_plane_gap_m=min(gap),maximum_contact_gap_m=max(abs(np.asarray(gap))),maximum_material_point_drift_m=max(fixed),
          active_segments=len(self.segments),convex_vertices=len(self.v),scope='Piecewise rigid rolling on exact local plane; no dynamics or hardware-success claim.')

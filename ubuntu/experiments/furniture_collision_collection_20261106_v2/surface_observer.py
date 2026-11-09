"""Anchor a fresh camera patch using encoders/gravity and fixed foot geometry."""
import numpy as np
from scipy.spatial import ConvexHull,Delaunay
from humanoid_sitting.shared_frame import apply,level_rotation,transform
from contact_surface import associate


class Observer:
    def __init__(self,intent):
        self.intent=intent;self.last=self.raw_last=None;self.anchors=None

    def __call__(self,model,K,T,depth,q,gravity,prior=None):
        from perception import observe
        value,info=observe(model,K,T,depth,q,gravity,self.intent,prior)
        self.raw_last=(value,info)
        if value is None:self.last=(value,info);return self.last
        try:
            value,local,report=self.refine(model,q,gravity,value,np.array(info['selected_points']))
            info=dict(info,surface_anchor=report,selected_points=local.tolist())
            # Pixel correspondence is retained by matching an exact subset.
            if self.intent=='seat':
                source=np.array(self.raw_last[1]['selected_points'])
                wanted={tuple(x) for x in local};ids=[i for i,x in enumerate(source) if tuple(x) in wanted]
                info['selected_pixels']=np.array(self.raw_last[1]['selected_pixels'])[ids].tolist()
        except (ValueError,AssertionError) as e:
            value=None;info=dict(info,anchored_patch_rejected=repr(e))
        self.last=(value,info);return self.last

    def refine(self,model,q,gravity,value,points):
        R=level_rotation(np.asarray(gravity,float).copy());fk=model.fk(q)
        if self.anchors is None:
            out,local,report,cloud,height,target=associate(model,q,gravity,value,points,self.intent)
            self.anchors=[(transform(R)@fk[s+'_ankle_roll_link'])[:3,3].copy() for s in ['left','right']]
            self.target=R@out[6:9]
            if self.intent=='seat':self.boundary=ConvexHull((local@R.T)[:,:2]).equations.copy()
            return out,local,dict(report,anchored_to_sensor_feet=True)
        origin=np.mean([anchor-R@fk[s+'_ankle_roll_link'][:3,3] for s,anchor in zip(['left','right'],self.anchors)],axis=0)
        cloud=points@R.T+origin
        if self.intent=='seat':
            inside=np.all(cloud[:,:2]@self.boundary[:,:2].T+self.boundary[:,2]<=1e-6,axis=1)
            local=points[inside]
            if len(local)<30:raise ValueError('ANCHORED_PATCH_NOT_VISIBLE')
            center=local.mean(0);_,_,axes=np.linalg.svd(local-center,full_matrices=False)
            normal=axes[-1]
            if normal@(-np.asarray(gravity))<0:normal=-normal
            offset=float(-normal@center);error=float(abs(local@normal+offset).max())
            if error>.002 or (R@normal)[2]<=.97:raise ValueError('ANCHORED_PLANE_INCONSISTENT')
            span=np.ptp((local-center)@axes[:2].T,axis=0)
        else:
            triangulation=Delaunay(cloud[:,:2]);simplex=int(triangulation.find_simplex(self.target[:2]))
            if simplex<0:raise ValueError('ANCHORED_CONTACT_TARGET_NOT_VISIBLE')
            tri=cloud[triangulation.simplices[simplex]]
            if max(np.linalg.norm(tri[(i+1)%3]-tri[i]) for i in range(3))>.05:raise ValueError('UNOBSERVED_CONTACT_GAP')
            n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n)
            if n[2]<0:n=-n
            if n[2]<=.97:raise ValueError('ANCHORED_CONTACT_SLOPE')
            d=float(-n@tri[0]);normal=R.T@n;offset=d+n@origin
            target=self.target.copy();target[2]=-(d+n[:2]@target[:2])/n[2]
            center=R.T@(target-origin);span=np.ptp(cloud[:,:2],axis=0);local=points;error=0.
        rear=np.concatenate([apply(fk[b],model.vertices[b]) for b in ['torso_link','logo_link']])
        out=np.array([(np.asarray(model.vertices['pelvis'])@normal+offset).min(),(rear@normal+offset).min(),
            *normal,offset,*center,*span,1.,0.],np.float32)
        return out,local,dict(anchored_to_sensor_feet=True,maximum_plane_residual_m=error,
                              actor_truth_input=False,visible_points=len(local))

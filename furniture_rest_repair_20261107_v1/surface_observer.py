"""Fresh-plane tracking on the same initially observed connected facet.
The contact patch stays fixed. A visible neighbouring part of that measured
facet may refresh its plane while the contact patch is hidden by the robot.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from scipy.ndimage import label
from scipy.optimize import linprog
from humanoid_sitting.shared_frame import level_rotation,apply
path=Path(__file__).resolve().parent.parent/'furniture_camera_coverage_20261106_v1/surface_observer.py'
spec=importlib.util.spec_from_file_location('original_anchor',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def candidates(model,K,T,depth,q,g):
    from perception import cloud
    P,valid=cloud(depth,K,T);h,w=valid.shape
    valid &= model.self_mask(P.reshape(-1,3),model.fk(q),margin=.009).reshape(h,w)
    dx=P[1:-1,2:]-P[1:-1,:-2];dy=P[2:,1:-1]-P[:-2,1:-1]
    n=np.cross(dx,dy);length=np.linalg.norm(n,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    ok &= (length>1e-8)&(np.linalg.norm(dx,axis=-1)<.15)&(np.linalg.norm(dy,axis=-1)<.15)
    n/=np.maximum(length[...,None],1e-30);points=P[1:-1,1:-1]
    n[(n*(points-T[:3,3])).sum(-1)>0]*=-1
    up=-np.asarray(g,float);up/=np.linalg.norm(up);ok &= n@up>.75
    return points,ok


class Observer(base.Observer):
    def __call__(self,model,K,T,depth,q,gravity,prior=None):
        if self.intent!='seat':return super().__call__(model,K,T,depth,q,gravity,prior)
        if self.anchors is None:
            value,info=super().__call__(model,K,T,depth,q,gravity,prior)
            if value is None:return value,info
            points,ok=candidates(model,K,T,depth,q,gravity)
            ok &= np.abs(points@value[2:5]+value[5])<=.002
            components,_=label(ok,np.ones((3,3),int))
            pixels=np.array(info['selected_pixels'],int)-1
            pixels=pixels[(pixels[:,0]>=0)&(pixels[:,1]>=0)&(pixels[:,0]<ok.shape[0])&(pixels[:,1]<ok.shape[1])]
            ids,counts=np.unique(components[tuple(pixels.T)],return_counts=True);counts[ids==0]=0
            if not len(ids) or counts.max()<30:raise ValueError('INITIAL_FACET_CONNECTION_MISSING')
            mask=components==ids[counts.argmax()];R=level_rotation(np.asarray(gravity,float).copy())
            self.facet=points[mask]@R.T;self.triangulation=Delaunay(self.facet[:,:2])
            tri=self.facet[self.triangulation.simplices]
            self.triangle_valid=np.max(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2),axis=1)<=.05
            self.initial_n=R@value[2:5];self.initial_d=float(value[5]);self.contact_boundary=self.boundary.copy();self.contact_span=np.asarray(value[9:11],np.float32).copy();value=np.asarray(value,np.float32)
            info=dict(info,tracking_facet_points=len(self.facet),contact_boundary_unchanged=True)
            self.last=(value,info);return self.last
        points,ok=candidates(model,K,T,depth,q,gravity);pixels=np.argwhere(ok)+1;points=points[ok]
        R=level_rotation(np.asarray(gravity,float).copy());fk=model.fk(q)
        origin=np.mean([a-R@fk[s+'_ankle_roll_link'][:3,3] for s,a in zip(['left','right'],self.anchors)],axis=0)
        world=points@R.T+origin;simplex=self.triangulation.find_simplex(world[:,:2]);inside=simplex>=0
        inside &= self.triangle_valid[np.maximum(simplex,0)]
        # Association tolerance remains original10mm; final measured-plane residual is still2mm.
        bary=self.triangulation.transform[np.maximum(simplex,0),:2]@(world[:,:2]-self.triangulation.transform[np.maximum(simplex,0),2])[...,None]
        weights=np.c_[bary[...,0],1-bary[...,0].sum(1)]
        predicted_z=(weights*self.facet[self.triangulation.simplices[np.maximum(simplex,0)],2]).sum(1)
        # Match the measured piecewise surface, not any point within a10mm slab.
        inside &= abs(world[:,2]-predicted_z)<=.002
        local=points[inside];pixels=pixels[inside]
        info=dict(intent=self.intent,actor_truth_input=False,tracking_facet_points=len(self.facet),contact_boundary_unchanged=bool(np.array_equal(self.boundary,self.contact_boundary)),selected_points=local.tolist(),selected_pixels=pixels.tolist())
        try:
            if len(local)<30:raise ValueError('SAME_MEASURED_FACET_NOT_VISIBLE')
            level_points=local@R.T;center=level_points.mean(0);v=level_points-center
            design=np.c_[v[:,:2],np.ones(len(v))]
            fit=linprog([0.,0.,0.,1.],A_ub=np.vstack([np.c_[design,-np.ones(len(v))],np.c_[-design,-np.ones(len(v))]]),b_ub=np.r_[v[:,2],-v[:,2]],bounds=[(None,None)]*3+[(0,None)],method='highs',options={'time_limit':1.})
            if not fit.success:raise ValueError('FRESH_MINIMAX_PLANE_UNCERTIFIED')
            nw=np.r_[-fit.x[:2],1.];scale=np.linalg.norm(nw);nw/=scale
            dw=-nw@center-fit.x[2]/scale;n=R.T@nw;d=float(dw)
            error=float(np.max(abs(local@n+d)))
            if error>.002 or (R@n)[2]<=.97:raise ValueError('CURRENT_FACET_PLANE_INCONSISTENT')
            if np.linalg.norm(R@n-self.initial_n)>.03 or abs(d-(R@n)@origin-self.initial_d)>.01:raise ValueError('FACET_ASSOCIATION_FAILED')
            # Fixed contact target, not the moving centroid of the visible patch.
            target=self.target.copy();nw=R@n;dw=d-nw@origin
            target[2]=-(dw+nw[:2]@target[:2])/nw[2];contact=R.T@(target-origin)
            rear=np.concatenate([apply(fk[b],model.vertices[b]) for b in ['torso_link','logo_link']])
            value=np.array([(np.asarray(model.vertices['pelvis'])@n+d).min(),(rear@n+d).min(),*n,d,*contact,*self.contact_span,1.,0.],np.float32)
            info['surface_anchor']=dict(maximum_plane_residual_m=error,visible_points=len(local),fixed_contact_target=True,fit_objective='maximum_distance_matching_original_acceptance',fit_status=int(fit.status))
        except (ValueError,AssertionError) as e:
            value=None;info['anchored_patch_rejected']=repr(e)
        self.last=(value,info);return self.last

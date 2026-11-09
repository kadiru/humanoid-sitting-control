"""Refresh a fixed contact patch from its measured rigid surface map.

The original contact patch stays planar within2mm. Its visible neighbours are
a measured triangle surface, not a new infinite plane. Current depth estimates
the normal translation that fixed-foot odometry cannot observe during rocking.
No simulator pose, object ID/mesh or contact truth enters this observer.
"""
import importlib.util
from pathlib import Path
import numpy as np
from humanoid_sitting.shared_frame import level_rotation,apply
path=Path(__file__).resolve().parents[2]/'furniture_rest_repair_20261107_v1/surface_observer.py'
spec=importlib.util.spec_from_file_location('initial_facet',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class Observer(base.Observer):
    def __call__(self,model,K,T,depth,q,gravity,prior=None):
        if self.intent!='seat' or self.anchors is None:
            return super().__call__(model,K,T,depth,q,gravity,prior)
        points,ok=base.candidates(model,K,T,depth,q,gravity)
        pixels=np.argwhere(ok)+1;points=points[ok]
        R=level_rotation(np.asarray(gravity,float).copy());fk=model.fk(q)
        origins=np.array([a-R@fk[s+'_ankle_roll_link'][:3,3] for s,a in zip(['left','right'],self.anchors)])
        origin=origins.mean(0);world=points@R.T+origin
        simplex=self.triangulation.find_simplex(world[:,:2]);inside=simplex>=0
        inside &= self.triangle_valid[np.maximum(simplex,0)]
        info=dict(actor_truth_input=False,registration='one_normal_translation_to_initial_measured_height_field',contact_boundary_unchanged=bool(np.array_equal(self.boundary,self.contact_boundary)))
        try:
            if np.linalg.norm(origins[0]-origins[1])>.003:raise ValueError('BILATERAL_ODOMETRY_INCONSISTENT')
            ids=np.flatnonzero(inside)
            if len(ids)<30:raise ValueError('INITIAL_MEASURED_FACET_NOT_VISIBLE')
            bary=(self.triangulation.transform[simplex[ids],:2]@(world[ids,:2]-self.triangulation.transform[simplex[ids],2])[...,None])[...,0]
            weights=np.c_[bary,1-bary.sum(1)]
            predicted=(weights*self.facet[self.triangulation.simplices[simplex[ids]],2]).sum(1)
            dz=float(np.median(predicted-world[ids,2]))
            if abs(dz)>.01:raise ValueError('ORIGINAL10MM_ASSOCIATION_LIMIT')
            residual=world[ids,2]+dz-predicted
            match=abs(residual)<=.002
            if match.sum()<30:raise ValueError('CURRENT_MEASURED_MAP_MATCH_TOO_SMALL')
            # A majority must agree: do not make an arbitrary tiny fragment fresh.
            if match.sum()*2<len(ids):raise ValueError('CURRENT_MAP_MAJORITY_DISAGREES')
            origin[2]+=dz
            n=R.T@self.initial_n;d=float(self.initial_d+self.initial_n@origin)
            if np.linalg.norm(n) < .999999 or (R@n)[2]<=.97:raise ValueError('CONTACT_NORMAL_INVALID')
            contact=R.T@(self.target-origin)
            rear=np.concatenate([apply(fk[b],model.vertices[b]) for b in ['torso_link','logo_link']])
            value=np.array([(np.asarray(model.vertices['pelvis'])@n+d).min(),(rear@n+d).min(),*n,d,*contact,*self.contact_span,1.,0.],np.float32)
            selected=ids[match]
            info.update(selected_points=points[selected].tolist(),selected_pixels=pixels[selected].tolist(),
                surface_anchor=dict(current_map_residual_m=float(abs(residual[match]).max()),fresh_matches=len(selected),projected_candidates=len(ids),normal_translation_m=dz,
                    original_contact_plane_preserved=True,current_visible_neighbours_not_claimed_planar=True,estimated_origin=origin.tolist()))
        except (ValueError,AssertionError) as e:
            value=None;info['anchored_patch_rejected']=repr(e)
        self.last=(value,info);return self.last

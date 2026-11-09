"""Short, uncertainty-bounded transport of an observed seat patch using foot FK.

P is pelvis; E is an arbitrary local gravity/yaw frame. IMU supplies orientation;
stationary-foot FK supplies translation for at most 100 ms. Common foot motion
is unobservable: the full inferred displacement is charged to the error budget.
No simulator state or privileged contact input is used.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from .shared_frame import apply,level_rotation
from .support_landing_geometry import extract_scene


class ContinuousPatchFeatures:
    def __init__(self,tracker):
        self.tracker=tracker;self.stamp=None;self.yaw=0.;self.anchor=None

    @staticmethod
    def unavailable(reason,**detail):
        v=np.zeros(10,dtype=np.float32);v[-1]=1.
        return v,dict(usable=False,reason=reason,**detail)

    def update(self,stamp,result,q,gravity,gyro,depth=None):
        t=float(stamp);g=np.asarray(gravity);w=np.asarray(gyro)
        if not np.isfinite(q).all() or not np.isfinite(g).all() or not np.isfinite(w).all() or np.linalg.norm(g)<.5:
            self.anchor=None;self.stamp=None;return self.unavailable('invalid_proprioception')
        if self.stamp is not None:
            dt=t-self.stamp
            if dt<=0:raise ValueError('Non-increasing feature timestamp')
            if dt>.12:self.anchor=None
            self.yaw+=float(w@(-g/np.linalg.norm(g)))*dt
        self.stamp=t
        R=Rotation.from_rotvec([0,0,self.yaw]).as_matrix()@level_rotation(g)
        model=self.tracker.model;poses=model.fk(q)
        feet=np.array([poses[s+'_ankle_roll_link'][:3,3]@R.T for s in ('left','right')])
        direct=result['valid'] and result['source'] not in ('kinematic_prediction','seat_from_observed_floor','unavailable')
        observed=None
        if depth is not None and direct:
            scene=extract_scene(model,self.tracker.K,self.tracker.T_P_C,depth,q,g)
            matches=[]
            for p in scene['planes']:
                n=scene['R'].T@p['normal'];c=float((model.vertices['pelvis']@n+p['offset']).min())
                if np.linalg.norm(n-np.asarray(result['normal_P']))<1e-6 and abs(c-result['clearance_m'])<1e-6:
                    matches.append(p)
            if len(matches)==1:observed=matches[0]['points']@scene['R']
        if observed is not None:
            probe_ids=np.unique([observed[:,k].argmin() for k in (0,1)]+[observed[:,k].argmax() for k in (0,1)])
            self.anchor=dict(t=t,points_E=observed@R.T,feet=feet.copy(),points_P=observed.copy(),
                budget=float(result['allowance_m']),probe_ids=probe_ids)
            points=observed;budget=self.anchor['budget'];age=0.;translation=np.zeros(3);kind='current_observed_patch'
        else:
            if not result['valid']:return self.unavailable('distance_unavailable')
            if self.anchor is None:return self.unavailable('no_observed_patch')
            a=self.anchor;age=t-a['t']
            if age>.1+1e-7:return self.unavailable('observed_patch_expired',patch_age_s=age)
            translations=a['feet']-feet;translation=translations.mean(0)
            disagreement=float(np.linalg.norm(translations[0]-translations[1]))
            points=(a['points_E']-translation)@R
            n=np.asarray(result['normal_P']);offset=result['clearance_m']-float((model.vertices['pelvis']@n).min())
            normal_correction=points@n+offset
            budget=max(float(result['allowance_m']),a['budget']+float(np.linalg.norm(translation))+.020*age+
                       disagreement/2+float(abs(normal_correction).max()))
            if disagreement>=.015 or budget>=.02:
                return self.unavailable('patch_motion_budget',allowance_m=budget,foot_disagreement_m=disagreement)
            # A currently observed floor can correct the vertical component of
            # this short rigid-map transport; its entire correction is charged.
            points=points-normal_correction[:,None]*n
            kind='kinematic_transport_of_observed_patch'
        values=np.array([result['clearance_m'],*result['normal_P'],*points[:,:2].mean(0),
                         *np.ptp(points[:,:2],axis=0),1.,age],dtype=np.float32)
        a=self.anchor;ids=a['probe_ids']
        return values,dict(usable=True,reason=kind,patch_age_s=age,allowance_m=budget,
            anchor_stamp=a['t'],anchor_probes_P=a['points_P'][ids].tolist(),current_probes_P=points[ids].tolist(),
            inferred_translation_E=translation.tolist(),foot_support_assumption='stationary feet; common-mode motion charged in full')

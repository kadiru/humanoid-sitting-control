"""Measured-load-gated command-path follower. No simulator or file access.

The nominal path is the caller's current-input model plan. The frozen parent
remains an additive residual outside this object. A stopped path holds its
nominal command; it never consumes contact labels or pretends assumed loads
were measured. This is a research controller, not a contact certificate.
"""
import numpy as np


def support_limit(geometry_valid, geometry_age, plane_height_error,
                  plane_slope_error_deg, foot_conditions, foot_wrench,
                  weight, normal_z, joint_speed, gyro_speed, near_plane,
                  patch_margin, housing_clearance):
    """Conservative permission to model unloading, independent of outcome labels.

    A 0.02 BW reserve exceeds the <0.005 BW static estimator discrepancy from
    preparation127. Dynamics/unmodeled contacts can still bias this observer.
    These are controller guards, not replacements for the physical evaluator.
    """
    vals=np.r_[geometry_age,plane_height_error,plane_slope_error_deg,
               foot_conditions,np.asarray(foot_wrench).ravel(),weight,normal_z,
               joint_speed,gyro_speed,near_plane,patch_margin,housing_clearance]
    finite=bool(np.isfinite(vals).all())
    fresh=bool(finite and geometry_valid and 0<=geometry_age<=.5)
    consistent=bool(fresh and abs(plane_height_error)<=.003 and abs(plane_slope_error_deg)<=.5)
    estimate_valid=bool(consistent and weight>0 and .8<=normal_z<=1 and
                        max(foot_conditions)<200 and min(np.asarray(foot_wrench)[:,1])>=.05*weight)
    estimated_vertical=1-float(np.asarray(foot_wrench)[:,1].sum())/weight if estimate_valid else None
    quiet=bool(finite and joint_speed<=.5 and gyro_speed<=.2)
    contact_geometry=bool(finite and abs(near_plane)<=.003 and patch_margin>=.015 and housing_clearance>=.003)
    allowed=bool(estimate_valid and quiet and contact_geometry)
    max_normal=float(np.clip((estimated_vertical-.02)*normal_z,0,.1)) if allowed else 0.
    return dict(fresh=fresh,plane_consistent=consistent,estimate_valid=estimate_valid,
                quiet=quiet,contact_geometry=contact_geometry,estimated_vertical_seat_bw=estimated_vertical,
                max_normal_bw=max_normal,estimated_not_certified=True)


class GatedCommandPath:
    def __init__(self,plan,model):
        assert plan['model_passed']
        self.path=plan['path'];self.ids=np.asarray(plan['moving_ids'])
        self.Q=np.asarray([e['load_checks'][0]['command'] for e in self.path])
        self.loads=np.asarray([e['assumed_seat_normal_fraction'] for e in self.path])
        assert np.all(self.loads[:81]==0) and np.all(np.diff(self.loads)>=0)
        self.arc=np.r_[0,np.cumsum(np.max(abs(np.diff(self.Q,axis=0)),axis=1))]
        assert np.all(np.diff(self.arc)>0), 'Zero command arc needs explicit handling'
        limits={j['name']:np.deg2rad(j['limits']) for j in model.joints if j['type']=='PhysicsRevoluteJoint'}
        self.lo=np.array([limits[model.names[j]][0]+.03 for j in self.ids],np.float32)
        self.hi=np.array([limits[model.names[j]][1]-.03 for j in self.ids],np.float32)
        self.alpha=np.float32(.1);self.cap=np.float32(.003)
        self.nominal=self.Q[0].astype(np.float32);self.position=0.
        self.last_target=self.nominal.copy();self.last_info={}

    def at(self,s):
        k=max(0,min(int(np.searchsorted(self.arc,s,side='right')-1),len(self.Q)-2))
        f=(s-self.arc[k])/(self.arc[k+1]-self.arc[k])
        return (self.Q[k]+f*(self.Q[k+1]-self.Q[k])).astype(np.float32)

    def advance(self,progress_allowed,max_normal_bw):
        # This is a nominal reference state. Do not replace it with the actual
        # parent-perturbed command, which would silently cancel the parent.
        prev=self.nominal.copy();start=self.position
        max_normal_bw=float(np.clip(max_normal_bw,0,.1))
        limit=float(np.interp(max_normal_bw,self.loads[80:],self.arc[80:]))
        if not progress_allowed or self.position>=limit:
            # Holding nominal command also preserves the external parent's
            # ordinary additive response through the inherited filter.
            self.last_target=prev.copy()
            self.last_info=dict(start=start,end=start,limit=limit,paused=True,max_normal_bw=max_normal_bw)
            return self.last_target.copy()
        def valid(candidate):
            goal=self.at(candidate);raw=prev[self.ids]+(goal[self.ids]-prev[self.ids])/self.alpha
            return bool(np.all(raw>=self.lo) and np.all(raw<=self.hi) and
                        np.all(abs(self.alpha*(raw-prev[self.ids]))<=self.cap)),goal,raw
        edge=int(np.searchsorted(self.arc,start,side='right'));safe=start
        while edge<len(self.arc):
            proposed=min(float(self.arc[edge]),limit)
            okay,_,_=valid(proposed)
            if okay:
                safe=proposed
                if proposed==limit:break
                edge+=1
            else:
                low,high=safe,proposed
                for _ in range(45):
                    mid=(low+high)/2
                    if valid(mid)[0]:low=mid
                    else:high=mid
                safe=low;break
        assert safe>start+1e-12,(start,safe)
        okay,goal,raw=valid(safe);assert okay
        target=prev.copy();target[self.ids]=raw
        self.nominal[self.ids]=prev[self.ids]+np.clip(self.alpha*(np.clip(raw,self.lo,self.hi)-prev[self.ids]),-self.cap,self.cap)
        self.position=safe;self.last_target=target
        self.last_info=dict(start=start,end=safe,limit=limit,paused=False,max_normal_bw=max_normal_bw,
                            demanded_normal_bw=float(np.interp(safe,self.arc,self.loads)))
        return target.copy()

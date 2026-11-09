"""Sensor-only landing geometry; T_P_C maps optical coordinates to pelvis.

No simulator or evaluation-label imports. Distances are metres; IMU gravity
sets vertical, encoder FK supplies foot and pelvis collision geometry.
"""
import numpy as np
from scipy.spatial import ConvexHull
from .shared_frame import (BodyModel, apply, level_rotation, horizontal_planes,
                           associate_seat_plane, plane_clearance_allowance)


def foot_geometry(model, poses, R):
    bodies=[side+'_ankle_roll_link' for side in ('left','right')]
    vertices=[apply(poses[name],model.vertices[name])@R.T for name in bodies]
    return np.array([v[:,2].min() for v in vertices]),np.array([poses[name][:3,3]@R.T for name in bodies])


def extract_scene(model,K,T_P_C,depth,q,gravity,poses=None,R=None):
    depth=np.asarray(depth).squeeze();h,w=depth.shape;v,u=np.mgrid[:h,:w]
    valid=np.isfinite(depth)&(depth>.08)&(depth<4.);raw_count=int(valid.sum())
    z=np.where(valid,depth,0.)
    C=np.stack(((u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z),-1)
    P=apply(T_P_C,C.reshape(-1,3)).reshape(h,w,3)
    poses=model.fk(q) if poses is None else poses
    valid &= model.self_mask(P.reshape(-1,3),poses).reshape(h,w)
    R=level_rotation(gravity) if R is None else R;L=P@R.T
    dx=L[1:-1,2:]-L[1:-1,:-2];dy=L[2:,1:-1]-L[:-2,1:-1]
    normals=np.cross(dx,dy);length=np.linalg.norm(normals,axis=-1)
    ok=(valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]&
        (length>1e-8)&(np.linalg.norm(dx,axis=-1)<.12)&(np.linalg.norm(dy,axis=-1)<.12))
    normals=normals[ok]/length[ok,None];points=L[1:-1,1:-1][ok]
    planes=horizontal_planes(points,normals)
    diagnostic=dict(raw_depth_points=raw_count,after_self_mask=int(valid.sum()),
        usable_normal_points=int(ok.sum()),horizontal_normal_points=int((abs(normals[:,2])>.96).sum()),
        planes=[dict(height_m=p['height'],points=len(p['points']),rms_m=p['rms'],
                     area_m2=float(ConvexHull(p['points'][:,:2]).volume) if np.linalg.matrix_rank(p['points'][:,:2]-p['points'][:,:2].mean(0))==2 else 0.) for p in planes])
    return dict(R=R,poses=poses,planes=planes,points=points,normals=normals,diagnostic=diagnostic)


def original_measurement(model,scene,seat_height):
    plane,diagnostic=associate_seat_plane(scene['planes'],seat_height,0.,False)
    values=np.zeros(10,dtype=np.float32);values[-1]=1.
    if plane is None:return values,diagnostic,None
    R=scene['R'];vertices=model.vertices['pelvis']@R.T
    allowance=plane_clearance_allowance(plane,vertices)
    if not allowance['conditioned']:return values,diagnostic|dict(usable=False,allowance=allowance),None
    n=R.T@plane['normal'];clearance=float((model.vertices['pelvis']@n+plane['offset']).min())
    points_P=plane['points']@R;center=points_P[:,:2].mean(0);span=np.ptp(points_P[:,:2],axis=0)
    values[:]=[clearance,*n,*center,*span,1.,0.]
    return values,diagnostic|dict(usable=True,allowance=allowance,points=len(points_P),clearance_m=clearance),plane


class LandingGeometryTracker:
    """Track local clearance, not hidden seat bounds or global robot pose.

    Startup floor/seat separation must be measured from a depth image. Foot FK
    transfers that measured vertical reference as the legs bend. Current depth
    can then identify a seat without demanding a large floor patch in the same
    image. Predictions are explicitly marked and expire; two-foot agreement
    cannot observe common-mode foot motion or certify real contact.
    """
    def __init__(self,model_path,K,T_P_C,seat_height):
        self.model=BodyModel(model_path);self.K=np.asarray(K);self.T_P_C=np.asarray(T_P_C)
        self.seat_height=float(seat_height);self.reset()

    def reset(self):
        self.stamp=None;self.floor_anchor=None;self.seat_anchor=None
        self.last_depth_stamp=None;self.visual_floor=None;self.seat_last_seen=None

    def initialize_from_measured_floor(self,stamp,q,gravity,floor_height):
        """floor_height is the startup depth fit in the leveled pelvis frame."""
        self.reset();R=level_rotation(gravity);poses=self.model.fk(q)
        soles,_=foot_geometry(self.model,poses,R)
        offsets=float(floor_height)-soles
        if not np.isfinite(offsets).all() or abs(offsets).max()>.035:
            return False
        self.floor_anchor=dict(stamp=float(stamp),height=float(floor_height),soles=soles.copy())
        self.visual_floor=dict(stamp=float(stamp),height=float(floor_height));self.seat_last_seen=float(stamp)
        self.seat_anchor=dict(stamp=float(stamp),height=float(floor_height)+self.seat_height,
            normal=np.array([0.,0.,1.]),offset=-float(floor_height)-self.seat_height,
            soles=soles.copy(),allowance=.012)
        return True

    def _output(self,stamp,R,soles,source,valid,reason,normal=None,offset=None,allowance=None,**diagnostic):
        result=dict(stamp=float(stamp),valid=bool(valid),source=source,reason=reason,
            clearance_m=None,normal_P=None,allowance_m=allowance,
            allowance_kind='Engineering error budget conditional on stationary support; not a calibrated confidence interval',
            foot_support_assumption='Two feet remain on the same rigid floor; common-mode foot motion is unobservable from encoders alone',
            **diagnostic)
        if normal is not None and offset is not None:
            n=R.T@normal
            result.update(clearance_m=float((self.model.vertices['pelvis']@n+offset).min()),normal_P=n.tolist())
        return result

    def update(self,stamp,q,gravity,gyro,depth=None):
        stamp=float(stamp);q=np.asarray(q);gravity=np.asarray(gravity);gyro=np.asarray(gyro)
        if not (np.isfinite(q).all() and np.isfinite(gravity).all() and np.isfinite(gyro).all()) or np.linalg.norm(gravity)<.5:
            self.reset();return dict(stamp=stamp,valid=False,source='unavailable',reason='invalid_proprioception',clearance_m=None)
        if self.stamp is not None and stamp<=self.stamp:
            raise ValueError('Non-increasing sensor timestamp; reset explicitly')
        gap=self.stamp is not None and stamp-self.stamp>.12
        if gap:self.reset()
        self.stamp=stamp
        R=level_rotation(gravity);poses=self.model.fk(q);soles,feet=foot_geometry(self.model,poses,R)
        vertices=self.model.vertices['pelvis']@R.T
        tilt=float(np.arccos(np.clip(-gravity[2]/np.linalg.norm(gravity),-1,1)))
        motion_ok=tilt<np.deg2rad(45) and np.linalg.norm(gyro)<2.
        floor_pred=None;foot_disagreement=None
        if self.floor_anchor is not None:
            shifts=soles-self.floor_anchor['soles'];foot_disagreement=float(np.ptp(shifts))
            if foot_disagreement<.015 and motion_ok:
                floor_pred=self.floor_anchor['height']+float(shifts.mean())
        scene=None;legacy_reason='no_new_depth';selected=None;source=None;budget=None
        if depth is not None:
            self.last_depth_stamp=stamp
            scene=extract_scene(self.model,self.K,self.T_P_C,depth,q,gravity,poses,R)
            old,info,plane=original_measurement(self.model,scene,self.seat_height)
            legacy_reason='accepted' if old[8] else ('plane_extrapolation_unconstrained' if 'allowance' in info else info['basis'])
            if plane is not None:
                selected=plane;source='current_floor_and_seat';budget=info['allowance']['allowance_m']
                # Same observed floor used by the original association rule.
                floors=[p for p in scene['planes'] if abs(p['height']-plane['height']+self.seat_height)<.015 and
                    len(p['points'])>=150 and p['rms']<.008 and
                    np.linalg.matrix_rank(p['points'][:,:2]-p['points'][:,:2].mean(0))==2 and
                    ConvexHull(p['points'][:,:2]).volume>.15]
                if len(floors)==1:
                    self.visual_floor=dict(stamp=stamp,height=floors[0]['height'])
                if len(floors)==1 and abs(soles-floors[0]['height']).max()<.035:
                    self.floor_anchor=dict(stamp=stamp,height=floors[0]['height'],soles=soles.copy())
                    floor_pred=floors[0]['height'];foot_disagreement=0.
            if selected is None and floor_pred is not None:
                # Anchor identity with independently observed floor/seat height
                # and FK, rather than treating the lowest visible plane as floor.
                candidates=[p for p in scene['planes'] if len(p['points'])>=60 and p['rms']<.008 and
                    p['normal'][2]>np.cos(np.deg2rad(2.)) and
                    abs(p['height']-floor_pred-self.seat_height)<.018]
                conditioned=[(p,plane_clearance_allowance(p,vertices)) for p in candidates]
                conditioned=[(p,b) for p,b in conditioned if b['conditioned']]
                if len(conditioned)==1:
                    selected,b=conditioned[0];source='depth_with_kinematic_floor';budget=b['allowance_m']
            if selected is None and self.seat_anchor is not None and stamp-self.seat_anchor['stamp']<=.2+1e-7:
                # A recently observed seat keeps its identity even when feet lift
                # and the floor is occluded. Do not relabel the lowest plane.
                previous=self.seat_anchor
                candidates=[p for p in scene['planes'] if len(p['points'])>=60 and p['rms']<.008 and
                    p['normal'][2]>np.cos(np.deg2rad(2.)) and
                    abs(p['offset']-previous['offset'])<.025]
                conditioned=[(p,plane_clearance_allowance(p,vertices)) for p in candidates]
                conditioned=[(p,b) for p,b in conditioned if b['conditioned']]
                if len(conditioned)==1:
                    selected,b=conditioned[0];source='depth_with_temporal_identity';budget=b['allowance_m']
            # If the seat is occluded, a current floor plane can transfer the
            # previously observed rigid floor-to-seat relationship. Feet need
            # not stay loaded. The seat itself must have been seen recently.
            floor_now=None
            if self.visual_floor is not None and stamp-self.visual_floor['stamp']<=.2+1e-7:
                floors=[p for p in scene['planes'] if len(p['points'])>=150 and p['rms']<.008 and
                    p['normal'][2]>np.cos(np.deg2rad(2.)) and
                    abs(p['height']-self.visual_floor['height'])<.03 and
                    np.linalg.matrix_rank(p['points'][:,:2]-p['points'][:,:2].mean(0))==2 and
                    ConvexHull(p['points'][:,:2]).volume>.15]
                if len(floors)==1:
                    floor_now=floors[0];self.visual_floor=dict(stamp=stamp,height=floor_now['height'])
            if selected is None and floor_now is not None and self.seat_last_seen is not None:
                age=stamp-self.seat_last_seen
                # Horizontal-floor model: gravity supplies its normal, so
                # fit the offset rather than extrapolating an unconstrained
                # plane tilt from a floor patch beyond the pelvis. Account for
                # observed tilt plus a 0.25-degree IMU/calibration allowance.
                b=plane_clearance_allowance(floor_now,vertices)
                angle=float(np.arccos(np.clip(floor_now['normal'][2],-1,1)))
                center=np.median(floor_now['points'][:,:2],axis=0)
                lever=float(np.linalg.norm(vertices[:,:2]-center,axis=1).max())
                mapped_budget=(.008+3*floor_now['rms']/np.sqrt(max(1,b['effective_samples']))+
                    np.sin(angle+np.deg2rad(.25))*lever+.005+.005*age)
                if age<=.6+1e-7 and angle<np.deg2rad(.5) and b['effective_samples']>=6 and mapped_budget<.02:
                    selected=dict(height=floor_now['height']+self.seat_height,
                        normal=np.array([0.,0.,1.]),offset=-floor_now['height']-self.seat_height)
                    source='seat_from_observed_floor';budget=mapped_budget
        if selected is not None:
            if source!='seat_from_observed_floor':self.seat_last_seen=stamp
            self.seat_anchor=dict(stamp=stamp,height=selected['height'],normal=selected['normal'].copy(),
                offset=float(selected['offset']),soles=soles.copy(),allowance=float(budget))
            return self._output(stamp,R,soles,source,True,'rigid_map_from_current_floor' if source=='seat_from_observed_floor' else 'measured_seat_plane',selected['normal'],selected['offset'],budget,
                measurement_age_s=0.,seat_last_seen_age_s=stamp-self.seat_last_seen,foot_disagreement_m=foot_disagreement,legacy_reason=legacy_reason,
                scene=scene['diagnostic'] if scene else None)
        if self.seat_anchor is None:
            return self._output(stamp,R,soles,'unavailable',False,'no_measured_reference',
                legacy_reason=legacy_reason,foot_disagreement_m=foot_disagreement)
        anchor=self.seat_anchor;age=stamp-anchor['stamp'];shifts=soles-anchor['soles']
        disagreement=float(np.ptp(shifts));delta=float(shifts.mean())
        # Charge the entire inferred displacement to the budget as a guard
        # against common-mode foot lift: bilateral agreement is not contact.
        budget=anchor['allowance']+.020*age+disagreement/2+abs(delta)
        valid=(age<=.10+1e-7 and disagreement<.015 and motion_ok and floor_pred is not None and budget<.02 and self.seat_last_seen is not None and stamp-self.seat_last_seen<=.6+1e-7)
        reason=('kinematic_bridge' if valid else 'prediction_expired' if age>.10+1e-7 else
                'foot_support_inconsistent' if floor_pred is None or disagreement>=.015 else 'uncertainty_or_motion_limit')
        # Expired predictions are not exposed as usable measurements.
        return self._output(stamp,R,soles,'kinematic_prediction' if valid else 'unavailable',valid,reason,
            anchor['normal'] if valid else None,anchor['offset']-anchor['normal'][2]*delta if valid else None,budget,
            measurement_age_s=age,foot_disagreement_m=disagreement,legacy_reason=legacy_reason,
            scene=scene['diagnostic'] if scene else None)

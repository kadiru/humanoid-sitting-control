"""Current measured plane plus encoder/IMU fixed-foot clearance guard.

No live truth, saved trajectory, contact label or simulator API. Both-foot
anchoring is an explicit estimator assumption; common foot slip is unobservable
here and must still fail the independent physical evaluation.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import apply,transform
from plan_seat_load_targets import clipped_vertices

class MeasuredGuard:
    def __init__(self,model,plan,initial_geometry):
        self.model=model;self.feet=plan['path'][0]['foot_targets']
        root=np.array(plan['path'][0]['root']);R=Rotation.from_quat(root[3:]).as_matrix()
        raw_n=R@np.array(initial_geometry[2:5]);raw_n/=np.linalg.norm(raw_n)
        raw_d=float(initial_geometry[5]-raw_n@root[:3])
        n=np.array(plan['normal']);self.normal_correction=Rotation.align_vectors([n],[raw_n])[0].as_matrix()
        self.offset_correction=float(plan['offset']-raw_d)
        self.height=plan['height'];self.normal=n

    def observe(self,q,gravity,geometry):
        q=np.asarray(q);g=np.asarray(gravity);v=np.asarray(geometry)
        if (not np.isfinite(np.r_[q,g,v]).all() or not v[11] or not 0<=v[12]<=.5
                or np.linalg.norm(g)<.5 or np.linalg.norm(v[2:5])<.5):
            return dict(available=False,reason='missing_stale_or_nonfinite_current_geometry')
        fk=self.model.fk(q);rots=[]
        for side,f in self.feet.items():
            rots.append(np.array(f['rotation'])@fk[side+'_ankle_roll_link'][:3,:3].T)
        R=Rotation.from_matrix(np.array(rots)).mean().as_matrix()
        roots=np.array([np.array(f['position'])-R@fk[s+'_ankle_roll_link'][:3,3] for s,f in self.feet.items()])
        root=roots.mean(0);T=transform(R,root)
        anchor_error=float(np.linalg.norm(roots[0]-roots[1]))
        orientation_error=float(np.linalg.norm(Rotation.from_matrix(rots[0].T@rots[1]).as_rotvec()))
        imu_error=float(np.arccos(np.clip((-R.T[:,2])@(g/np.linalg.norm(g)),-1,1)))
        raw_n=R@v[2:5];raw_n/=np.linalg.norm(raw_n)
        raw_d=float(v[5]-raw_n@root)
        n=self.normal_correction@raw_n;d=raw_d+self.offset_correction
        if n[2]<=.8:return dict(available=False,reason='plane_outside_near_horizontal_contract')
        center=np.array([-.4,0,(.4*n[0]-d)/n[2]])
        x=np.cross([0,1,0],n);x/=np.linalg.norm(x);y=np.cross(n,x);seat=np.stack([x,y,n],axis=1)
        pelvis=apply(T,self.model.vertices['pelvis']);depth=pelvis@n+d;patch=depth<depth.min()+.002
        local=(pelvis-center)@seat
        pm=float(min(.25-abs(local[patch,0]).max(),.45-abs(local[patch,1]).max()))
        clear=[]
        for b,vertices in self.model.vertices.items():
            if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
            vertices=(apply(T@fk[b],vertices)-center)@seat;vertices[:,0]-=.4
            vertices=clipped_vertices(vertices)
            if len(vertices):clear.append((float(vertices[:,2].min()),b))
        nearest=min(clear) if clear else (1.,'none_over_finite_seat')
        return dict(available=True,root_estimate=root.tolist(),rotation_estimate=R.tolist(),
                    anchor_disagreement_m=anchor_error,foot_orientation_disagreement_rad=orientation_error,imu_disagreement_rad=imu_error,
                    anchor_valid=bool(anchor_error<=.003 and orientation_error<=np.deg2rad(.5) and imu_error<=np.deg2rad(.5)),
                    plane_height_error_m=float(center[2]-self.height),
                    plane_slope_error_deg=float(np.rad2deg(np.arccos(np.clip(n@self.normal,-1,1)))),normal_z=float(n[2]),
                    pelvis_gap_m=float(depth.min()),patch_margin_m=pm,housing_clearance_m=nearest[0],nearest_body=nearest[1],
                    current_geometry_age_s=float(v[12]),fixed_feet_assumption=True)

"""Newton–Euler extension of the spatial leg-wrench observer.

Inputs are joint encoders/rates/accelerations, motor-axis effort and pelvis
IMU signals. Coordinates are pelvis body frame, then the existing gravity
frame. Root translational velocity and simulator contact forces are not inputs.
This candidate does not supply an uncertainty interval or admit release.
"""
import numpy as np
from observer import SpatialLoadObserver
from humanoid_sitting.shared_frame import level_rotation


def shift_acceleration(acceleration, omega, alpha, displacement):
    """Rigid-body point acceleration, all vectors in the same frame."""
    return (acceleration + np.cross(alpha, displacement)
            + np.cross(omega, np.cross(omega, displacement)))


class DynamicLoadObserver(SpatialLoadObserver):
    def __init__(self, model, inertia, inertia_body, armature):
        super().__init__(model, inertia)
        self.inertia_body = {name: np.asarray(value,float) for name,value in inertia_body.items()}
        self.armature = np.asarray(armature,float)
        if self.armature.shape != (len(model.names),) or (self.armature < 0).any():
            raise ValueError('INVALID_ARMATURE')
        for name in self.mass:
            value=self.inertia_body[name]
            if value.shape!=(3,3) or not np.isfinite(value).all() or not np.allclose(value,value.T):
                raise ValueError('INVALID_INERTIA: '+name)
            if np.linalg.eigvalsh(value).min() < -1e-10:
                raise ValueError('NEGATIVE_INERTIA: '+name)

    def motion(self, q, qd, qdd, gravity, gyro, angular_acceleration, origin_acceleration):
        """Propagate measured pelvis motion and encoder motion through the tree."""
        rotation=level_rotation(np.array(gravity,float,copy=True))
        fk=self.model.fk(q)
        omega={'pelvis':rotation@gyro}
        alpha={'pelvis':rotation@angular_acceleration}
        acceleration={'pelvis':rotation@origin_acceleration}
        speed=dict(zip(self.model.names,qd)); change=dict(zip(self.model.names,qdd))
        pending=self.model.joints.copy()
        while pending:
            remaining=[]
            for joint in pending:
                if not joint['body0'] or joint['body0'][0] not in omega:
                    remaining.append(joint);continue
                parent,child=joint['body0'][0],joint['body1'][0]
                frame=fk[parent]@self.model.frames[joint['name']][0]
                axis=(rotation@frame[:3,'XYZ'.index(joint['axis'])]
                      if joint['name'] in speed else np.zeros(3))
                rate=speed.get(joint['name'],0.)
                omega[child]=omega[parent]+axis*rate
                alpha[child]=(alpha[parent]+axis*change.get(joint['name'],0.)
                              +np.cross(omega[parent],axis)*rate)
                rp=rotation@(frame[:3,3]-fk[parent][:3,3])
                rc=rotation@(frame[:3,3]-fk[child][:3,3])
                at_joint=shift_acceleration(acceleration[parent],omega[parent],alpha[parent],rp)
                acceleration[child]=shift_acceleration(at_joint,omega[child],alpha[child],-rc)
            if len(remaining)==len(pending):raise ValueError('DISCONNECTED_MOTION_TREE')
            pending=remaining
        com_acceleration={}
        inertia_world={}
        for name in self.mass:
            orient=rotation@fk[name][:3,:3]
            com_acceleration[name]=shift_acceleration(acceleration[name],omega[name],alpha[name],
                                                      orient@self.com[name])
            inertia_world[name]=orient@self.inertia_body[name]@orient.T
        return fk,rotation,omega,alpha,com_acceleration,inertia_world

    def estimate_dynamic(self, q, torque, gravity, qd, qdd, gyro, angular_acceleration,
                         origin_acceleration):
        inputs=[np.asarray(x,float) for x in
                (q,torque,gravity,qd,qdd,gyro,angular_acceleration,origin_acceleration)]
        q,torque,gravity,qd,qdd,gyro,angular_acceleration,origin_acceleration=inputs
        shapes=[(len(self.model.names),)]*2+[(3,)]+[(len(self.model.names),)]*2+[(3,)]*3
        if any(x.shape!=shape or not np.isfinite(x).all() for x,shape in zip(inputs,shapes)):
            return None, {'reason':'INVALID_SENSOR_INPUT'}
        if np.linalg.norm(gravity)==0:return None,{'reason':'INVALID_GRAVITY'}
        jac,grav=self.equations(q,gravity)
        conditions=np.array([np.linalg.cond(a) for a in jac])
        if not np.isfinite(conditions).all() or conditions.max()>=200:
            return None,{'reason':'CONDITION_LIMIT'}
        fk,rotation,omega,alpha,accel,inertias=self.motion(
            q,qd,qdd,gravity,gyro,angular_acceleration,origin_acceleration)
        indices=[[self.model.names.index(n) for n in side] for side in self.sides]
        dynamic=np.zeros((2,6))
        for side,names in enumerate(self.sides):
            for k,name in enumerate(names):
                joint=self.joints[name]
                frame=fk[joint['body0'][0]]@self.model.frames[name][0]
                axis=rotation@frame[:3,'XYZ'.index(joint['axis'])]
                origin=rotation@frame[:3,3]
                for body in self.desc[name]:
                    if body not in self.mass:continue
                    position=rotation@(fk[body][:3,3]+fk[body][:3,:3]@self.com[body])
                    force=self.mass[body]*accel[body]
                    moment=inertias[body]@alpha[body]+np.cross(omega[body],inertias[body]@omega[body])
                    dynamic[side,k]+=axis@(np.cross(position-origin,force)+moment)
                idx=indices[side][k]
                dynamic[side,k]+=self.armature[idx]*qdd[idx]
        foot=np.stack([np.linalg.solve(a,b) for a,b in zip(jac,grav+dynamic-torque[indices])])
        inertial_force=sum((self.mass[b]*accel[b] for b in self.mass),np.zeros(3))
        rest=np.array([0.,0.,self.weight])+inertial_force-foot[:,:3].sum(0)
        result=np.r_[rest,foot.ravel()]
        if not np.isfinite(result).all():return None,{'reason':'NONFINITE_OUTPUT'}
        return result,dict(condition=conditions,dynamic_joint_torque=dynamic,
                           total_inertial_force=inertial_force,com_acceleration=accel,
                           uncertainty_admitted=False,release_admitted=False)

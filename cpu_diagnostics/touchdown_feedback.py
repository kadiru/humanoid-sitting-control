"""Use rear depth and foot kinematics to correct the final seat approach."""

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_sitting_repairs import leg_fk, smoothstep


FIRST_TICK = 300
LAST_TICK = 745
CONTROL_DT = .02
SENSOR_FRAME = slice(222,333)


class TouchdownFeedback:
    def __init__(self,model,metadata,config,protected,routes=None):
        self.feet,self.legs = leg_fk(model,metadata['joint_names'])
        self.pelvis = model.vertices['pelvis']
        self.profiles = config['profiles']
        self.profile_by_row = config.get('profile_by_row')
        self.protected = protected
        self.routes = routes

    def reset(self,signature):
        count = len(signature)
        active = np.ones(count,bool)
        if self.protected is not None:
            err = (signature[:,None]-self.protected['signature'])/self.protected['scale']
            active &= np.square(err).mean(-1).min(1)>5e-7
        indices = self.profile_by_row if self.profile_by_row is not None else [0]*count
        assert len(indices)==count
        self.active = np.asarray([i for i in range(count) if active[i] and indices[i]>=0],dtype=int)
        self.configuration = [self.profiles[i] if i>=0 else None for i in indices]
        self.prev = np.full((count,2),np.nan)
        self.rate = np.zeros((count,2))
        self.offset = np.zeros((count,2))
        self.inset_start = np.full(count,-1)
        self.contact_start = np.full(count,-1)
        self.map = [None]*count
        self.map_tick = np.full(count,-1000)
        self.samples = []

    def approach(self,row,feature):
        if self.routes is None or not np.any(self.active==row):
            return None
        profile = self.configuration[row].get('route_set')
        if profile is None:
            return None
        library = self.routes
        distance = np.square((library['features']-feature)/library['scale']).sum(1)
        index = int(distance.argmin())
        if distance[index]>library['radius'][index]:
            return None
        return (index,float(distance[index]),library['paths'][profile,index],library['handoff'][profile,index])

    def measure(self,frame,seat):
        # The newest observation frame contains joint angles and projected
        # gravity. Both ankles define the foot frame used by rear depth.
        up = -frame[69:72].astype(float)
        up /= np.linalg.norm(up)
        feet = self.feet(frame[:23])
        center = feet[:,:3,3].mean(0)
        forward = np.cross(feet[0,:3,3]-feet[1,:3,3],up)
        forward /= np.linalg.norm(forward)
        edge,height,pitch = seat
        rad = np.deg2rad(pitch)
        tangent = np.cos(rad)*forward-np.sin(rad)*up
        normal = np.sin(rad)*forward+np.cos(rad)*up
        verts = self.pelvis-center
        distance = verts@normal
        nearest = distance.min()
        weights = np.exp(-(distance-nearest)/.001)
        contact_tangent = float(np.average(verts@tangent,weights=weights))
        plane_gap = float(nearest-height*np.cos(rad))
        return feet,tangent,normal,np.array([contact_tangent,plane_gap]),edge-contact_tangent

    def foot_translation(self,joints,feet):
        """Map a millimetre-scale pelvis shift to leg angles with fixed feet."""
        jacobian = np.empty((12,len(self.legs)))
        for col,joint in enumerate(self.legs):
            offset = joints.copy()
            offset[joint] += .0005
            predicted = self.feet(offset)
            dp = predicted[:,:3,3]-feet[:,:3,3]
            dr = Rotation.from_matrix(predicted[:,:3,:3]@feet[:,:3,:3].transpose(0,2,1)).as_rotvec()
            jacobian[:,col] = np.c_[dp,.18*dr].ravel()/.0005
        goal = np.zeros((2,6,3))
        goal[:,:3,:] = -np.eye(3)
        inverse = np.linalg.solve(jacobian.T@jacobian+.0002*np.eye(len(self.legs)),jacobian.T)
        return inverse@goal.reshape(12,3)

    def edge_offset(self,i,tick,gap,inset,speed,pitch,has_route,profile):
        if 'edge_bias_m' not in profile:
            return None
        shallow = pitch < 9.
        steep_fallback = pitch >= 16. and not has_route
        projected_inset = inset-.18*speed
        if self.inset_start[i]<0 and (shallow or steep_fallback):
            if gap<.052 and projected_inset<(.015 if shallow else .023):
                self.inset_start[i] = tick
        start = self.inset_start[i]
        if start<0:
            return np.zeros(2) if shallow else None
        if gap<=-.002 and self.contact_start[i]<0:
            self.contact_start[i] = tick
        settle = self.contact_start[i]
        release = smoothstep((tick-settle-30)/60) if settle>=0 else 0.
        # Avoid feeding the impact velocity back into the command. Settle
        # behind the edge before loading it, and keep the bias through impact.
        amount = smoothstep((tick-start)/40)*(1-release)*smoothstep((LAST_TICK-tick)/40)
        bias = profile['edge_bias_m'][0 if shallow else 1]
        return np.array([-bias*amount,0.])

    def step(self,tick,obs,seats,has_route,adjusted):
        corrections = np.zeros((len(obs),23),dtype=np.float32)
        if tick<FIRST_TICK or tick>LAST_TICK or not self.active.size:
            return corrections
        trace = np.full((len(obs),7),np.nan,dtype=np.float32)
        for i in self.active:
            if adjusted[i] or not np.isfinite(seats[i]).all():
                continue
            frame = obs[i,SENSOR_FRAME]
            feet,tangent,normal,position,inset = self.measure(frame,seats[i])
            if np.isfinite(self.prev[i]).all():
                velocity = np.clip((position-self.prev[i])/CONTROL_DT,-.15,.15)
                self.rate[i] += .35*(velocity-self.rate[i])
            self.prev[i] = position
            profile = self.configuration[i]
            if profile.get('preserve_shallow') and seats[i,2]<9.:
                continue
            speed,closing = self.rate[i]
            gap = position[1]
            phase = smoothstep((.058-gap)/.035)*smoothstep((LAST_TICK-tick)/45)
            edge_need = max(0.,profile['inset']-(inset-.08*speed))
            edge_need *= smoothstep((gap+.009)/.016)
            tangential = -profile['edge_gain']*edge_need-profile['damping']*speed
            upward = profile['normal_damping']*max(0.,-.012-closing)+profile['edge_lift']*edge_need
            requested = phase*np.clip([tangential,upward],[-.014,0.],[.009,.007])
            edge_offset = self.edge_offset(i,tick,gap,inset,speed,seats[i,2],has_route[i],profile)
            if edge_offset is not None:
                requested = edge_offset
            self.offset[i] += np.clip(requested-self.offset[i],-.0012,.0012)
            if np.max(np.abs(self.offset[i]))>1e-6:
                if tick-self.map_tick[i]>=10:
                    self.map[i] = self.foot_translation(frame[:23].astype(float),feet)
                    self.map_tick[i] = tick
                displacement = self.offset[i,0]*tangent+self.offset[i,1]*normal
                joint_delta = self.map[i]@displacement
                peak = max(.07,float(np.max(np.abs(joint_delta))))
                corrections[i,self.legs] = joint_delta*.07/peak
            trace[i] = [inset,gap,speed,closing,*self.offset[i],np.max(np.abs(corrections[i]))]
        self.samples.append((tick,trace))
        return corrections

    def save(self,path):
        ticks,trace = zip(*self.samples) if self.samples else ([],[])
        np.savez_compressed(path,ticks=ticks,values=np.asarray(trace),active=self.active,
            columns=['inset_m','gap_m','tangent_m_s','normal_m_s','offset_t_m','offset_n_m','max_residual_rad'])

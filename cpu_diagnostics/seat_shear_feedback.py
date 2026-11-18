"""Relieve measured seat shear during load transfer with both foot poses fixed."""

import numpy as np

from seat_plane_approach import stance_axes


class ContactContinuity:
    """Detect an initial seat contact that moved and then unloaded."""

    def __init__(self, count):
        self.origin = np.full((count, 3), np.nan)
        self.travel = np.zeros(count)
        self.light_ticks = np.zeros(count, dtype=int)
        self.support_ticks = np.zeros(count, dtype=int)
        self.settled = np.zeros(count, dtype=bool)
        self.release_tick = np.full(count, -1)

    def observe(self, tick, load_bw, point_w, armed):
        watching = armed & ~self.settled & (self.release_tick < 0)
        measured = watching & (load_bw > .01) & np.isfinite(point_w).all(axis=1)
        first = measured & ~np.isfinite(self.origin).all(axis=1)
        self.origin[first] = point_w[first]
        movement = np.linalg.norm(point_w[measured] - self.origin[measured], axis=1)
        self.travel[measured] = np.maximum(self.travel[measured], movement)
        self.light_ticks = np.where(watching & (load_bw <= .01), self.light_ticks + 1, 0)
        self.support_ticks = np.where(watching & (load_bw >= .05), self.support_ticks + 1, 0)
        # At 50 Hz, 100 ms of sustained support completes initial load transfer.
        # A patch shift of one seat-edge margin followed by 60 ms unloaded
        # invalidates the original grip decision. The braked reference resumes.
        self.settled |= self.support_ticks >= 5
        self.release_tick[watching & (self.travel > .010) & (self.light_ticks >= 3)] = tick
        return self.release_tick >= 0


class SeatShearFeedback:
    def __init__(self, approach, normal_transfer=False, follow_descent=False, verify_contact=False):
        self.approach = approach
        count = len(approach.roll)
        self.first = np.full(count, -1)
        self.armed = np.zeros(count, dtype=bool)
        self.offset = np.zeros((count, 2))
        self.delta = np.zeros_like(approach.delta)
        self.trace = []
        self.normal_transfer = normal_transfer
        self.follow_descent = follow_descent
        self.anchor = np.zeros_like(approach.delta)
        self.latched = np.zeros(count, dtype=bool)
        self.depth = np.zeros(count)
        self.filtered_load = np.zeros(count)
        self.unloaded_speed = np.full(count, np.nan)
        self.demonstrated = np.zeros(count,dtype=bool)
        self.continuity = ContactContinuity(count) if verify_contact else None

    def remaining_descent(self, row, frame, reference):
        # Measure the original reference's normal travel relative to the
        # captured command, using the same stationary feet for both poses.
        feet = self.approach.feet(frame[:23])
        anchored = self.approach.feet(self.anchor[row])
        nominal = self.approach.feet(reference)
        axes = anchored[0,:3,:3] @ feet[0,:3,:3].T @ stance_axes(feet,frame[69:72])
        normal = axes @ self.approach.normal[row]
        cap = self.approach.vertices[np.argmin(self.approach.vertices @ normal)]
        rotation = anchored[0,:3,:3] @ nominal[0,:3,:3].T
        anchor_patch = cap - anchored[:,:3,3].mean(0)
        nominal_patch = rotation @ (cap - nominal[:,:3,3].mean(0))
        return float((anchor_patch-nominal_patch) @ normal)

    def update(self, tick, normal, shear, weight, braked, hold, frames, desired, previous, brake, point_w=None):
        # Forces arrive in the pelvis frame. Both sensor readings are from the
        # last completed physics step. The brake and clearance hold own approach.
        load = np.linalg.norm(normal, axis=1)
        new = (self.first < 0) & (load > .01 * weight)
        if self.normal_transfer and tick % 5 == 0 and len(brake.trace) >= len(brake.active) and len(brake.active):
            samples = np.asarray(brake.trace[-len(brake.active):])
            rows = samples[:,1].astype(int)
            unloaded = (self.first[rows]<0) & ~new[rows]
            self.unloaded_speed[rows[unloaded]] = samples[unloaded,3]
        self.first[new] = tick
        self.armed[new] = braked[new] & (np.linalg.norm(shear[new], axis=1) >= .48 * load[new])
        if self.normal_transfer:
            self.armed[new] |= braked[new] & (self.unloaded_speed[new] <= .002)
        if self.continuity is not None:
            released = self.continuity.observe(tick, load / weight, point_w, self.armed)
            self.armed[released] = False
            self.latched[released] = False
        eligible = self.armed & (hold.first_load >= 0) & ~hold.enabled
        if self.normal_transfer:
            newly_latched = eligible & ~self.latched
            self.anchor[newly_latched] = previous[newly_latched]
            self.latched |= newly_latched
            self.filtered_load = .8*self.filtered_load + .2*load/weight
        if tick % 2:
            return
        for i in np.flatnonzero(eligible):
            frame = frames[i]
            if self.normal_transfer:
                rate = np.clip(.08*(.12-self.filtered_load[i]),0.,.009)
                limit = .012
                if self.follow_descent and self.depth[i] >= limit and rate > 0:
                    limit = max(self.depth[i], self.remaining_descent(i,frame,desired[i]))
                self.depth[i] = max(self.depth[i],min(limit,self.depth[i]+.04*rate))
                # With an aligned demonstration and sustained load, tangential
                # grip is already measurable before the full 12 mm descent.
                supported = self.demonstrated[i] and self.continuity.settled[i] and load[i] >= .05*weight[i]
                if (self.depth[i] > .012 or supported) and load[i] > .01*weight[i]:
                    axes = stance_axes(self.approach.feet(frame[:23]),frame[69:72])
                    n = axes @ self.approach.normal[i]
                    forward = axes[:,0] - n*(axes[:,0]@n)
                    forward /= np.linalg.norm(forward)
                    ratio = np.stack((forward,np.cross(n,forward))) @ shear[i] / load[i]
                    magnitude = np.linalg.norm(ratio)
                    if magnitude > .36:
                        # Unload tangential spring force as pressure is restored.
                        # Stay below the scored sliding speed during this adjustment.
                        self.offset[i] += .04*min(.008,.05*(magnitude-.36))*ratio/magnitude
                        self.offset[i] /= max(1.,np.linalg.norm(self.offset[i])/.012)
                x, y = self.offset[i]
                solution = self.approach.correct(self.anchor[i],frame,self.approach.normal[i],
                                                 0.,forward_m=x,lateral_m=y,normal_m=-self.depth[i])
                if solution is not None:
                    self.delta[i] = solution[0]
                self.trace.append((tick,i,load[i]/weight[i],self.filtered_load[i],self.depth[i],x,y))
                continue
            axes = stance_axes(self.approach.feet(frame[:23]), frame[69:72])
            n = axes @ self.approach.normal[i]
            forward = axes[:, 0] - n * (axes[:, 0] @ n)
            forward /= np.linalg.norm(forward)
            tangent = np.stack((forward, np.cross(n, forward)))
            ratio = tangent @ shear[i] / max(load[i], .01 * weight[i])
            magnitude = np.linalg.norm(ratio)
            # A damped admittance shifts the leg targets in the direction of
            # seat friction, so the feet can supply more of that force.
            if load[i] > .01 * weight[i] and magnitude > .36:
                rate = min(.045, .23 * (magnitude - .36))
                self.offset[i] += .04 * rate * ratio / magnitude
                self.offset[i] /= max(1., np.linalg.norm(self.offset[i]) / .012)
            if np.any(self.offset[i]):
                x, y = self.offset[i]
                solution = self.approach.correct(desired[i], frame, self.approach.normal[i],
                                                 0., forward_m=x, lateral_m=y)
                if solution is not None:
                    self.delta[i] = solution[0]
            self.trace.append((tick, i, load[i]/weight[i], *ratio, *self.offset[i]))

    def save(self, path):
        columns = (['tick','row','load_bw','filtered_load_bw','normal_descent_m','forward_m','lateral_m'] if self.normal_transfer
                   else ['tick','row','load_bw','shear_forward','shear_lateral','forward_m','lateral_m'])
        evidence = {} if self.continuity is None else dict(
            release_tick=self.continuity.release_tick, patch_travel_m=self.continuity.travel)
        np.savez_compressed(path,first_load_tick=self.first,armed=self.armed,latched=self.latched,
                            trace=np.asarray(self.trace),columns=columns,**evidence)

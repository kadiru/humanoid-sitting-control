"""Current-scan support measurement for RL, using depth + encoder FK + IMU.

No simulator imports or global poses. No odometry or map registration is needed.
The observed patch centroid/spans describe visible samples, NOT furniture bounds.
Invalid/ambiguous planes are withheld. The original shared-frame tracker remains
unchanged and is still the authority for complete alignment geometry.
"""
import numpy as np
from .shared_frame import (BodyModel, apply, transform, level_rotation,
    horizontal_planes, associate_seat_plane, plane_clearance_allowance)


class CurrentSupportDepth:
    def __init__(self, model_path, K, T_P_C, seat_height):
        self.model = BodyModel(model_path)
        self.K = K
        self.T_P_C = T_P_C
        self.seat_height = seat_height

    def read(self, depth, q, gravity):
        depth = np.asarray(depth).squeeze()
        h, w = depth.shape
        v, u = np.mgrid[:h, :w]
        valid = np.isfinite(depth) & (depth > .08) & (depth < 4.)
        z = np.where(valid, depth, 0.)
        C = np.stack(((u-self.K[0, 2])*z/self.K[0, 0],
                      (v-self.K[1, 2])*z/self.K[1, 1], z), -1)
        P = apply(self.T_P_C, C.reshape(-1, 3)).reshape(h, w, 3)
        poses = self.model.fk(q)
        valid &= self.model.self_mask(P.reshape(-1, 3), poses).reshape(h, w)
        R = level_rotation(gravity)
        L = P @ R.T
        dx = L[1:-1, 2:]-L[1:-1, :-2]
        dy = L[2:, 1:-1]-L[:-2, 1:-1]
        normals = np.cross(dx, dy)
        length = np.linalg.norm(normals, axis=-1)
        ok = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] &
              valid[2:, 1:-1] & valid[:-2, 1:-1] & (length > 1.e-8) &
              (np.linalg.norm(dx, axis=-1) < .12) & (np.linalg.norm(dy, axis=-1) < .12))
        normals = normals[ok] / length[ok, None]
        planes = horizontal_planes(L[1:-1, 1:-1][ok], normals)
        plane, diagnostic = associate_seat_plane(planes, self.seat_height, 0., False)
        values = np.zeros(10, dtype=np.float32)
        values[-1] = 1.
        if plane is None:
            return values, diagnostic
        vertices = self.model.vertices['pelvis'] @ R.T
        allowance = plane_clearance_allowance(plane, vertices)
        if not allowance['conditioned']:
            return values, diagnostic | dict(usable=False, allowance=allowance)
        n_P = R.T @ plane['normal']
        clearance = float((self.model.vertices['pelvis'] @ n_P + plane['offset']).min())
        points_P = plane['points'] @ R
        center = points_P[:, :2].mean(0)
        span = np.ptp(points_P[:, :2], axis=0)
        values[:] = [clearance, *n_P, *center, *span, 1., 0.]
        return values, diagnostic | dict(usable=True, allowance=allowance,
            points=len(points_P), clearance_m=clearance)

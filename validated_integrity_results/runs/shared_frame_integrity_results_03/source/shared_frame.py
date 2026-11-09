"""Sensor-only common-frame geometry. No Isaac, controller, or ground-truth imports.

T_A_B maps column coordinates from B to A; metres, seconds, right handed,
quaternions XYZW. O is gravity aligned at startup, with observed floor z=0.
P is the pelvis root; C is ROS optical (+x right, +y down, +z forward).
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import label, binary_erosion
from scipy.optimize import least_squares


def transform(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def apply(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def level_rotation(gravity):
    up = -np.asarray(gravity, dtype=float)
    up /= np.linalg.norm(up)
    return Rotation.align_vectors([[0., 0., 1.]], up[None])[0].as_matrix()


def voxel(points, size=.018, normals=None):
    if not len(points):
        return (points, normals) if normals is not None else points
    _, ids = np.unique(np.floor(points / size).astype(int), axis=0, return_index=True)
    return (points[ids], normals[ids]) if normals is not None else points[ids]


def bounded_map(points, normals, limit=40000):
    """Preserve seat-height geometry, distribute the remaining budget spatially.

    Voxel output is coordinate-sorted. Taking its tail crops the negative-X
    side of a scene and can turn a complete seat into an artificial half-seat.
    Retain every support-height voxel up to 75% of the budget, and sample both
    sets across their entire sorted extent. No crop boundary becomes an edge.
    """
    if len(points) <= limit:
        return points, normals
    support = (points[:, 2] > .12) & (points[:, 2] < .75)
    keep = np.flatnonzero(support); rest = np.flatnonzero(~support)
    if len(keep) > 3*limit//4:
        keep = keep[np.linspace(0, len(keep)-1, 3*limit//4).astype(int)]
    space = limit-len(keep)
    if len(rest) > space:
        rest = rest[np.linspace(0, len(rest)-1, space).astype(int)]
    ids = np.sort(np.r_[keep, rest])
    return points[ids], normals[ids]


class BodyModel:
    """Static calibrated link geometry and encoder FK; ignores exported poses/masses."""
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        self.names = data['joint_names']
        self.joints = data['joints']
        self.frames = {}
        for j in self.joints:
            A = transform(Rotation.from_quat(j['q0']).as_matrix(), j['p0'])
            B = transform(Rotation.from_quat(j['q1']).as_matrix(), j['p1'])
            self.frames[j['name']] = (A, np.linalg.inv(B))
        self.vertices = {}
        for c in data['collisions']:
            pts = np.asarray(c['points'])
            pts = pts[ConvexHull(pts).vertices]
            self.vertices.setdefault(c['body'], []).append(pts)
        self.vertices = {k: np.concatenate(v) for k, v in self.vertices.items()}
        self.bounds = {k: (v.min(0), v.max(0)) for k, v in self.vertices.items()}

    def fk(self, q):
        if len(q) != len(self.names):
            raise ValueError('Encoder order/length does not match calibrated model')
        values = dict(zip(self.names, q))
        poses = {'pelvis': np.eye(4)}
        pending = self.joints.copy()
        while pending:
            remaining = []
            for j in pending:
                if not j['body0'] or j['body0'][0] not in poses:
                    remaining.append(j)
                    continue
                A, B = self.frames[j['name']]
                axis = np.zeros(3)
                if j['name'] in values:
                    axis['XYZ'.index(j['axis'])] = values[j['name']]
                poses[j['body1'][0]] = poses[j['body0'][0]] @ A @ transform(Rotation.from_rotvec(axis).as_matrix()) @ B
            if len(remaining) == len(pending):
                raise ValueError('Disconnected kinematic tree')
            pending = remaining
        return poses

    def self_mask(self, points, poses, margin=.014):
        # Conservative FK-oriented collision bounds. This deliberately discards
        # ambiguous near-body depth instead of admitting moving robot geometry
        # into the static map. It is not a precise visual silhouette mask.
        keep = np.ones(len(points), dtype=bool)
        for body, (lo, hi) in self.bounds.items():
            T = poses[body]
            local = (points - T[:3, 3]) @ T[:3, :3]
            keep &= ~((local >= lo-margin).all(1) & (local <= hi+margin).all(1))
        return keep

    def proxy(self, poses):
        return np.concatenate([apply(poses[n], self.vertices[n]) for n in ('pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link')])

    def skeleton(self, poses):
        return np.asarray([[poses[j['body0'][0]][:3, 3], poses[j['body1'][0]][:3, 3]]
                           for j in self.joints if j['body0']])


@dataclass
class SensorFrame:
    stamp: float
    received: float
    depth: np.ndarray
    K: np.ndarray
    q: np.ndarray
    gravity: np.ndarray
    delta_rotation: np.ndarray  # integrated body gyro since previous sensor stamp
    encoder_stamp: float
    imu_stamp: float
    epoch: int = 0


def unproject(depth, K, T_P_C):
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError('Depth must be HxW or HxWx1')
    v, u = np.mgrid[:d.shape[0], :d.shape[1]]
    valid = np.isfinite(d) & (d > .08) & (d < 4.)
    z = np.where(valid, d, 0.)
    xyz = np.stack(((u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z), axis=-1)
    return apply(T_P_C, xyz[valid])


def cloud_normals(points):
    """Neighbourhood normals at 6 cm scale; adequate for 5 mm depth noise."""
    p = voxel(points)
    if len(p) < 40:
        return p, np.zeros_like(p)
    _, ids = cKDTree(p).query(p, k=min(20, len(p)))
    d = p[ids] - p[ids].mean(1)[:, None, :]
    _, vec = np.linalg.eigh(np.einsum('nki,nkj->nij', d, d))
    return p, vec[:, :, 0]


def depth_edges(depth, K, T_P_C):
    """Near side of physical depth jumps; image borders are not scene edges."""
    d = np.asarray(depth)
    if d.ndim == 3:
        d = d[..., 0]
    valid = np.isfinite(d) & (d > .08) & (d < 4.)
    safe = np.where(valid, d, 0.)
    edge = np.zeros_like(valid)
    for axis in (0, 1):
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        jump = valid[a] & valid[b] & (np.abs(safe[a]-safe[b]) > .06)
        edge[a] |= jump & (safe[a] < safe[b])
        edge[b] |= jump & (safe[b] < safe[a])
    v, u = np.nonzero(edge); z = d[edge]
    return voxel(apply(T_P_C, np.c_[(u-K[0, 2])*z/K[0, 0], (v-K[1, 2])*z/K[1, 1], z]), .012)


def edge_normals(points):
    """Horizontal point-to-line normals avoid inventing along-edge information."""
    if len(points) < 8:
        return np.zeros_like(points)
    _, ids = cKDTree(points[:, :2]).query(points[:, :2], k=min(9, len(points)))
    d = points[ids, :2]-points[ids, :2].mean(1)[:, None, :]
    _, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', d, d))
    return np.c_[vectors[:, :, 0], np.zeros(len(points))]


def horizontal_planes(points, normals):
    """Connected support extraction happens later; no labels or height metadata."""
    p = points[np.abs(normals[:, 2]) > .96]
    planes = []
    while len(p) >= 60 and len(planes) < 10:
        bins = np.floor(p[:, 2] / .012).astype(int)
        ids, counts = np.unique(bins, return_counts=True)
        h = (ids[counts.argmax()] + .5) * .012
        selected = np.abs(p[:, 2]-h) < .022
        candidate = p[selected]
        if len(candidate) < 60:
            break
        for _ in range(3):
            A = np.c_[candidate[:, :2], np.ones(len(candidate))]
            coeff = np.linalg.lstsq(A, candidate[:, 2], rcond=None)[0]
            residual = candidate[:, 2] - A @ coeff
            candidate = candidate[np.abs(residual) < max(.008, 2.5*np.median(np.abs(residual)))]
            if len(candidate) < 30:
                break
        if len(candidate) >= 30:
            n = np.array([-coeff[0], -coeff[1], 1.])
            n /= np.linalg.norm(n)
            if n[2] > .97:
                planes.append(dict(normal=n, offset=-coeff[2]*n[2], points=candidate,
                                   rms=float(np.std(residual)), height=float(np.median(candidate[:, 2]))))
        p = p[~selected]
    return sorted(planes, key=lambda x: x['height'])


def regularize_plane_normals(points, normals):
    """Use fitted plane normals so depth noise cannot create fake XY information."""
    normals = normals.copy()
    planes = horizontal_planes(points, normals) if len(points) > 100 else []
    for plane in planes:
        selected = (np.abs(points @ plane['normal']+plane['offset']) < .020) & (np.abs(normals[:, 2]) > .96)
        normals[selected] = plane['normal']
    return normals, planes


def fit_curved_contour(points, center, dimensions, yaw):
    """Accept an ellipse only if actual depth edges support its shape and pose."""
    if points is None or len(points) < 40:
        return None
    xy = np.asarray(points)[:, :2]
    if len(xy) > 500:
        xy = xy[np.linspace(0, len(xy)-1, 500).astype(int)]
    def residual(v):
        c, s = np.cos(v[4]), np.sin(v[4])
        local = (xy-v[:2]) @ np.array([[c, -s], [s, c]])
        value = np.sum((local/v[2:4])**2, axis=1)-1.
        gradient = 2*np.linalg.norm(local/v[2:4]**2, axis=1)
        return value/np.maximum(gradient, .1)
    guess = np.r_[center, np.maximum(dimensions/2, .12), yaw]
    guess[2:4] = np.minimum(guess[2:4], [.64, .89])
    fit = least_squares(residual, guess,
        bounds=(np.r_[center-.35, .08, .12, yaw-np.pi/2], np.r_[center+.35, .65, .9, yaw+np.pi/2]),
        loss='soft_l1', f_scale=.004, max_nfev=60)
    v = fit.x; rms = float(np.sqrt(np.mean(residual(v)**2)))
    R = Rotation.from_euler('z', v[4]).as_matrix()[:2, :2]
    local = (xy-v[:2]) @ R/v[2:4]
    angles = np.sort(np.mod(np.arctan2(local[:, 1], local[:, 0]), 2*np.pi))
    coverage = 2*np.pi-np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))
    # Two parallel lines can have a large angular span while leaving almost all
    # of the inferred ellipse unseen. Reject bounded/poorly covered conic fits.
    angular_bins = len(np.unique(np.floor(angles/(2*np.pi/12)).astype(int)))
    if (not fit.success or np.any(fit.active_mask) or rms > .006 or
            coverage < np.deg2rad(200) or angular_bins < 7 or v[2]/v[3] > .85):
        return None
    return dict(center=v[:2], dimensions=2*v[2:4], yaw=(v[4]+np.pi/2)%np.pi-np.pi/2,
                rms_m=rms, angular_coverage_deg=float(np.rad2deg(coverage)), angular_bins=angular_bins)


def support_junctions(plane, support_xy, points, normals):
    """Bounded intersections of observed walls with observed support.

    Each segment needs planarity, near-base samples, vertical extent, and
    adjacent observed support. Never extend a floating object's upper surface
    to the seat or bridge an unseen span. These are not ICP landmarks.
    """
    empty = np.empty((0, 3))
    if points is None or normals is None or len(points) < 24:
        return empty, []
    up = plane['normal']; height = points @ up+plane['offset']
    selected = (height > -.015) & (height < .40) & (np.abs(normals @ up) < .30)
    p = points[selected]; n = normals[selected]
    tree = cKDTree(support_xy); segments = []; evidence = []
    for _ in range(6):
        if len(p) < 24:
            break
        ids = np.linspace(0, len(p)-1, min(96, len(p))).astype(int)
        proposals = n[ids]
        offsets = np.sum(proposals*p[ids], axis=1)
        masks = (np.abs(p @ proposals.T-offsets) < .012) & (np.abs(n @ proposals.T) > .85)
        inliers = masks[:, np.argmax(masks.sum(0))]
        if inliers.sum() < 24:
            break
        for _ in range(3):
            center = p[inliers].mean(0)
            delta = p[inliers]-center
            _, vectors = np.linalg.eigh(delta.T @ delta)
            normal = vectors[:, 0]; offset = -normal @ center
            inliers = (np.abs(p @ normal+offset) < .012) & (np.abs(n @ normal) > .85)
            if inliers.sum() < 24:
                break
        wall = p[inliers]; p = p[~inliers]; n = n[~inliers]
        if len(wall) < 24 or abs(normal @ up) >= .30:
            continue
        rms = float(np.sqrt(np.mean((wall @ normal+offset)**2)))
        h = wall @ up+plane['offset']
        tangent = np.cross(up, normal); tangent /= np.linalg.norm(tangent)
        s = wall @ tangent
        if rms > .008 or np.ptp(s) < .12 or np.quantile(h, .9) < .08:
            continue
        base = (h >= -.015) & (h < .05)
        high = h > .08
        if base.sum() < 8 or high.sum() < 8:
            continue
        origin = np.linalg.solve(np.stack((up, normal, tangent)), [-plane['offset'], -offset, 0.])
        samples = np.arange(s[base].min(), s[base].max()+.006, .012)
        # Require local observations at both heights; never bridge hidden spans.
        base_distance = cKDTree(s[base, None]).query(samples[:, None])[0]
        high_distance = cKDTree(s[high, None]).query(samples[:, None])[0]
        junction = origin+samples[:, None]*tangent
        near_support = tree.query(junction[:, :2])[0] < .03
        junction = junction[(base_distance < .025) & (high_distance < .04) & near_support]
        if len(junction) < 8 or np.ptp(junction @ tangent) < .10:
            continue
        segments.append(junction)
        evidence.append(dict(normal=normal, offset=float(offset), rms_m=rms,
            wall_points=len(wall), base_points=int(base.sum()), junction_points=len(junction),
            observed_base_min_height_m=float(h[base].min()),
            span_m=float(np.ptp(junction @ tangent)),
            endpoints=np.stack((junction[0], junction[-1]))))
    return (np.concatenate(segments) if segments else empty), evidence


def support_patch(plane, cell=.025, contour=None, context_points=None, context_normals=None):
    """Largest observed connected patch, with occupancy holes retained.

    Hull is for visualization only. Usability uses eroded occupied cells, never
    the hull/bounding rectangle (which would bridge unobserved gaps).
    """
    xy = plane['points'][:, :2]
    base = np.floor(xy.min(0)/cell).astype(int)-1
    ij = np.floor(xy/cell).astype(int)-base
    grid = np.zeros(tuple(ij.max(0)+2), bool)
    grid[ij[:, 0], ij[:, 1]] = True
    labels, num = label(grid)
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    if not num:
        return None
    component = labels == sizes.argmax()
    ids = component[ij[:, 0], ij[:, 1]]
    xy = xy[ids]
    if len(xy) < 30 or component.sum()*cell**2 < .05:
        return None
    if contour is not None and len(contour):
        # Height agreement alone also admits edges of unrelated objects. Match
        # contours to this connected support before fitting its shape/axis.
        # Otherwise a few distant edges reject a good ellipse and silently
        # substitute a biased polygon-facet rectangle orientation.
        contour = np.asarray(contour)
        contour = contour[cKDTree(xy).query(contour[:, :2])[0] < .03]
    boundary = xy[ConvexHull(xy).vertices]
    diff = np.roll(boundary, -1, axis=0)-boundary
    best = None
    for a in np.unique(np.mod(np.arctan2(diff[:, 1], diff[:, 0]), np.pi/2)):
        R = Rotation.from_euler('z', a).as_matrix()[:2, :2]
        local = xy @ R; lo = local.min(0); hi = local.max(0)
        if best is None or np.prod(hi-lo) < best[0]:
            best = (np.prod(hi-lo), a, R, lo, hi)
    _, yaw, R, lo, hi = best
    dims = hi-lo
    if dims[0] > dims[1]:
        yaw += np.pi/2
        dims = dims[::-1]
    yaw = (yaw+np.pi/2) % np.pi - np.pi/2
    center = ((lo+hi)/2) @ R.T
    hull_fill = float(ConvexHull(xy).volume/best[0])
    axis_method = 'minimum_area_rectangle'
    curved = fit_curved_contour(contour, center, dims, yaw)
    if curved is not None:
        center, dims, yaw = curved['center'], curved['dimensions'], curved['yaw']
        axis_method = 'fitted_observed_contour_ellipse'
    normal = plane['normal']
    z = -(plane['offset']+normal[:2] @ center)/normal[2]
    xaxis = np.array([np.cos(yaw), np.sin(yaw), 0.])
    xaxis -= normal * (xaxis @ normal); xaxis /= np.linalg.norm(xaxis)
    T = transform(np.column_stack((xaxis, np.cross(normal, xaxis), normal)), [*center, z])
    junctions, junction_evidence = support_junctions(plane, xy, context_points, context_normals)
    boundary_contour = np.r_[contour if contour is not None else np.empty((0, 3)), junctions]
    evidence = boundary_evidence(xy, boundary_contour, T, curved)
    evidence['depth_contour_sides'] = boundary_evidence(xy, contour, T, curved)['sides']
    evidence['junction_count'] = len(junction_evidence)
    if len(junctions) and curved is None:
        evidence['basis'] = 'opposing_contours_and_observed_plane_junctions'
    # 20 x 24 cm contact-proxy footprint; rotation conservatively enclosed here.
    eroded = binary_erosion(component, structure=np.ones((9, 11), bool))
    safe = (np.argwhere(eroded)+base+.5)*cell
    return dict(T=T, boundary=boundary, cells=(np.argwhere(component)+base+.5)*cell,
                safe_cells=safe, dimensions=dims, axis_ambiguous=bool(max(dims)/min(dims)<1.15),
                heading_modulo_pi=True, observed_area=float(component.sum()*cell**2), cell=cell,
                axis_method=axis_method, hull_fill=hull_fill, curved_fit=curved,
                boundary_evidence=evidence, junctions=junctions, junction_evidence=junction_evidence)


def boundary_evidence(xy, contour, T, curved=None):
    """Require physical depth contours at opposing extents, not an image crop.

    This is evidence for an observed patch center, not proof of the full shape.
    A successful, sufficiently covered ellipse is an explicit shape hypothesis.
    """
    if curved is not None:
        return dict(forward=True, lateral=True, yaw=True, basis='covered_ellipse_model',
                    sides=[[True, True], [True, True]])
    sides = [[False, False], [False, False]]
    if contour is not None and len(contour) >= 12:
        contour = np.asarray(contour)
        near = cKDTree(xy).query(contour[:, :2])[0] < .03
        contour = contour[near]
        if len(contour) >= 12:
            local = apply(np.linalg.inv(T), contour)[:, :2]
            local_xy = (xy-T[:2, 3]) @ T[:2, :2]
            normals = edge_normals(contour) @ T[:3, :3]
            for axis in (0, 1):
                for side, edge in enumerate((local_xy[:, axis].min(), local_xy[:, axis].max())):
                    selected = (abs(local[:, axis]-edge) < .025) & (abs(normals[:, axis]) > .85)
                    sides[axis][side] = bool(selected.sum() >= 5 and
                        np.ptp(local[selected, 1-axis]) >= .08)
    return dict(forward=all(sides[0]), lateral=all(sides[1]),
                yaw=any(all(s) for s in sides), basis='opposing_physical_contours', sides=sides)


class DirectionalBudget:
    """Directional engineering drift allowance; not a statistical covariance.

    Coordinates are [0.5*yaw, O.x, O.y, O.z] in metres. Observed eigenspaces
    reset to an 8 mm model allowance; only the nullspace retains/grows drift.
    Stationary-foot agreement reduces growth, but never measures common slip.
    This budget excludes unknown seat extents and extrinsic/IMU bias.
    """
    def __init__(self):
        self.matrix = np.eye(4)*.008

    def update(self, dt, diagnostic, kinematic_prior):
        observed = np.asarray(diagnostic.get('observed_projector', np.zeros((4, 4))))
        if not diagnostic['accepted']:
            observed = np.zeros((4, 4))
        null = np.eye(4)-observed
        predicted = self.matrix + dt*(.015 if kinematic_prior else .08)*np.eye(4)
        self.matrix = null @ predicted @ null + .008*observed

    def allowance(self, jacobian):
        return float(np.linalg.norm(np.asarray(jacobian) @ self.matrix))


def associate_seat_plane(planes, seat_height, expected_height, registered):
    """Do not assign seat identity from an untrusted predicted world height.

    With tracking, match the reference height. Without it, require an observed
    lower floor and the learned floor-to-seat separation in the SAME scan.
    A lone plane cannot identify itself as a seat after translation is lost.
    """
    reliable = [p for p in planes if len(p['points']) >= 60 and p['rms'] < .008]
    if registered:
        matches = [p for p in reliable if abs(p['height']-expected_height) < .015]
        basis = 'registered_reference_height'
    else:
        floors = []
        for p in reliable:
            xy = p['points'][:, :2]
            if (len(xy) >= 150 and np.linalg.matrix_rank(xy-xy.mean(0), tol=1.e-6) == 2 and
                    ConvexHull(xy).volume > .15):
                floors.append(p)
        if not floors:
            return None, dict(confirmed=False, basis='no_current_floor_reference')
        floor = min(floors, key=lambda p: p['height'])
        matches = [p for p in reliable if
                   abs(p['height']-floor['height']-seat_height) < .015 and
                   p['normal'] @ floor['normal'] > np.cos(np.deg2rad(2.))]
        basis = 'current_floor_seat_separation'
    if len(matches) != 1:
        return None, dict(confirmed=False, basis=basis, candidates=len(matches))
    return matches[0], dict(confirmed=True, basis=basis, candidates=1)


def plane_clearance_allowance(plane, vertices):
    """Conditioning of plane extrapolation to body vertices, in one frame.

    A small residual on a thin visible strip does not constrain its slope.
    Count 6 cm spatial cells as effective samples, with a 5 mm noise floor,
    and reserve 8 mm for unmodelled error. This is an engineering allowance,
    not a calibrated statistical confidence interval.
    """
    points = voxel(plane['points'], .06)
    if len(points) < 6:
        return dict(allowance_m=None, effective_samples=len(points), conditioned=False)
    center = points[:, :2].mean(0)
    X = np.c_[points[:, :2]-center, np.ones(len(points))]
    Q = np.c_[vertices[:, :2]-center, np.ones(len(vertices))]
    if np.linalg.matrix_rank(X, tol=1.e-5) < 3:
        return dict(allowance_m=None, effective_samples=len(points), conditioned=False)
    leverage = np.einsum('ij,jk,ik->i', Q, np.linalg.inv(X.T@X), Q)
    allowance = .008 + 3*max(.005, plane['rms'])*np.sqrt(max(0., leverage.max()))
    return dict(allowance_m=float(allowance), effective_samples=len(points),
                conditioned=bool(allowance < .02))


def task_quality(seat, T, diagnostic, budget, age, direct_plane, vertices, usable):
    """Separate plane distance from geometry and pose needed for alignment."""
    R = seat['T'][:3, :3]
    basis = dict(forward=np.r_[0., R[:, 0]], lateral=np.r_[0., R[:, 1]],
                 yaw=np.array([2., 0., 0., 0.]))
    null = np.eye(4)-np.asarray(diagnostic.get('observed_projector', np.zeros((4, 4))))
    if not diagnostic['accepted']:
        null = np.eye(4)
    evidence = seat['boundary_evidence']
    quality = dict(kind='observability and engineering allowances; not calibrated probabilities',
                   seat_boundary=evidence, quantities={})
    for name, J in basis.items():
        allowance = budget.allowance(J)
        null_fraction = float(np.sqrt(np.clip(J @ null @ J/(J @ J), 0., 1.)))
        reasons = []
        if null_fraction > .1:
            reasons.append('motion_direction_unobserved')
        if diagnostic.get('reacquisition_required', False):
            reasons.append('pose_reacquisition_required')
        if age >= 1.:
            reasons.append('seat_observation_expired')
        if allowance >= (np.deg2rad(5.) if name == 'yaw' else .035):
            reasons.append('unobserved_motion_budget')
        if not evidence[name]:
            reasons.append('physical_extent_unconfirmed' if name != 'yaw' else 'axis_unconfirmed')
        if seat['axis_ambiguous']:
            reasons.append('seat_axis_ambiguous')
        quality['quantities'][name] = dict(valid=not reasons, reasons=reasons,
            allowance=allowance, unit='rad' if name == 'yaw' else 'm',
            instantaneous_null_fraction=null_fraction)
    direct = None
    if direct_plane is not None:
        # Algebraically cancels odometry: the fitted current scan plane expressed
        # in P measures collider-to-plane distance even when XY is unobservable.
        n = T[:3, :3].T @ direct_plane['normal']
        d = direct_plane['normal'] @ T[:3, 3]+direct_plane['offset']
        direct = float(np.min(vertices @ n+d))
    conditioning = (plane_clearance_allowance(direct_plane, apply(T, vertices))
                    if direct_plane is not None else dict(allowance_m=None, effective_samples=0, conditioned=False))
    quality['quantities']['clearance'] = dict(valid=direct is not None and conditioning['conditioned'],
        reasons=([] if conditioning['conditioned'] else ['plane_extrapolation_unconstrained'])
                if direct is not None else ['no_current_seat_plane'],
        source='current_depth_plane' if direct is not None else 'map_prediction_only',
        **conditioning,
        plane_rms_m=float(direct_plane['rms']) if direct is not None else None)
    quality['alignment_valid'] = bool(usable and all(quality['quantities'][x]['valid'] for x in basis))
    quality['landing_patch_observed'] = bool(usable)
    return quality, direct


def register(points, target, normals, prior, iterations=12, edges=None, target_edges=None,
             source_normals=None, minimum_overlap=.35, maximum_rms=.015):
    """Gravity-constrained robust point-to-plane ICP; reports observable DOFs.

    Null-space increments are zero: gyro/kinematic prior is explicitly retained
    for those DOFs, rather than claiming a plane measures in-plane translation.
    """
    T = prior.copy()
    tree = cKDTree(target)
    edge_tree = cKDTree(target_edges) if target_edges is not None and len(target_edges) >= 12 else None
    edge_n = edge_normals(target_edges) if edge_tree is not None else None
    rank = 0; rms = np.inf; fraction = 0.; eigenvalues = np.zeros(4)
    observed_projector = np.zeros((4, 4)); null_vectors = np.eye(4)
    aborted = False
    for _ in range(iterations):
        rotated = points @ T[:3, :3].T
        world = rotated+T[:3, 3]
        distance, ids = tree.query(world)
        ok = distance < .065
        if source_normals is not None:
            transformed_normals = source_normals @ T[:3, :3].T
            ok &= np.abs(np.sum(transformed_normals*normals[ids], axis=1)) > .9
        fraction = float(ok.mean())
        if ok.sum() < 60:
            break
        n = normals[ids[ok]]
        r = np.sum((world[ok]-target[ids[ok]])*n, axis=1)
        w = np.minimum(1., .008/np.maximum(np.abs(r), 1.e-9))
        J = np.c_[np.cross(rotated[ok], n)[:, 2], n]
        edge_count = 0
        if edge_tree is not None and edges is not None and len(edges) >= 12:
            er = edges @ T[:3, :3].T
            ep = er+T[:3, 3]
            ed, ei = edge_tree.query(ep)
            good = ed < .055
            edge_count = int(good.sum())
            if edge_count:
                en = edge_n[ei[good]]
                residual = np.sum((ep[good]-target_edges[ei[good]])*en, axis=1)
                # One contour sample gets the same weight as one plane sample.
                # A straight line still has an unobservable tangential direction.
                J = np.r_[J, np.c_[np.cross(er[good], en)[:, 2], en]]
                r = np.r_[r, residual]
                w = np.r_[w, np.minimum(1., .008/np.maximum(np.abs(residual), 1.e-9))]
        # Compare translation with angular displacement at a 0.5 m lever arm;
        # raw radians/metres Hessian eigenvalues are not dimensionally comparable.
        lever_arm = .5
        J[:, 0] /= lever_arm
        H = (J.T*w) @ J
        eigenvalues, V = np.linalg.eigh(H)
        observed = eigenvalues > max(2., eigenvalues.max()*.002)
        rank = int(observed.sum())
        observed_projector = V[:, observed] @ V[:, observed].T
        null_vectors = V[:, ~observed].T
        inv = np.where(observed, 1./np.maximum(eigenvalues, 1.e-10), 0.)
        delta = -(V*inv) @ V.T @ (J.T @ (w*r))
        delta[0] /= lever_arm
        if abs(delta[0]) > .12 or np.linalg.norm(delta[1:]) > .08:
            aborted = True
            break
        T[:3, :3] = Rotation.from_euler('z', delta[0]).as_matrix() @ T[:3, :3]
        T[:3, 3] += delta[1:]
        rms = float(np.sqrt(np.mean(r*r)))
        if np.linalg.norm(delta) < 1.e-5:
            break
    # Evaluate the returned pose, not the residual before the last increment.
    world = apply(T, points)
    distance, ids = tree.query(world)
    ok = distance < .065
    if source_normals is not None:
        ok &= np.abs(np.sum((source_normals @ T[:3, :3].T)*normals[ids], axis=1)) > .9
    fraction = float(ok.mean())
    final_residual = np.sum((world[ok]-target[ids[ok]])*normals[ids[ok]], axis=1)
    rms = float(np.sqrt(np.mean(final_residual**2))) if len(final_residual) else np.inf
    accepted = bool(not aborted and fraction > minimum_overlap and rms < maximum_rms and rank >= 2)
    return (T if accepted else prior), dict(accepted=accepted, rank=rank, rms_m=rms,
        aborted_step=aborted,
        overlap=fraction, eigenvalues=eigenvalues.tolist(), edge_matches=edge_count if 'edge_count' in locals() else 0,
        observed_projector=observed_projector.tolist(), null_vectors=null_vectors.tolist(),
        coordinate_order=['0.5*yaw', 'O.x', 'O.y', 'O.z'])


def reacquire(points, target, normals, prior, edges, target_edges, source_normals=None):
    """Bounded XY search after depth loss; gyro supplies orientation.

    Rejects an ambiguous straight edge via the same four-DOF observability test.
    The map is fixed during this search; no predicted scan is fused on failure.
    """
    if len(edges) < 20 or len(target_edges) < 20:
        return prior, dict(accepted=False, rank=0, rms_m=None, overlap=0.,
                           recovery='insufficient_physical_contours')
    tree = cKDTree(target_edges)
    transformed = apply(prior, edges)
    candidates = []
    for dx in np.arange(-.25, .251, .05):
        for dy in np.arange(-.25, .251, .05):
            distance = tree.query(transformed+[dx, dy, 0.])[0]
            candidates.append((float(np.mean(np.minimum(distance, .10)**2)), dx, dy))
    seeds = []
    for _, dx, dy in sorted(candidates):
        if all(np.linalg.norm(np.array([dx, dy])-s) > .06 for s in seeds):
            seeds.append(np.array([dx, dy]))
        if len(seeds) == 4:
            break
    solutions = []
    for xy in seeds:
        seed = prior.copy(); seed[:2, 3] += xy
        T, diag = register(points, target, normals, seed, iterations=20, edges=edges,
                           target_edges=target_edges, source_normals=source_normals,
                           minimum_overlap=.6, maximum_rms=.008)
        distance = tree.query(apply(T, edges))[0]
        coverage = float(np.mean(distance < .025))
        score = float(np.sqrt(np.mean(np.minimum(distance, .10)**2)))
        if diag['accepted'] and diag['rank'] == 4 and coverage > .6 and score < .02:
            solutions.append((score, T, diag, coverage))
    if not solutions:
        return prior, dict(accepted=False, rank=0, rms_m=None, overlap=0., recovery='no_observable_match')
    solutions.sort(key=lambda x: x[0])
    score, T, diag, coverage = solutions[0]
    # Two well-fitting but separated solutions indicate repeated/ambiguous shape.
    if any(s <= score+.003 and np.linalg.norm(other[:3, 3]-T[:3, 3]) > .04
           for s, other, _, _ in solutions[1:]):
        return prior, dict(accepted=False, rank=0, rms_m=None, overlap=0., recovery='ambiguous_matches')
    diag.update(recovery='reacquired', recovery_edge_rms_m=score, recovery_edge_coverage=coverage,
                recovery_translation_m=(T[:3, 3]-prior[:3, 3]).tolist())
    return T, diag


class SharedFrameTracker:
    def __init__(self, model, T_P_C):
        self.model, self.T_P_C = model, np.asarray(T_P_C)
        self.reset()

    def reset(self):
        self.T = None; self.stamp = None; self.epoch = None
        self.map_points = np.empty((0, 3)); self.map_normals = np.empty((0, 3))
        self.map_edges = np.empty((0, 3))
        self.floor = None; self.seat = None; self.seat_seen = None
        self.foot_anchors = None; self.sigma = .008
        self.directional_budget = DirectionalBudget()
        self.last_depth_stamp = None; self.reacquisition_required = False
        # A scan may extend the display map, but cannot become its own pose
        # reference. Keep the first gravity-aligned observation immutable.
        self.reference_points = None; self.reference_normals = None; self.reference_edges = None
        self.last_registration_stamp = None; self.last_recovery_attempt = -np.inf
        self.reference_seat_height = None

    def update(self, f):
        if max(abs(f.stamp-f.encoder_stamp), abs(f.stamp-f.imu_stamp)) > .005:
            raise ValueError('Depth/encoder/IMU skew exceeds 5 ms; interpolate upstream')
        if f.received < f.stamp or f.received-f.stamp > .25:
            raise ValueError('Negative or excessive transport delay')
        if self.epoch is not None and f.epoch != self.epoch:
            self.reset()
        if self.stamp is not None and f.stamp <= self.stamp:
            raise ValueError('Out-of-order frame; explicit epoch reset required')
        dt = 0. if self.stamp is None else f.stamp-self.stamp
        if dt > .3:
            raise ValueError('Missing synchronized gyro interval / frame gap >300 ms')
        poses = self.model.fk(f.q)
        feet = np.array([poses[s+'_ankle_roll_link'][:3, 3] for s in ('left', 'right')])
        points = voxel(unproject(f.depth, f.K, self.T_P_C), .012)
        before_mask = len(points)
        points = points[self.model.self_mask(points, poses)]
        edges = depth_edges(f.depth, f.K, self.T_P_C)
        edges = edges[self.model.self_mask(edges, poses)]
        p, n = cloud_normals(points)
        if self.last_depth_stamp is not None and f.stamp-self.last_depth_stamp >= .5:
            self.reacquisition_required = True
        if len(p) > 100:
            self.last_depth_stamp = f.stamp
        diagnostic = dict(accepted=False, rank=0, rms_m=None, overlap=0.)
        kin_ok = False
        foot_disagreement = None; prior_T = None
        if self.T is None:
            T = transform(level_rotation(f.gravity))
            planes = horizontal_planes(apply(T, p), n @ T[:3, :3].T)
            floors = [x for x in planes if x['height'] < -.35 and len(x['points']) > 150]
            if not floors:
                raise ValueError('Initial floor unobservable; no pose fallback')
            T[2, 3] = -floors[0]['height']
        else:
            T = self.T.copy()
            T[:3, :3] = T[:3, :3] @ f.delta_rotation
            # Gravity measurement removes tilt drift, retaining the yaw gauge.
            up = T[:3, :3] @ (-f.gravity/np.linalg.norm(f.gravity))
            tilt = Rotation.align_vectors([[0., 0., 1.]], up[None])[0].as_matrix()
            T[:3, :3] = tilt @ T[:3, :3]
            if self.foot_anchors is not None:
                candidates = self.foot_anchors - feet @ T[:3, :3].T
                foot_disagreement = float(np.linalg.norm(candidates[0]-candidates[1]))
                kin_ok = bool(np.linalg.norm(candidates[0]-candidates[1]) < .025 and
                              np.linalg.norm(candidates.mean(0)-T[:3, 3]) < .08)
                if kin_ok:
                    T[:3, 3] = candidates.mean(0)
            prior_T = T.copy()
            if len(p) > 100 and len(self.map_points) > 100:
                edge_heights = apply(T, edges)[:, 2]
                edges = edges[(edge_heights > .12) & (edge_heights < .75)]
                if self.last_registration_stamp is not None and f.stamp-self.last_registration_stamp >= .5:
                    self.reacquisition_required = True
                if self.reacquisition_required and f.stamp-self.last_recovery_attempt >= .5-1.e-6:
                    self.last_recovery_attempt = f.stamp
                    T, diagnostic = reacquire(p, self.reference_points, self.reference_normals,
                                              T, edges, self.reference_edges, source_normals=n)
                    if diagnostic['accepted']:
                        self.reacquisition_required = False
                elif not self.reacquisition_required:
                    T, diagnostic = register(p, self.reference_points, self.reference_normals, T,
                                             edges=edges, target_edges=self.reference_edges, source_normals=n,
                                             minimum_overlap=.6, maximum_rms=.008)
                    yaw_correction = Rotation.from_matrix(T[:3, :3] @ prior_T[:3, :3].T).magnitude()
                    if diagnostic['accepted'] and (np.linalg.norm(T[:3, 3]-prior_T[:3, 3]) > .04 or yaw_correction > .06):
                        # A large local correction may be a different match
                        # basin. Require the same multi-hypothesis verification
                        # used after loss before it may affect map or quality.
                        self.last_recovery_attempt = f.stamp
                        T, diagnostic = reacquire(p, self.reference_points, self.reference_normals,
                            prior_T, edges, self.reference_edges, source_normals=n)
                        diagnostic['large_local_correction'] = True
                        self.reacquisition_required = not diagnostic['accepted']
                if diagnostic['accepted']:
                    self.last_registration_stamp = f.stamp
        diagnostic['reacquisition_required'] = self.reacquisition_required
        world = apply(T, p); normals = n @ T[:3, :3].T
        normals, planes = regularize_plane_normals(world, normals)
        if self.reference_points is None:
            self.last_registration_stamp = f.stamp
            self.reference_points = world.copy(); self.reference_normals = normals.copy()
            reference_edges = apply(T, edges)
            self.reference_edges = reference_edges[(reference_edges[:, 2] > .12) &
                                                     (reference_edges[:, 2] < .75)].copy()
            for reference in (self.reference_points, self.reference_normals, self.reference_edges):
                reference.flags.writeable = False
        diagnostic['reference'] = 'immutable_initial_depth_scan'
        fully_observed = diagnostic['accepted'] and diagnostic['rank'] == 4
        if fully_observed or self.T is None:
            self.sigma = max(.008, self.sigma*.8)
        else:
            self.sigma += dt*(.015 if kin_ok else .08)
        self.directional_budget.update(dt, diagnostic, kin_ok)
        # Map only accepted observations; fixed global voxel budget prevents growth.
        fusion_allowed = self.T is None or fully_observed
        diagnostic['fusion_allowed'] = fusion_allowed
        if fusion_allowed:
            self.map_points, self.map_normals = voxel(np.r_[self.map_points, world], .018,
                                                       np.r_[self.map_normals, normals])
            if len(self.map_points) > 40000:
                self.map_points, self.map_normals = bounded_map(self.map_points, self.map_normals)
            ew = apply(T, edges)
            ew = ew[(ew[:, 2] > .12) & (ew[:, 2] < .75)]
            self.map_edges = voxel(np.r_[self.map_edges, ew], .012)
            if len(self.map_edges) > 12000:
                self.map_edges = self.map_edges[np.linspace(0, len(self.map_edges)-1, 12000).astype(int)]
        floor_candidates = [x for x in planes if abs(x['height']) < .10]
        if floor_candidates:
            self.floor = max(floor_candidates, key=lambda x: len(x['points']))
        # Fit from accumulated observed geometry; current scan must confirm seat.
        seat_planes = [x for x in planes if .15 < x['height'] < .65]
        direct_plane = None
        if self.seat is not None:
            same_height = [x for x in seat_planes if abs(x['height']-self.seat['T'][2, 3]) < .035
                           and x['rms'] < .012 and len(x['points']) >= 60]
            if same_height:
                direct_plane = max(same_height, key=lambda x: len(x['points']))
        if seat_planes and fusion_allowed:
            candidate = max(seat_planes, key=lambda x: len(x['points']))
            if self.seat is None or abs(candidate['height']-self.seat['T'][2, 3]) < .035:
                accumulated = horizontal_planes(self.map_points, self.map_normals)
                matching = [x for x in accumulated if abs(x['height']-candidate['height']) < .035]
                if matching:
                    plane = max(matching, key=lambda x: len(x['points']))
                    edge_on_plane = np.abs(self.map_edges @ plane['normal']+plane['offset']) < .012
                    patch = support_patch(plane, contour=self.map_edges[edge_on_plane],
                                          context_points=self.map_points, context_normals=self.map_normals)
                    if patch is not None:
                        self.seat = patch; self.seat_seen = f.stamp
                        direct_plane = candidate
        floor_z = -self.floor['offset']/self.floor['normal'][2] if self.floor else 0.
        plane_association = dict(confirmed=False, basis='no_seat_reference')
        if self.seat is not None:
            if self.reference_seat_height is None:
                self.reference_seat_height = self.seat['T'][2, 3]-floor_z
            direct_plane, plane_association = associate_seat_plane(planes, self.reference_seat_height,
                self.seat['T'][2, 3], diagnostic['accepted'])
        else:
            direct_plane = None
        feet_world = apply(T, feet)
        # No force/contact label: two feet geometrically near the observed floor.
        soles = []
        for side in ('left', 'right'):
            body = side+'_ankle_roll_link'
            soles.append(apply(T @ poses[body], self.model.vertices[body])[:, 2].min())
        if max(abs(np.array(soles)-floor_z)) < .035 and (fully_observed or self.T is None):
            self.foot_anchors = feet_world
        elif max(abs(np.array(soles)-floor_z)) > .07:
            self.foot_anchors = None
        self.T = T; self.stamp = f.stamp; self.epoch = f.epoch
        result = dict(stamp=f.stamp, latency_s=f.received-f.stamp, T_O_P=T.copy(),
            cloud=world, skeleton=apply(T, self.model.skeleton(poses).reshape(-1, 3)).reshape(-1, 2, 3),
            floor=self.floor, seat=self.seat, registration=diagnostic, kinematic_prior=kin_ok,
            seat_plane_association=plane_association,
            position_sigma_m=self.sigma, uncertainty_kind='engineering drift budget, not calibrated covariance')
        result['sensor_diagnostics'] = dict(voxel_points_before_mask=before_mask,
            voxel_points_after_mask=len(points), registration_points=len(p), physical_edge_points=len(edges),
            current_seat_plane_points=0 if direct_plane is None else len(direct_plane['points']),
            map_points=len(self.map_points), map_edges=len(self.map_edges),
            foot_translation_disagreement_m=foot_disagreement,
            visual_translation_correction_m=None if prior_T is None else (T[:3, 3]-prior_T[:3, 3]).tolist())
        if self.seat is None:
            result.update(valid=False, status='no_seat')
            return result
        T_S_P = np.linalg.inv(self.seat['T']) @ T
        proxy = apply(T_S_P, self.model.proxy(poses))
        age = f.stamp-self.seat_seen
        fresh = age <= .35
        usable = len(self.seat['safe_cells']) > 0
        pelvis_proxy = apply(T_S_P, self.model.vertices['pelvis'])
        result.update(T_S_P=T_S_P, clearance_m=float(pelvis_proxy[:, 2].min()),
            support_proxy_plane_clearance_m=float(proxy[:, 2].min()),
            pelvis_offset_m=T_S_P[:3, 3].copy(), yaw_axis_error_rad=float(np.arctan2(T_S_P[1, 0], T_S_P[0, 0])),
            seat_age_s=age, valid=bool(self.sigma < .035 and age < 1. and usable),
            status='visible' if fresh else ('occluded_prediction' if age < 1. else 'stale'),
            landing_patch_observed=usable,
            support_distance_m=float(cKDTree(self.seat['cells']).query(T[:2, 3])[0]))
        result['legacy_validity_reasons'] = [name for name, fail in (
            ('scalar_drift_budget', self.sigma >= .035), ('seat_observation_expired', age >= 1.),
            ('landing_patch_unobserved', not usable)) if fail]
        result['quality'], result['direct_clearance_m'] = task_quality(
            self.seat, T, diagnostic, self.directional_budget, age, direct_plane,
            self.model.vertices['pelvis'], usable)
        # No deployed controller consumes this observer yet. Make the main flag
        # conservative and explicit; retain the former rule for audit only.
        result['legacy_valid'] = result['valid']
        result['valid'] = result['quality']['alignment_valid']
        result['valid_scope'] = 'observed-patch alignment; clearance has a separate flag'
        result['proxy_world'] = apply(T, self.model.proxy(poses))
        # Plane clearance is distinct from finite observed support clearance.
        proxy_world = result['proxy_world']
        distance, _ = cKDTree(self.seat['cells']).query(proxy_world[:, :2])
        supported = distance < self.seat['cell']*.72
        result['overlap_clearance_m'] = float(proxy[supported, 2].min()) if supported.any() else None
        result['proxy_over_observed_support'] = bool(supported.any())
        result['body_clearances_m'] = {body: float(apply(np.linalg.inv(self.seat['T']) @ T @ poses[body], self.model.vertices[body])[:, 2].min())
            for body in ('left_knee_link', 'right_knee_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'torso_link')}
        obstacles = self.map_points[self.map_points[:, 2] > self.seat['T'][2, 3]+.06]
        result['observed_obstacle_distance_m'] = {}
        if len(obstacles):
            tree = cKDTree(obstacles)
            result['observed_obstacle_distance_m'] = {body: float(tree.query(apply(T @ poses[body], self.model.vertices[body]))[0].min())
                for body in ('left_knee_link', 'right_knee_link', 'torso_link', 'left_elbow_roll_link', 'right_elbow_roll_link')}
        return result

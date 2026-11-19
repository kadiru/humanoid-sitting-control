"""Standing scan from the existing rear camera. All distances are in metres."""
from itertools import combinations
import numpy as np
from scipy.spatial.transform import Rotation


TILT_DEG = 25.
SCAN_TICKS = (0,10,20,30,40)


def configure_scan(support):
    """Render two indexed tilt stops of the same optical center and intrinsics."""
    from copy import deepcopy
    original = support.make_cfg
    mount = support.REAR_CAMERA.copy()
    nominal = Rotation.from_matrix(mount[:3,:3])
    rotated = nominal * Rotation.from_euler('x',-TILT_DEG,degrees=True)
    mount[:3,:3] = rotated.as_matrix()

    def make_cfg(args,geometry):
        cfg = original(args,geometry)
        scan = deepcopy(cfg.scene.rear_camera)
        scan.prim_path = cfg.scene.rear_camera.prim_path+'Down'
        scan.offset.rot = tuple(rotated.as_quat())
        scan.update_latest_camera_pose = True
        cfg.scene.rear_scan = scan
        return cfg

    support.make_cfg = make_cfg
    return mount


def exposure(env,support,mount,nominal_depth):
    camera = env.raw.scene['rear_scan']
    clock = env.raw.sim.get_physics_step_count()
    support.base.acquire_current_camera(camera)
    depth = camera.data.output['distance_to_image_plane'].torch.cpu().numpy()[...,0].copy()
    root_rotation = Rotation.from_quat(env.robot.data.root_quat_w.torch.cpu().numpy())
    expected = root_rotation * Rotation.from_matrix(mount[:3,:3])
    actual = Rotation.from_quat(camera.data.quat_w_ros.torch.cpu().numpy())
    err = np.max((expected.inv()*actual).magnitude())
    if err > 1e-4 or env.raw.sim.get_physics_step_count() != clock:
        raise RuntimeError(f'Invalid scan exposure or orientation ({err} rad)')
    if np.any(np.all(depth==nominal_depth,axis=(1,2))):
        raise RuntimeError('Tilted exposure duplicates the nominal view')
    np.testing.assert_allclose(camera.data.intrinsic_matrices.torch.cpu().numpy(),
                               env.raw.scene['rear_camera'].data.intrinsic_matrices.torch.cpu().numpy())
    return depth


def rim_points(depth,k,mount,normal,offset):
    v,u = np.indices(depth.shape)
    rays = np.stack(((u+.5-k[0,2])/k[0,0],(v+.5-k[1,2])/k[1,1],np.ones_like(u)),axis=-1) @ mount[:3,:3].T
    numerator = mount[:3,3] @ normal + offset
    slope = rays @ normal
    with np.errstate(invalid='ignore',divide='ignore'):
        height = numerator + slope*depth
    valid = np.isfinite(depth) & (depth>.08) & (depth<3.)
    top = valid & (np.abs(height)<.003)
    below = (valid & (height < -.003)) | np.isposinf(depth)
    # Two top pixels followed by two deeper returns distinguish the lip
    # from a foreground robot part and from the bottom of the image.
    boundary = top[1:-2] & top[:-3] & below[2:-1] & below[3:]
    v,u = np.nonzero(boundary)
    ray = (rays[v+1,u]+rays[v+2,u])*.5
    return mount[:3,3] - ray*(numerator/(ray@normal))[:,None]


def fit_rim(y,x):
    # Aggregate by lateral strip so a long visible side cannot dominate the fit.
    samples = []
    for left in np.arange(-.20,.20,.02):
        sel = (y>=left)&(y<left+.02)
        if sel.any():
            samples.append((np.median(y[sel]),np.median(x[sel])))
    if len(samples)<5:
        return None
    yy,xx = np.asarray(samples).T
    design = np.column_stack((np.ones_like(yy),yy,yy**2))
    local = np.abs(yy)<=.10
    if local.sum()>=4 and np.min(np.abs(y))<=.010:
        slope,intercept = np.polyfit(yy[local],xx[local],1)
        residual = xx[local]-(slope*yy[local]+intercept)
        if abs(slope)<=.85 and np.ptp(residual)<=.009:
            return np.array([intercept,slope,0.]),float(np.ptp(residual)),int(local.sum())
    # A front boundary may be straight, chamfered, or curved. Select a smooth
    # local fit supported on both sides of the robot's centerline.
    best = None
    for ids in combinations(range(len(yy)),3):
        i,j,k = ids
        if yy[j]-yy[i]<.035 or yy[k]-yy[j]<.035:
            continue
        coef = np.linalg.solve(design[list(ids)],xx[list(ids)])
        if abs(coef[1])>.85 or not -6. <= coef[2] <= 1.:
            continue
        error = abs(design@coef-xx)
        keep = error<.007
        if keep.sum()<5 or yy[keep].min()>-.024 or yy[keep].max()<.024:
            continue
        rank = (int(keep.sum()),-float(np.mean(error[keep])))
        if best is None or rank>best[0]:
            best = rank,keep
    if best is None:
        return None
    keep = best[1]
    coef = np.linalg.lstsq(design[keep],xx[keep],rcond=None)[0]
    return coef, float(np.ptp((design@coef-xx)[keep])), int(keep.sum())


def measure(model,depth,k,mount,joints,gravity,plane):
    if not plane[11] or plane[12]!=0:
        return None
    up = -gravity/np.linalg.norm(gravity)
    normal = np.asarray(plane[2:5],float)
    if normal@up < .8:
        return None
    fk = model.fk(joints)
    feet = np.stack([fk[s+'_ankle_roll_link'][:3,3] for s in ('left','right')])
    forward = np.cross(feet[0]-feet[1],up)
    forward /= np.linalg.norm(forward)
    tangent = forward-(forward@normal)*normal
    tangent /= np.linalg.norm(tangent)
    side = np.cross(normal,tangent)
    pts = rim_points(depth,k,mount,normal,plane[5])
    pts = pts[model.self_mask(pts,fk,margin=.005)] - feet.mean(0)
    fit = fit_rim(pts@side,pts@tangent)
    if fit is None:
        return None
    coef,spread,n = fit
    center = feet.mean(0)
    return dict(edge=float(coef[0]),span=spread,samples=n,
                height=float(-(center@normal+plane[5])/(normal@up)),
                pitch=float(np.degrees(np.arctan2(normal@forward,normal@up))),
                normal=normal,forward=tangent,feet_center=center,up=up,direct=True)


def true_intercept(spec,root,rotation,measured):
    # Audit only. Intersect the simulated top polygon with the sagittal plane
    # used by the estimator. Yawed and round seats need the same lateral reference.
    if spec.get('mesh_file'):
        return np.nan
    shape = spec.get('shape','rect')
    if shape=='ellipse':
        t = np.arange(16)*2*np.pi/16
        outline = np.column_stack((np.cos(t),np.sin(t)))/2
    elif shape=='chamfered':
        outline = np.array([[-.35,-.5],[.35,-.5],[.5,-.35],[.5,.35],[.35,.5],[-.35,.5],[-.5,.35],[-.5,-.35]])
    else:
        outline = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    local = np.column_stack((outline*[spec['depth'],spec['width']],np.zeros(len(outline))))
    surface = Rotation.from_quat(spec['rotation_xyzw']).apply(local)+np.asarray(spec['top_center'])
    rel = (surface-root)@rotation-measured['feet_center']
    x = rel@measured['forward']
    y = rel@np.cross(measured['normal'],measured['forward'])
    cross = np.flatnonzero((y>=0)!=(np.roll(y,-1)>=0))
    if not len(cross):
        return np.nan
    end = (cross+1)%len(y)
    return max(x[cross]-y[cross]*(x[end]-x[cross])/(y[end]-y[cross]))

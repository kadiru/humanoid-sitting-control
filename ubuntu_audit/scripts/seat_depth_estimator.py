"""Calibrated depth-only horizontal sitting-surface estimator. No GT/segmentation."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

def estimate(depth,K,position,quaternion):
    depth=np.asarray(depth).squeeze();h,w=depth.shape
    v,u=np.mgrid[:h,:w];safe=np.where(np.isfinite(depth),depth,0.);camera=np.stack(((u-K[0,2])*safe/K[0,0],(v-K[1,2])*safe/K[1,1],safe),axis=-1)
    valid=np.isfinite(depth)&(depth>.1)&(depth<5)
    camera=np.nan_to_num(camera,posinf=0,neginf=0)
    world=camera@Rotation.from_quat(quaternion).as_matrix().T+position
    dx=world[1:-1,2:]-world[1:-1,:-2];dy=world[2:,1:-1]-world[:-2,1:-1]
    normal=np.cross(dx,dy);norm=np.linalg.norm(normal,axis=-1);horizontal=np.abs(normal[...,2])> .97*np.maximum(norm,1e-10)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    p=world[1:-1,1:-1][horizontal&ok&(norm>1e-10)]
    p=p[(p[:,2]>.12)&(p[:,2]<.75)]
    if len(p)<100:raise ValueError('Insufficient visible horizontal support points')
    hist,edges=np.histogram(p[:,2],bins=np.arange(.12,.755,.005));index=hist.argmax();height=float(np.median(p[(p[:,2]>=edges[index])&(p[:,2]<edges[index+1]),2]))
    plane=p[np.abs(p[:,2]-height)<.003];xy=plane[:,:2]
    if len(xy)<100:raise ValueError('Support plane too small')
    hull=ConvexHull(xy);boundary=xy[hull.vertices];diff=np.roll(boundary,-1,axis=0)-boundary
    angles=np.unique(np.mod(np.arctan2(diff[:,1],diff[:,0]),np.pi/2));best=None
    for yaw in angles:
        c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s],[s,c]]);local=xy@R
        lo=local.min(0);hi=local.max(0);dims=hi-lo;area=float(dims.prod())
        if best is None or area<best[0]:best=(area,yaw,R,lo,hi)
    area,yaw,R,lo,hi=best;dimensions=hi-lo;center=(lo+hi)/2@R.T
    # Validated furniture has width > depth. Orientation is inherently modulo pi.
    if dimensions[0]>dimensions[1]:yaw+=np.pi/2;dimensions=dimensions[::-1]
    yaw=(yaw+np.pi/2)%np.pi-np.pi/2
    # Resolve front/back only when an elevated central backrest is visible.
    # Side armrests are excluded by the central-width band; no family label is used.
    direction_R=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    above=world[valid & (world[...,2]>height+.06) & (world[...,2]<height+.6)]
    local_above=(above[:,:2]-center)@direction_R
    central=np.abs(local_above[:,1])<dimensions[1]/2-.06
    near=np.abs(local_above[:,0])<dimensions[0]/2+.15
    rear=int((central&near&(local_above[:,0]<-dimensions[0]/2+.08)).sum())
    front=int((central&near&(local_above[:,0]>dimensions[0]/2-.08)).sum())
    resolved=max(rear,front)>100 and max(rear,front)>3*max(1,min(rear,front))
    heading=yaw+(np.pi if resolved and front>rear else 0)
    heading=(heading+np.pi)%(2*np.pi)-np.pi
    corners=np.array([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])@R.T
    return dict(surface_center=[float(center[0]),float(center[1]),height],depth=float(dimensions[0]),width=float(dimensions[1]),yaw=float(yaw),heading=float(heading),direction_resolved=bool(resolved),backrest_evidence_counts=[rear,front],available_area_m2=area,visible_points=len(xy),polygon=corners.tolist(),extent_kind='visible unobstructed top surface, not full hidden furniture bounds')

def estimate_body_frame(depth,K,mount_position,mount_quaternion,gravity_body):
    """Estimate floor and seating area from depth, fixed extrinsics and IMU gravity.

    No camera world pose, robot root pose, furniture labels or simulator state.
    Output is in the gravity-aligned initial body frame, with floor-relative height.
    """
    g=np.asarray(gravity_body);up=-g/np.linalg.norm(g)
    level,_=Rotation.align_vectors(np.array([[0.,0.,1.]]),up[None])
    rotation=level*Rotation.from_quat(mount_quaternion);position=level.apply(mount_position)
    d=np.asarray(depth).squeeze();v,u=np.mgrid[:d.shape[0],:d.shape[1]];valid=np.isfinite(d)&(d>.1)&(d<4)
    safe=np.where(valid,d,0);points=np.stack(((u-K[0,2])*safe/K[0,0],(v-K[1,2])*safe/K[1,1],safe),-1)@rotation.as_matrix().T+position
    normal=np.cross(points[1:-1,2:]-points[1:-1,:-2],points[2:,1:-1]-points[:-2,1:-1]);norm=np.linalg.norm(normal,axis=-1)
    ok=valid[1:-1,1:-1]&valid[1:-1,2:]&valid[1:-1,:-2]&valid[2:,1:-1]&valid[:-2,1:-1]
    heights=points[1:-1,1:-1,2][ok&(np.abs(normal[...,2])>.97*np.maximum(norm,1e-10))]
    histogram,edges=np.histogram(heights,bins=np.arange(-1.5,-.3,.005))
    from scipy.signal import find_peaks
    peaks,_=find_peaks(histogram,height=max(100,histogram.max()*.025),distance=4)
    if not len(peaks):raise ValueError('Floor not observable; no height fallback to simulator')
    index=peaks[0];floor=float(np.median(heights[(heights>=edges[index])&(heights<edges[index+1])]))
    shifted=position.copy();shifted[2]-=floor
    est=estimate(depth,K,shifted,rotation.as_quat())
    # Sitting requires a visible contiguous landing patch, not a small backrest
    # top or a bounding box spanning an occluded region. This gate is restricted
    # to the already-aligned milestone; it does not perform approach/alignment.
    if est['depth'] < .20 or est['width'] < .35 or est['available_area_m2'] < .08:
        raise ValueError('No sufficiently large visible seating surface')
    plane=points[valid].copy();plane[:,2]-=floor
    plane=plane[np.abs(plane[:,2]-est['surface_center'][2])<.006]
    center_x=est['surface_center'][0]
    histogram,_,_=np.histogram2d(plane[:,0],plane[:,1],bins=(np.linspace(center_x-.10,center_x+.10,11),np.linspace(-.12,.12,13)))
    coverage=float((histogram>=2).mean())
    if coverage < .75:
        raise ValueError('Aligned landing patch is occluded or not observable')
    est['aligned_landing_patch_coverage']=coverage
    est['floor_height_body_m']=floor
    est['position_initial_body_level']=est['surface_center'][:2]+[est['surface_center'][2]+floor]
    est['confidence']=float(min(1.,est['visible_points']/1500))
    return est

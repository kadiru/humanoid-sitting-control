"""Two current-depth edge associations, without fixture coordinates.

Selected plane component and farther-neighbour rays define observed boundaries.
Nearer robot occlusion is never a furniture edge. The left/right observed flanks
must each support a straight line with independently agreeing half segments.
"""
import numpy as np
from scipy.ndimage import label
from edge import classify_depth

def fit_line(points):
    center=points.mean(0)
    _,_,vh=np.linalg.svd(points-center,full_matrices=False)
    n=vh[-1]
    if n[0]<0:n=-n
    return np.r_[n,-center@n]

def fit_flanks(segments):
    s=np.asarray(segments,float).reshape(-1,10)
    mids=(s[:,:2]+s[:,3:5])/2
    edges=[];checks={};details=[]
    for side in [-1,1]:
        selected=(side*mids[:,1]>=.06)&(side*mids[:,1]<=.30)
        a,b=s[selected,:2],s[selected,3:5];p=mids[selected]
        if len(p)<20:return dict(passed=False,reason='insufficient_two_edge_visibility',count=len(p)),None
        order=np.argsort(p[:,1]);first,second=np.array_split(order,2)
        line=fit_line(p);l1=fit_line(p[first]);l2=fit_line(p[second])
        angle=float(np.rad2deg(np.arccos(np.clip(l1[:2]@l2[:2],-1,1))))
        residual=float(np.max(abs(p@line[:2]+line[2])))
        width=float(np.max(abs((b-a)@line[:2])))
        center=p.mean(0);disagreement=float(abs(np.r_[center,1]@(l1-l2)))
        straddle=float(np.maximum(a@line[:2]+line[2],-(b@line[:2]+line[2])).max())
        passed=(angle<=1. and disagreement<=.003 and residual<=.003 and width<=.006 and straddle<=.003)
        checks[str(side)]=bool(passed)
        edges.append(line)
        details.append(dict(side=side,count=len(p),line=line.tolist(),angle_half_difference_deg=angle,
            half_offset_difference_m=disagreement,max_midpoint_residual_m=residual,max_pixel_width_m=width,
            max_bracket_violation_m=straddle,observed_y_range=[float(p[:,1].min()),float(p[:,1].max())]))
    lines=np.asarray(edges);angle=float(np.rad2deg(np.arccos(np.clip(lines[0,:2]@lines[1,:2],-1,1))))
    checks['two_distinct_forward_edges']=bool(30<=angle<=120 and np.min(lines[:,0])>.25)
    if not checks['two_distinct_forward_edges']:return dict(passed=False,tests=checks,edges=details),None
    tip=np.linalg.solve(lines[:,:2],-lines[:,2])
    checks['tip_between_observed_flanks']=bool(abs(tip[1])<.06)
    # Keep6mm geometric uncertainty in addition to20mm required contact inset.
    return dict(passed=all(checks.values()),tests=checks,edges=details,tip_xy_m=tip.tolist(),
        outward_lines=lines.tolist(),edge_uncertainty_m=.006,actor_truth_input=False,
        tip_directly_seen=False,interpretation='Intersection of two measured straight flanks; independent mesh association required.'),lines

def detect(depth,K,camera_in_body,normal,offset,level,selected_pixels):
    depth=np.asarray(depth);h,w=depth.shape;v,u=np.mgrid[:h,:w]
    rays=np.stack([(u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],np.ones_like(u)],-1)
    T=np.asarray(camera_in_body);directions=rays@T[:3,:3].T;n=np.asarray(normal)
    den=directions@n
    expected=np.divide(-(T[:3,3]@n+offset),den,out=np.full((h,w),np.inf),where=abs(den)>1e-9)
    on,occluded,beyond=classify_depth(depth,expected)
    labels,count=label(on);ij=np.asarray(selected_pixels,int)
    sizes=np.bincount(labels[ij[:,0],ij[:,1]],minlength=count+1);sizes[0]=0
    if not sizes.any():return dict(passed=False,reason='no_selected_component'),{}
    plane=labels==sizes.argmax();surface=(T[:3,3]+directions*expected[...,None])@np.asarray(level).T
    rows=[];rejected=0
    for dv,du in [(1,0),(-1,0),(0,1),(0,-1)]:
        a0,a1=max(1,1-dv),min(h-1,h-1-dv);b0,b1=max(1,1-du),min(w-1,w-1-du)
        sa=(slice(a0,a1),slice(b0,b1));sb=(slice(a0+dv,a1+dv),slice(b0+du,b1+du))
        p,q=surface[sa],surface[sb]
        # Near facing boundary only; reject far edges and side-wall transitions.
        outward=(q[...,0]-p[...,0])>1e-5
        boundary=plane[sa]&~plane[sb]&outward
        rejected+=int((boundary&occluded[sb]).sum())
        rr,cc=np.nonzero(boundary&beyond[sb])
        for r,c in zip(rr,cc):rows.append([*p[r,c],*q[r,c],r+a0,c+b0,r+a0+dv,c+b0+du])
    segments=np.asarray(rows,float).reshape(-1,10)
    report,lines=fit_flanks(segments)
    report.update(dropoff_segments=len(segments),rejected_occlusion_segments=rejected,selected_plane_pixels=int(plane.sum()))
    return report,dict(edge_segments=segments,plane_mask=plane,occlusion_mask=occluded,background_mask=beyond)

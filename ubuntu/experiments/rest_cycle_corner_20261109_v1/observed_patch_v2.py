"""Depth-supported polygon with local normals, holes and clipped boundaries.

The original plane selector admits a5mm band. Its convex hull bridged visible
grooves. This representation uses the3mm depth-plane test AND5degree local
normal test, and retains holes. Whole-region mesh coverage stays mandatory.
"""
import numpy as np
from scipy.ndimage import binary_erosion
from shapely.geometry import Polygon
from shapely.ops import unary_union

def build(frame,T,level):
    d=np.asarray(frame['depth']);K=frame['K'];g=frame['geometry'];h,w=d.shape
    v,u=np.mgrid[:h,:w];ray=np.stack([(u+.5-K[0,2])/K[0,0],(v+.5-K[1,2])/K[1,1],np.ones_like(u)],-1)
    directions=ray@T[:3,:3].T;p=T[:3,3]+directions*d[...,None];n=np.asarray(g[2:5])
    den=directions@n;expected=np.divide(-(T[:3,3]@n+g[5]),den,out=np.full((h,w),np.inf),where=abs(den)>1e-9)
    on=np.isfinite(d)&(d>.08)&(abs(d-expected)<=.003)
    local=np.zeros_like(p);local[1:-1,1:-1]=np.cross(p[1:-1,2:]-p[1:-1,:-2],p[2:,1:-1]-p[:-2,1:-1])
    norm=np.linalg.norm(local,axis=2);agreement=abs(local@n)/np.maximum(norm,1e-15)
    mask=binary_erosion(on&(agreement>=np.cos(np.deg2rad(5)))&(norm>1e-12),structure=np.ones((3,3)))
    nl=level@n;nl/=np.linalg.norm(nl);e=np.array([1.,0,0])-nl*nl[0];e/=np.linalg.norm(e);b=np.cross(nl,e)
    def project(pixels):
        r=np.c_[(pixels[:,0]-K[0,2])/K[0,0],(pixels[:,1]-K[1,2])/K[1,1],np.ones(len(pixels))]@T[:3,:3].T
        t=-(T[:3,3]@n+g[5])/(r@n);p=(T[:3,3]+r*t[:,None])@level.T
        return np.c_[p@e,p@b]
    parts=[]
    for row in range(h):
        ends=np.flatnonzero(np.diff(np.r_[False,mask[row],False].astype(int)))
        for first,last in ends.reshape(-1,2):
            parts.append(Polygon(project(np.array([[first,row],[last,row],[last,row+1],[first,row+1]]))))
    if not parts:raise ValueError('NO_LOCAL_PLANAR_PIXELS')
    region=unary_union(parts).buffer(-.006,join_style=2)
    components=list(region.geoms) if region.geom_type=='MultiPolygon' else [region]
    components=[p for p in components if p.geom_type=='Polygon' and p.area>=.002]
    if not components or len(components)>32:raise ValueError('OBSERVED_SUPPORT_COMPONENT_COUNT')
    # Select the visible support component nearest the robot in forward x.
    # Selection uses only currentdepth; no mesh result can change this choice.
    def front(poly):
        xy=np.asarray(poly.exterior.coords);xyz=xy[:,0,None]*e+xy[:,1,None]*b-float(g[5])*nl
        return xyz[:,0].max()
    poly=max(components,key=front)
    return dict(e=e.tolist(),b=b.tolist(),vertices_uv=list(map(list,poly.exterior.coords[:-1])),
        holes_uv=[list(map(list,r.coords[:-1])) for r in poly.interiors],area_m2=float(poly.area),
        eligible_components=len(components),uncertainty_m=.006,actor_truth_input=False,hidden_boundary_extrapolated=False,
        source='current3mmplane and5deg local normals; rasterpolygon retainingholes;6mmerosion',convex=False),mask

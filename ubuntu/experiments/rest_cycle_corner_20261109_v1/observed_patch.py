"""Conservative directly observed convex support polygon; no hidden-edge fit."""
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def build(points,normal,uncertainty=.006):
    normal=np.asarray(normal,float);normal=normal/np.linalg.norm(normal)
    e=np.array([1.,0,0])-normal*normal[0];e/=np.linalg.norm(e);b=np.cross(normal,e)
    uv=np.c_[points@e,points@b];h=ConvexHull(uv)
    poly=Polygon(uv[h.vertices]).buffer(-uncertainty,join_style=2)
    if poly.is_empty or poly.geom_type!='Polygon':raise ValueError('NO_UNCERTAINTY_ERODED_PATCH')
    vertices=np.asarray(poly.exterior.coords[:-1]);eq=ConvexHull(vertices).equations
    return dict(e=e.tolist(),b=b.tolist(),vertices_uv=vertices.tolist(),equations=eq.tolist(),
        observed_hull_uv=uv[h.vertices].tolist(),uncertainty_m=uncertainty,actor_truth_input=False,
        hidden_boundary_extrapolated=False,requires_independent_triangle_coverage=True)

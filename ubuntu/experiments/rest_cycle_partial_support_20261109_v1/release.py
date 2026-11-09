"""Finite-surface separation certificate. It never estimates positive load.

Requires an admitted front-edge upper bound in current gravity-aligned body
coordinates and all relevant collider vertices. World pose/contact truth are
not inputs. Unknown geometry cannot be silently promoted to zero support.
"""
import numpy as np


def separation(*, vertices, plane_normal, plane_offset, front_upper,
               front_error, normal_error, qualified_edge, geometry_age,
               plane_center=(0.,0.,0.), normal_angle_error=.03):
    names=('pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link')
    if (not qualified_edge or geometry_age<0 or geometry_age>.21 or
        not all(k in vertices for k in names)):
        return dict(state='unknown',load_upper_bw=None,reason='unqualified_current_geometry')
    n=np.asarray(plane_normal,float)
    values=[front_upper,front_error,normal_error,normal_angle_error,plane_offset,*n,*plane_center]
    if not np.isfinite(values).all() or min(front_error,normal_error,normal_angle_error)<0:
        return dict(state='unknown',load_upper_bw=None,reason='invalid_geometry_bounds')
    if abs(np.linalg.norm(n)-1)>.001 or n[2]<.97:
        return dict(state='unknown',load_upper_bw=None,reason='not_horizontal_support')
    bodies={}
    for body in vertices:
        p=np.asarray(vertices[body],float)
        if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
            return dict(state='unknown',load_upper_bw=None,reason='invalid_body_vertices')
        # A whole convex collider is separated if it is entirely above the
        # support plane OR entirely forward of the support's front bound.
        # Per-vertex OR would be UNSOUND for an edge-crossing convex hull.
        reserve=normal_error+normal_angle_error*np.linalg.norm(p-np.asarray(plane_center),axis=1)
        above=float((p@n+plane_offset-reserve).min())
        forward=float(p[:,0].min())-front_upper-front_error
        bodies[body]=dict(above_margin_m=above,forward_margin_m=forward,
                          separated=above>.006 or forward>.006)
    clear=all(r['separated'] for r in bodies.values())
    return dict(state='geometrically_separated' if clear else 'unknown',
        load_upper_bw=0. if clear else None,load_lower_bw=0. if clear else None,
        bodies=bodies,reason='finite_surface_separation' if clear else 'possible_contact',
        positive_contact_claim=False,physical_sensor_validation_required=True)

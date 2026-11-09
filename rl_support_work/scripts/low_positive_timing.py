"""Preserve the spatial reference path, slow only its touchdown region."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from positive_slope_planner_v2 import plan as previous

def plan(*args,clock_knots,progress_knots,**kwargs):
    commands,qs,a=previous(*args,**kwargs,target_x=-.18,target_pitch=.25,contact_gap=.001,land_fraction=.6,travel_fraction=1.)
    if not a.get('positive_candidate_applied',False):return commands,qs,a
    grid=np.linspace(0,1,25);progress=PchipInterpolator(clock_knots,progress_knots)(grid)
    assert progress[0]==0 and progress[-1]==1 and (np.diff(progress)>0).all()
    def resample(x):
        x=np.asarray(x);return np.stack([np.interp(progress,grid,x[:,j]) for j in range(x.shape[1])],1)
    c=resample(commands);q=resample(qs);root=resample(a['root_path'])
    root[:,3:]/=np.linalg.norm(root[:,3:],axis=1,keepdims=True)
    c[0]=commands[0];c[-1]=commands[-1];q[0]=qs[0];q[-1]=qs[-1]
    a.update(root_path=root.tolist(),original_static_path=a.pop('static_path'),
             time_warp=dict(clock=clock_knots,progress=progress_knots,sampled_progress=progress.tolist(),duration_s=5.,spatial_endpoints_exact=True),
             certification='Resampled original path; full physical validation required')
    return c,q,a

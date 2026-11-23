"""Modify only the final approach, using a constrained contact pose fitted to the physical trace."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from run import P,read,write
from stance23_planner import smooth
def make():
    i=30;spec=read(P/'first_touch_trim_30.json')
    assert spec['admitted'] and spec['path_min'][1]>.004 and spec['path_min'][2]>.003
    f=P/'sole_moment_16/evaluation';j=14
    tr=np.load(f/'trajectory.npz');diag=np.load(f/'corridor_feedback.npz')['data'];old=np.load(P/'contact_corridor_path_30.npz')['motion']
    delta=np.load(P/'first_touch_trim_30.npz')['delta']
    stop=spec['event']-1;active=np.isfinite(diag[:stop-39,j,1]);t=np.arange(40,stop+1)[active]
    ph=diag[t-40,j,1].astype(int);p,ix=np.unique(ph,return_index=True)
    final=p[-1]+68;clock=np.arange(1000);ticks=np.interp(clock,p,t[ix],left=t[0],right=stop)+np.maximum(0,clock-p[-1])
    ticks=np.minimum(ticks,999).astype(int)
    out=old[clock.clip(0,p[-1])].copy()
    out[:,:23]+=delta[ticks]
    w=smooth((ticks-stop+143)/110)
    out[:,69:72]+=w[:,None]*spec['offset']
    out[:,71]+=.005*w*(1-smooth((ticks-stop+17)/48))-.0015*smooth((ticks-stop-13)/54)
    rot=R.from_matrix(out[:,72:81].reshape(-1,3,3));out[:,72:81]=(R.from_rotvec(w[:,None]*spec['rotation'])*rot).as_matrix().reshape(-1,9)
    np.savez_compressed(P/'measured_corner_path_30.npz',motion=out)
    write(P/'measured_corner_timing.json',dict(source_event=spec['event'],source_phase=int(p[-1]),end=int(final)))
if __name__=='__main__':make()

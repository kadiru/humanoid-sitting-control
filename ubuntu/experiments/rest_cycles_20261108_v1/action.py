"""Original absolute-target filter/rate/margin contract, including fingers."""
import numpy as np


def apply(previous,desired,lower,upper):
    previous=np.asarray(previous,np.float32);desired=np.asarray(desired,np.float32)
    if previous.shape!=(37,) or desired.shape!=(23,) or not np.isfinite(desired).all() or not np.isfinite(previous).all():raise ValueError('INVALID_ACTION')
    lo=np.asarray(lower,np.float32)+np.float32(.03);hi=np.asarray(upper,np.float32)-np.float32(.03)
    if lo.shape!=(23,) or hi.shape!=(23,) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo>hi):raise ValueError('INVALID_BOUNDS')
    if np.any(previous[:23]<lo-2e-7) or np.any(previous[:23]>hi+2e-7):raise ValueError('PREVIOUS_COMMAND_OUT_OF_BOUNDS')
    bounded=np.clip(desired,lo,hi)
    delta=np.clip(np.float32(.1)*(bounded-previous[:23]),-np.float32(.003),np.float32(.003))
    result=previous.copy();result[:23]+=delta
    if np.any(abs(result[:23]-previous[:23])>.003+2e-7):raise ValueError('ACTION_RATE')
    assert np.array_equal(result[23:],previous[23:])
    return result

"""Unchanged full-original numerical gate, also checked on augmented system."""
def check_point(M,E,a,w,marginal):
 import numpy as np
 y=-marginal['inequality'];nu=-marginal['equality'];up=-marginal['upper'];low=marginal['lower']
 values=M@w-a['rhs'];eq=E@w-a['equality_rhs']
 station=a['cost']+M.T@y+E.T@nu+up-low
 lf=np.isfinite(a['bounds'][:,0]);uf=np.isfinite(a['bounds'][:,1])
 primal=max(0.,float(values.max(initial=0)),float(abs(eq).max(initial=0)),
   float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 dual=float(abs(station).max(initial=0))
 gap=abs(float(a['cost']@w+a['rhs']@y+a['equality_rhs']@nu+a['bounds'][uf,1]@up[uf]-a['bounds'][lf,0]@low[lf]))
 finite=all(np.isfinite(v).all() for v in [w,y,nu,up,low,values,eq,station,np.asarray([primal,dual,gap])])
 signs=bool(np.all(y>=0) and np.all(up>=0) and np.all(low>=0) and np.all(up[~uf]==0) and np.all(low[~lf]==0))
 return dict(passed=bool(finite and signs and max(primal,dual,gap)<=1e-9),primal=primal,dual=dual,
   absolute_gap=gap,finite=bool(finite),dual_signs=signs),dict(inequality=values,equality=eq,stationarity=station)


def added_duals_zero(values):
 import numpy as np
 a=np.asarray(values)
 return bool(a.shape==(2,) and np.isfinite(a).all() and np.all(a==0))

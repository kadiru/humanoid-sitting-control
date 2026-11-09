"""One resource-bounded transformed LP call. No retries or alternate solver."""
from pathlib import Path
import sys,json,time,resource
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
def main():
 p=Path(sys.argv[1]);limit=40*1024**3
 resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
 a=dict(np.load(p/'arrays.npz'));M=sparse.load_npz(p/'matrix.npz');E=sparse.load_npz(p/'equality.npz')
 (p/'scipy_started.json').write_text(json.dumps(dict(actual_lp_call=True,method='highs-ipm',time_limit_s=45.,address_space_bytes=limit)))
 start=time.monotonic()
 r=linprog(a['cost'],A_ub=M,b_ub=a['rhs'],A_eq=E,b_eq=a['equality_rhs'],bounds=a['bounds'],method='highs-ipm',options=dict(time_limit=45.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
 if r.x is not None:np.save(p/'witness.npy',r.x)
 meta=dict(status=int(r.status),message=str(r.message),candidate=r.x is not None,success=bool(r.success),iterations=int(r.nit),crossover_iterations=int(getattr(r,'crossover_nit',0)),elapsed_s=time.monotonic()-start,actual_lp_call=True)
 (p/'result.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
if __name__=='__main__':main()

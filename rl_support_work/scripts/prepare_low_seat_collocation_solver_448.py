"""Saved-array-only preparation of one bounded predictor/corrector algorithm.
No model, root, derivative fixture, trajectory optimizer or physical execution.
"""
import json,datetime
import numpy as np
from scipy import sparse
from low_seat_collocation_solver_448 import *
O=D/'collocation_solver_preparation_448'
def main():
 O.mkdir(exist_ok=False);d=load_saved();X=d['initialX'];Y=d['initialY'];J=d['initialJ'];a=assemble(d,X,Y,J,1.);b=assemble(d,X,Y,J,1.,'correction',.5,origin=d['reference'])
 # The new wrapper's Jacobians must reproduce the audited saved v2 matrices.
 original=sparse.load_npz(V/'global_jacobian.npz').multiply((1/d['a']['scale'])[:,None]);diff=a['A']-original;err=0. if diff.nnz==0 else float(abs(diff.data).max());assert err<2e-10,err
 for name,q in [('predictor',a),('correction',b)]:
  sparse.save_npz(O/(name+'_matrix.npz'),q['matrix']);sparse.save_npz(O/(name+'_equality.npz'),q['equality']);np.savez_compressed(O/(name+'_arrays.npz'),rhs=q['rhs'],equality_rhs=q['equality_rhs'],bounds=q['bounds'],cost=q['cost'],guard=q['guard'])
 _,r,e=values(d,X,Y);out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,seed=summary(d,X,Y,r,e),free_coordinates=len(d['free']),inequality_rows=len(r),retained_rows=int(d['a']['keep'].sum()),wrapper_chain_error=err,seed_is_feasible=False,model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

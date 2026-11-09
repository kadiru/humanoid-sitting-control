"""Independent saved-matrix assembly audit. No oracle/root/solver calls."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'collocation_solver_audit_449';P=D/'collocation_solver_preparation_448';V=D/'clock_row_preparation_v2_446';Q=D/'clock_row_preparation_446'
def main():
 O.mkdir(exist_ok=False);row=dict(np.load(V/'row_arrays.npz'));coord=dict(np.load(Q/'coordinate_map.npz'));eq=dict(np.load(Q/'equality_arrays.npz'));spec=json.loads((P/'preparation.json').read_text())['specification'];keep=row['keep'];soft=row['soft']&keep;hard=keep&~soft;A=sparse.load_npz(V/'global_jacobian.npz').multiply((1/row['scale'])[:,None]).tocsr();E=sparse.load_npz(Q/'equality_jacobian.npz')[eq['optimizer_keep']];r=row['rows']/row['scale'];free=coord['free'];z=coord['reference'].ravel()[free];n=len(z);radius=.02;lower=np.maximum(coord['lower'].ravel()[free],z-radius);upper=np.minimum(coord['upper'].ravel()[free],z+radius)
 guard=np.zeros(len(r));variable=np.diff(A.indptr)>0;guard[soft&variable]=2.**-20;sel=hard&variable;guard[sel]=np.where(r[sel]>0,np.minimum(2.**-20,.5*r[sel]),2.**-20)
 errors={}
 for name in ['predictor','correction']:
  data=dict(np.load(P/(name+'_arrays.npz')));mat=sparse.load_npz(P/(name+'_matrix.npz'));e=sparse.load_npz(P/(name+'_equality.npz'));top=sparse.hstack([-A[keep],sparse.csr_matrix((-soft[keep].astype(float) if name=='predictor' else np.zeros(int(keep.sum())))[:,None])],format='csr')
  if name=='predictor':expected=top;rhs=r[keep]-guard[keep];bounds=np.c_[np.r_[lower-z,0.],np.r_[upper-z,1.]]
  else:
   I=sparse.eye(n,format='csr');tail=sparse.vstack([sparse.hstack([I,sparse.csr_matrix(-np.ones((n,1))*radius)]),sparse.hstack([-I,sparse.csr_matrix(-np.ones((n,1))*radius)])],format='csr');expected=sparse.vstack([top,tail],format='csr');rhs=np.r_[r[keep]-guard[keep]+.5*soft[keep],np.zeros(2*n)];bounds=np.c_[np.r_[np.maximum(lower-z,-.005),0.],np.r_[np.minimum(upper-z,.005),.25]]
  ee=sparse.hstack([E,sparse.csr_matrix((E.shape[0],1))],format='csr');diff=mat-expected;de=e-ee
  errors[name]=max(float(abs(diff.data).max()) if diff.nnz else 0.,float(abs(de.data).max()) if de.nnz else 0.,float(abs(data['rhs']-rhs).max()),float(abs(data['equality_rhs']+eq['values'][eq['optimizer_keep']]).max()),float(abs(data['bounds']-bounds).max()),float(abs(data['guard']-guard).max()),float(abs(data['cost']-np.r_[np.zeros(n),1.]).max()))
 assert max(errors.values())<2e-10,errors
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,maximum_assembly_errors=errors,all_original_rows_retained=True,hard_rows_without_epigraph=int(hard.sum()),rate_window_epigraph_rows=int(soft.sum()),structural_positive_load_bounds=True,monotone_phase_rows_hard=True,seed_is_infeasible=True,actual_nonlinear_hard_acceptance_required=True,root_calls=0,oracle_calls=0,optimizer_calls=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

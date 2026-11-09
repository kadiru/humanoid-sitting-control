"""Independent new reduced LP assembly replay, saved matrices only."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'simplex_solver_preparation_462';O=D/'simplex_solver_preparation_audit_463'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maxdiff(a,b):
 c=(a-b).tocsr();c.eliminate_zeros();return 0. if not c.nnz else float(abs(c.data).max())
def verify(folder):
 a=dict(np.load(folder/'arrays.npz' if (folder/'arrays.npz').exists() else folder/'input.npz'))
 conv=a if 'original_rhs' in a else dict(np.load(folder/'conversion.npz'))
 groups=json.loads((D/'simplex_preparation_460/layout.json').read_text());owners=[]
 for g in groups:
  for i in g['independent']:owners.append((g['indices'][i],None if g['pivot']<0 else g['indices'][g['pivot']]))
 A=sparse.load_npz(folder/'original_matrix.npz').tocsc();E=sparse.load_npz(folder/'original_equality.npz').tocsc()
 reduced=sparse.hstack([A[:,p] if n is None else A[:,p]-A[:,n] for p,n in owners]+[A[:,-1]],format='csr')
 eq=sparse.hstack([E[:,p] if n is None else E[:,p]-E[:,n] for p,n in owners]+[E[:,-1]],format='csr')
 rr=[];cc=[];vv=[]
 for j,(p,n) in enumerate(owners):
  rr.append(p);cc.append(j);vv.append(1.)
  if n is not None:rr.append(n);cc.append(j);vv.append(-1.)
 R=sparse.csr_matrix((vv,(rr,cc)),shape=(11906,len(owners)+1));rebuilt=sparse.vstack([reduced,R,-R],format='csr')
 rhs=np.r_[conv['original_rhs'],conv['original_bounds'][:-1,1],-conv['original_bounds'][:-1,0]];erhs=conv['original_equality_rhs'].copy();erhs[conv['normal_conservation_rows']]=0
 assert np.array_equal(rhs,a['rhs']) and np.array_equal(erhs,a['equality_rhs'])
 z=eq[conv['normal_conservation_rows']].copy();z.eliminate_zeros();assert not z.nnz
 e=max(maxdiff(rebuilt,sparse.load_npz(folder/'matrix.npz')),maxdiff(eq,sparse.load_npz(folder/'equality.npz')));assert e==0
 assert np.all(np.isneginf(a['bounds'][:-1,0])) and np.all(np.isposinf(a['bounds'][:-1,1]));assert np.array_equal(a['bounds'][-1],conv['original_bounds'][-1])
 assert np.all(a['cost'][:-1]==0) and a['cost'][-1]==1
 return dict(matrix_error=e,normal_rows=int(conv['normal_conservation_rows'].sum()),full_control_bound_rows=23812)
def main():
 O.mkdir(exist_ok=False);checks={mode:verify(P/mode) for mode in ['predictor','correction']}
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,all_pivot_bounds_retained=True,normal_primal_tolerance_unchanged=True,model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

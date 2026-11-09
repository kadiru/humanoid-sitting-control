"""One 45s-total full-size Schur/HSD child; no alternative or retry."""
from pathlib import Path
import sys,time,resource,json
limit=40*1024**3
resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
import numpy as np
from scipy import sparse
from low_seat_schur_solver_532 import solve_lp,normalize,sha,Terminal
def same_matrix(A,B):
 if A.shape!=B.shape:return False
 A=A.tocsr(copy=True);B=B.tocsr(copy=True);A.sort_indices();B.sort_indices()
 return np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
from certify_low_seat_schur_input_533 import certify
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def main():
 out=Path(sys.argv[1]);start=time.monotonic()
 (out/'hsd_started.json').write_text(json.dumps(dict(actual_lp_call=True,method='hsd-schur-dr',time_limit_s=45.,address_space_bytes=limit)))
 try:
  a=dict(np.load(out/'arrays.npz'));A=sparse.load_npz(out/'matrix.npz').tocsr();E=sparse.load_npz(out/'equality.npz').tocsr();info=read(out.parent/'projection_scope.json');role=info['role'];base=D/'torque_projection_preparation_522'/role;ba=dict(np.load(base/'arrays.npz'))
  inherited=same_matrix(A,sparse.load_npz(base/'matrix.npz')) and same_matrix(E,sparse.load_npz(base/'equality.npz')) and all(np.array_equal(a[k],ba[k]) for k in a)
  if inherited:
   P=D/'schur_hsd_preparation_530'/role
   assert read(D/'schur_hsd_audit_531/independent_audit.json')['passed']
   ar=dict(np.load(P/'arrays.npz'));norm=dict(matrix=sparse.load_npz(P/'matrix.npz'),original=sparse.load_npz(P/'cone_original.npz'),**ar)
   local=ar['local'];separator=ar['separator'];ptr=ar['block_ptr']
   proof=dict(passed=True,inherited=True,source=str(P),source_proof_sha256=sha(D/'schur_hsd_audit_531/SHA256.json'),source_input_hashes={str(q):sha(q) for q in [base/'matrix.npz',base/'arrays.npz',base/'equality.npz',P/'matrix.npz',P/'cone_original.npz',P/'arrays.npz']},old_scaling_or_proof_repeated=False)
  else:
   norm=normalize(A,a['rhs'],a['cost'],a['bounds'],E,a['equality_rhs']);old=np.load(D/'schur_hsd_preparation_530/primary/arrays.npz');local=old['local'];separator=old['separator'];ptr=old['block_ptr']
   proof=certify(A,a,E,norm,local,separator,ptr);proof.update(inherited=False,source=str(out/'normalization'))
   P=out/'normalization';P.mkdir();sparse.save_npz(P/'matrix.npz',norm['matrix']);sparse.save_npz(P/'cone_original.npz',norm['original']);np.savez_compressed(P/'arrays.npz',**{k:v for k,v in norm.items() if k not in ['matrix','original']},local=local,separator=separator,block_ptr=ptr)
   proof['source_input_hashes']={str(q):sha(q) for q in P.iterdir()};proof['source_input_hashes'].update({str(q):sha(q) for q in [out/'matrix.npz',out/'arrays.npz',out/'equality.npz']})
  (out/'normalization_proof.json').write_text(json.dumps(proof,indent=2))
  r=solve_lp(norm,local,separator,ptr,out,start)
 except (AssertionError,Terminal,MemoryError,ValueError,FloatingPointError) as exc:
  r=dict(status=1,message='input/resource failure:'+repr(exc),candidate=False,success=False,iterations=0,crossover_iterations=0,elapsed_s=time.monotonic()-start,actual_lp_call=True,statistics={},physical_admission=False,method='hsd-schur-dr')
  (out/'result.json').write_text(json.dumps(r,indent=2))
 print(json.dumps(r),flush=True)
if __name__=='__main__':main()

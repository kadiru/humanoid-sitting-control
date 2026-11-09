"""Independent support/copy proof and original-operator residual audit; no production import."""
from pathlib import Path
import hashlib,json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'arrow_operator_audit_529';P=D/'arrow_operator_preparation_528'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def equal_sparse(a,b):
 assert a.shape==b.shape
 for name in ['data','indices','indptr']:assert np.array_equal(getattr(a,name),getattr(b,name)),name
def relative(error,*terms):return float(np.max(abs(error),initial=0)/max(1.,*(np.max(abs(t),initial=0) for t in terms)))
def main():
 assert read(D/'arrow_operator_preparation_exit_528.json')['actual_exit']==0
 launch=read(D/'arrow_operator_launch_528.json')
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 O.mkdir(exist_ok=False);extra=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];results={}
 for role in ['primary','witness']:
  src=D/'torque_projection_preparation_522'/role;out=P/role
  A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz');arr=dict(np.load(src/'arrays.npz'));ret=np.load(src/'projection_ledger.npz')['retained_columns'];p=dict(np.load(out/'partition.npz'));z=dict(np.load(out/'operator_example.npz'));B=sparse.load_npz(out/'local.npz');C=sparse.load_npz(out/'separator.npz')
  initial=np.full(A.shape[1],-1,dtype=np.int64)
  for c,old in enumerate(ret):
   if old>=10540 and old<78790:
    f=int(extra[old-10540]);k=f%75
    if k<3 or 4<=k<16 or 72<=k<75:initial[c]=f//75
  assert np.array_equal(initial,p['initial_labels'])
  # Independent row support sets, rather than production's vector label test.
  promote=set();cross=[]
  for i in range(A.shape[0]):
   a,b=A.indptr[i:i+2];cs=[int(c) for c,v in zip(A.indices[a:b],A.data[a:b]) if v!=0 and initial[c]>=0]
   if len({int(initial[c]) for c in cs})>1:cross.append(i);promote.update(cs)
  assert np.array_equal(np.array(sorted(promote),dtype=np.int64),p['promoted']);assert np.array_equal(cross,p['cross_rows'])
  labels=initial.copy();labels[list(promote)]=-1;assert np.array_equal(labels,p['labels'])
  local=p['local'];sep=p['separator'];assert np.array_equal(np.sort(np.r_[local,sep]),np.arange(A.shape[1]));assert np.all(labels[local]>=0) and np.all(labels[sep]<0)
  equal_sparse(A[:,local].tocsr(),B);equal_sparse(A[:,sep].tocsr(),C);assert B.nnz+C.nnz==A.nnz
  ptr=p['block_ptr'];states=p['states'];assert ptr[0]==0 and ptr[-1]==len(local) and np.all(np.diff(ptr)>0)
  assert np.array_equal(labels[local],np.repeat(states,np.diff(ptr))) and len(np.unique(states))==len(states)
  owner=p['row_owner'];inverse=np.empty(len(local),dtype=np.int64)
  for k,(l,u) in enumerate(zip(ptr[:-1],ptr[1:])):inverse[l:u]=k
  for i in range(A.shape[0]):
   a,b=B.indptr[i:i+2];groups=set(inverse[B.indices[a:b][B.data[a:b]!=0]].tolist())
   assert len(groups)<=1 and (next(iter(groups)) if groups else -1)==owner[i]
  # LP objects are not rewritten: unchanged original paths/hash identities govern.
  assert E.shape[1]==A.shape[1] and arr['bounds'].shape==(A.shape[1],2) and len(arr['cost'])==A.shape[1]
  expected=np.frexp(np.maximum(1.,np.asarray(abs(A).sum(axis=1)).ravel()))[1];assert np.array_equal(expected,z['weight_exponents']);assert np.array_equal(np.ldexp(np.ones(A.shape[0]),-2*expected),z['weights'])
  assert np.all(z['weights']>0) and np.all(np.isfinite(z['weights']))
  w=z['weights'];fptr=z['factor_ptr'];factor_errors=[]
  bc=B.tocsc()
  for k,(l,u) in enumerate(zip(ptr[:-1],ptr[1:])):
   rows=np.unique(bc.indices[bc.indptr[l]:bc.indptr[u]]);bk=A[rows][:,local[l:u]];G=(bk.T@bk.multiply(w[rows,None])).toarray()+np.eye(u-l)
   F=z['factor_data'][fptr[k]:fptr[k+1]].reshape(u-l,u-l);assert np.all(np.diag(F)>0) and np.all(np.triu(F,1)==0)
   factor_errors.append(relative(F@F.T-G,G))
  y=z['separator_vector'];bl=z['local_rhs'];bg=z['separator_rhs'];u=z['local_response'];q=z['local_rhs_response']
  assert np.array_equal(y,((np.arange(len(sep))%17)-8).astype(float)/16)
  assert np.array_equal(bl,((np.arange(len(local))%13)-6).astype(float)/16) and np.array_equal(bg,((np.arange(len(sep))%19)-9).astype(float)/16)
  # Original unsplit A is used for every independent operator product.
  def apply(x):return x+A.T@(w*(A@x))
  v=np.zeros(A.shape[1]);v[local]=-u;v[sep]=y;kv=apply(v)
  t=np.zeros(A.shape[1]);t[local]=q;kt=apply(t)
  r=np.zeros(A.shape[1]);r[local]=z['recovered_local'];r[sep]=y;kr=apply(r)
  errors=dict(factor=max(factor_errors),local_homogeneous=relative(kv[local],y,u),schur_action=relative(kv[sep]-z['schur_action'],kv[sep],z['schur_action']),local_rhs=relative(kt[local]-bl,bl,q),condensed_rhs=relative(bg-kt[sep]-z['condensed_rhs'],bg,kt[sep]),recovery_local=relative(kr[local]-bl,bl,z['recovered_local']),recovery_separator=relative(kr[sep]-bg-(z['schur_action']-z['condensed_rhs']),kr[sep],bg,z['schur_action']))
  assert all(np.isfinite(e) and e<=1e-11 for e in errors.values()),errors
  assert np.array_equal(z['recovered_local'],q-u)
  np.savez_compressed(O/(role+'_verification.npz'),factor_relative_errors=factor_errors,original_operator_homogeneous=kv,original_operator_rhs=kt,original_operator_recovered=kr)
  results[role]=dict(coefficient_copy_error=0,row_support_proof=True,all_original_rows_retained=A.shape[0],all_original_nonzeros_retained=A.nnz,local_blocks=len(states),local_columns=len(local),separator_columns=len(sep),errors=errors,full_normal_or_schur_matrix_assembled=False,local_factorization_or_solve_repeated=False)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Exact support/copy proof and independent floating residual audit of two NEW algorithm-unused weighted Gram operator examples. No LP/Schur solve, no old mechanics inverse/proof/example repeated.',systems=results,exact_structure_theorem='No row touches two local blocks, so every off-diagonal local Gram block is structurally zero for arbitrary diagonal weights. Positive w,d imply SPD; the standard block factorization and Schur minimization identities follow algebraically. Numerical factors only checked for the saved example weights.',finite_precision_limit='Residual audit is not a directed-rounding certificate, uniform conditioning guarantee or proof for future changing weights.1e-11 is algebra-only; unchanged original1e-7 and physical gates are not replaced.',lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

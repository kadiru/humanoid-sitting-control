"""Independent proof of genuinely NEW source quotient/split/metric; reference539/541 proofs never rerun."""
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import json
from audit_low_seat_capped_pdhg_541 import metric_audit

def same(A,B):
 assert A.shape==B.shape
 A=A.tocsr();B=B.tocsr();assert np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def entries(A,i):
 st,en=A.indptr[i:i+2];return {int(j):F(float(v)) for j,v in zip(A.indices[st:en],A.data[st:en]) if v}
def prove(full,a,src,sa,ids,ledger,H,S,z,metric,out,check):
 same(src,full[ids]);assert np.array_equal(sa['rhs'],a['rhs'][ids]);chosen=set(map(int,ids));flags=np.zeros(full.shape[0],np.uint8)
 for i,j in enumerate(ledger['representative']):
  if i%128==0:check()
  if j<0:assert not entries(full,i) and F(float(a['rhs'][i]))>=0
  elif i!=j:
   assert int(j) in chosen;v=entries(full,i);u=entries(full,int(j));assert v.keys()==u.keys() and v;c=min(v);ratio=v[c]/u[c];assert ratio>0 and all(v[k]*u[c]==u[k]*v[c] for k in v) and F(float(a['rhs'][i]))>=ratio*F(float(a['rhs'][j]))
  else:assert int(j) in chosen
  flags[i]=1
 np.save(out/'new_implication_verified.npy',flags)
 h=z['hard_source_positions'];s=z['coupled_source_positions'];assert np.array_equal(np.sort(np.r_[h,s]),np.arange(src.shape[0]));same(H,src[h,:-1]);same(S,src[s,:-1]);e=float(z['objective_sign']);col=src[:,-1].toarray().ravel();assert np.all(col[h]==0) and np.all(col[s]==-e)
 assert np.array_equal(z['hard_rhs'],sa['rhs'][h]) and np.array_equal(z['coupled_rhs'],sa['rhs'][s]);assert np.array_equal(z['source_row_ids'],ids)
 for k in ['cost','bounds','equality_rhs']:assert np.array_equal(sa[k],a[k])
 assert np.array_equal(z['native_cost'],a['cost'][:-1]) and np.array_equal(z['native_bounds'],a['bounds'][:-1]);assert float(z['lower_v'])==min(e*a['bounds'][-1]) and float(z['upper_v'])==max(e*a['bounds'][-1])
 check();rv,cv,met=metric_audit(H,S,metric);check();np.savez(out/'new_metric_verified.npz',rows=rv,columns=cv)
 return dict(passed=True,new_quotient_proof_rows=len(flags),new_split_proof_rows=len(h)+len(s),new_metric=met,original_source_copy_error=0,old_proofs_repeated=False)

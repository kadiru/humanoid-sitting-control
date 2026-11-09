"""Independent scalar assembly of only the two frozen physical-gap targets."""
from common import *
import numpy as np
from scipy import sparse
from independent_input import csr,same

def prove_targets(P,J,actual,scale,G,rhs):
 saved=read(ATTR/'attribution.json');assert [r['row'] for r in saved['records']]==[214643,215327]
 entries=[];rr=[];cc=[];vv=[];bb=[]
 for k,(row,pairs) in enumerate(((214643,((622,-1.),(1273,1.))),(215327,((1400,-1.),(631,1.))))):
  entry=saved['records'][k];assert entry['group']=='mesh_normal_descent' and entry['scale']==float(scale[row])
  assert entry['actual_after']==float(actual[row])
  assert {(t['state'],t['coefficient']) for t in entry['terms']}==set(pairs)
  acc={}
  for state,sign in pairs:
   for j in range(24):
    v=(-sign*float(J[state,47,j]))/float(scale[row])
    if v==0:continue
    ii=state*75+j
    for pos in range(int(P.indptr[ii]),int(P.indptr[ii+1])):
     col=int(P.indices[pos]);term=v*float(P.data[pos])
     if term:acc[col]=acc.get(col,0.)+term
  for col in sorted(acc):
   rr.append(k);cc.append(col);vv.append(acc[col])
  bb.append(float(actual[row])-2.**-20)
  entries.append(dict(target_row=row,states=[x[0] for x in pairs],nnz=len(acc),source_actual_residual=float(actual[row]),row_scale=float(scale[row]),additional_normalized_inward_guard=2.**-20))
 expected=sparse.csr_matrix((vv,(rr,cc)),shape=(2,78791));csr(expected,G);same(np.array(bb),rhs)
 return dict(passed=True,targets=entries,new_gap_tangents=2,physical_output=47,epigraph_coefficients_exact_zero=True,original_LP_equivalence_not_claimed=True)

def prove_augmented(M,a,original,oa,G,rhs):
 from independent_input import csr,same
 n=original.shape[0]
 assert M.shape==(n+2,original.shape[1])
 csr(M[:n],original);csr(M[n:],G)
 same(a['rhs'],np.r_[oa['rhs'],rhs])
 for k in ['cost','bounds','equality_rhs']:same(a[k],oa[k])
 return dict(original_prefix_exact=True,appended_rows_exact=True,cost_and_bounds_unchanged=True)

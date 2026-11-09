"""Synthetic new-policy checks only; no study arrays or optimizer."""
import json,tempfile,time
from pathlib import Path
import numpy as np
from scipy import sparse
import common
import producer
import independent_return as auditor

def run():
 checks=[]
 try:common.contract()
 except PermissionError:checks.append('approval_gate_blocks_before_scientific_access')
 else:raise AssertionError('Unapproved execution admitted')
 rho=float.fromhex(common.RAW_HEX)
 original=dict(inequality=np.array([-0.,rho,-1.,0.]),equality=np.array([-0.]),lower=np.array([0.,0.]),upper=np.array([-0.,-1.]))
 raw={n:a.tobytes() for n,a in original.items()}
 corrected=producer.correct_named_multiplier(original,1,common.RAW_HEX)
 assert auditor.check_change(original,corrected,1,common.RAW_HEX)
 assert all(a.tobytes()==raw[n] for n,a in original.items())
 assert int(corrected['inequality'].view(np.uint64)[1])==0
 checks.append('only_named_entry_changes_original_untouched')
 for case in ('second_entry','negative_zero'):
  bad={n:a.copy() for n,a in corrected.items()}
  if case=='second_entry':bad['inequality'][2]=np.nextafter(bad['inequality'][2],0.)
  else:bad['inequality'][1]=-0.
  try:auditor.check_change(original,bad,1,common.RAW_HEX)
  except AssertionError:checks.append('auditor_rejects_'+case)
  else:raise AssertionError('Unauthorized correction passed')
 altered={n:a.copy() for n,a in original.items()};altered['inequality'][1]=np.nextafter(rho,np.inf)
 try:producer.correct_named_multiplier(altered,1,common.RAW_HEX)
 except AssertionError:checks.append('different_source_value_rejected')
 else:raise AssertionError('Unfrozen source value allowed')
 with tempfile.TemporaryDirectory(prefix='certificate_fixture_') as d:
  root=Path(d);n=78791;nr=837063;ne=1366
  point=np.zeros(n);point[-1]=.5;cost=np.zeros(n);cost[-1]=-1.
  bounds=np.c_[np.full(n,-np.inf),np.full(n,np.inf)];bounds[-1]=[0,.5]
  rhs=np.zeros(nr);rhs[0]=rhs[common.ROW]=1.
  arrays=dict(rhs=rhs,cost=cost,bounds=bounds,equality_rhs=np.zeros(ne));E=sparse.csr_matrix((ne,n))
  margins=dict(inequality=np.zeros(nr),equality=np.zeros(ne),lower=np.zeros(n),upper=np.zeros(n));margins['upper'][-1]=-1.;margins['inequality'][common.ROW]=rho;margins['inequality'][0]=-rho
  m=producer.correct_named_multiplier(margins,common.ROW,common.RAW_HEX)
  for scale,want in [(1.,True),(1e12,False)]:
   out=root/str(scale);out.mkdir();producer.O=out;auditor.O=out
   M=sparse.csr_matrix(([scale,scale],([0,common.ROW],[0,0])),shape=(nr,n))
   p=producer.numerical_check(M,E,arrays,point,m);a=auditor.audit_kkt(M,E,arrays,point,m)
   assert p==a and p['passed']==want and p['dual_signs']
   checks.append('small_certificate_change_passes_full_gate' if want else 'tiny_multiplier_with_large_coefficients_rejected')
  # Exact first-margin branch: a boundary point must still fail after certificate correction.
  for x,want in [(.5,True),(0.,False)]:
   out=root/('margin_'+str(x));out.mkdir();producer.O=out;auditor.O=out
   H=sparse.csr_matrix([[1.]]);im=dict(lower=[[0,1]],upper=[[1,1]],widths=[[1,1]])
   p=producer.exact_margin(H,im,np.array([x]),time.monotonic());a=auditor.audit_exact(H,im,np.array([x,.25]),time.monotonic())
   assert p==a and p['passed']==want
   checks.append('strict_margin_passes' if want else 'zero_exact_margin_rejected')
 return dict(passed=True,checks=checks,count=len(checks),study_data_loaded=False,solver_calls=0)
if __name__=='__main__':print(json.dumps(run(),indent=2))

"""Independent NumPy decoder and scalar comparisons; no producer import."""
from common import *
import sys
def label(family,i):
 if family!='inequality':return dict(kind='eta' if i==78790 else 'original_direction_coordinate',index=i)
 ends=[331495,399745,467995,560262,652529,744796,837063]
 names=['kept_auxiliary','positive_consistency','negative_consistency','closed_upper_interval','closed_lower_interval','reserved_upper_interval','reserved_lower_interval'];start=0
 for end,name in zip(ends,names):
  if i<end:return dict(kind=name,index=i-start)
  start=end
 raise AssertionError(i)
def main(origin):
 c=admit('audit',origin)
 for p,h in c['numpy_runtime_sha256'].items():assert sha(p)==h,p;check(origin)
 import numpy as np
 with np.load(OLD/'backend_marginals.npz',allow_pickle=False) as z:arrays={n:z[n] for n in ('inequality','lower','upper')}
 with np.load(OLD/'arrays.npz',allow_pickle=False) as z:bounds=z['bounds']
 assert bounds.shape==(78791,2) and bounds.dtype==np.float64
 counts={};worst={};records=0
 with (O/'violations.jsonl').open() as stream:
  for family,n in (('inequality',837063),('lower',78791),('upper',78791)):
   a=arrays[family];assert a.shape==(n,) and a.dtype==np.float64 and np.isfinite(a).all()
   for i,entry in enumerate(a):
    if i%4096==0:check(origin)
    x=float(entry);bad=[]
    if family=='inequality' and x>0:bad.append('positive_inequality_marginal')
    if family=='upper' and x>0:bad.append('positive_upper_marginal')
    if family=='lower' and x<0:bad.append('negative_lower_marginal')
    if family=='lower' and bounds[i,0]==float('-inf') and x!=0:bad.append('nonzero_lower_at_negative_infinity')
    if family=='upper' and bounds[i,1]==float('inf') and x!=0:bad.append('nonzero_upper_at_positive_infinity')
    if not bad:continue
    row=json.loads(next(stream));assert row==dict(family=family,index=i,raw_hex=x.hex(),raw_value=x,conditions=bad,association=label(family,i));records+=1
    for condition in bad:
     counts[condition]=counts.get(condition,0)+1
     if condition not in worst or abs(x)>abs(worst[condition]['raw_value']):worst[condition]=dict(family=family,index=i,raw_hex=x.hex(),raw_value=x)
  assert stream.read()==''
 summary=read(O/'summary.json');assert summary['records']==records and summary['counts']==counts
 for condition,item in worst.items():assert item=={k:v for k,v in summary['worst'][condition].items() if k!='magnitude_bits'}
 assert sha(O/'violations.jsonl')==summary['violation_trace_sha256']
 write(O/'audit.json',dict(passed=True,records=records,counts=counts,raw_values_and_associations_verified=True,no_candidate_admission=True,solver_calls=0,KKT_replayed=False,physical_success=False));check(origin)
if __name__=='__main__':main(float(sys.argv[1]))

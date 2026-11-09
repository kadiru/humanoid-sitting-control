"""ONE exact local pressure-quotient necessary test; no solver or point."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from low_seat_pressure_quotient_602 import *
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import datetime
def pair(x):return [x.numerator,x.denominator]
def main():
 out=D/'pressure_quotient_preparation_602';assert not out.exists()
 assert read(D/'pressure_quotient_freeze_exit_602.json')['actual_exit']==0
 out.mkdir();progress={}
 try:
  for p,h in read(D/'pressure_quotient_launch_602.json')['source_input_sha256'].items():assert sha(p,START)==h,p
  layout=read(D/'gap_coordinate_preparation_v3_466/layout.json');B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsr();mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
  assert np.flatnonzero(~mesh['fixed'])[0]==1 and not np.any(mesh['inactive'][1,:4])
  for col,corner in zip([2,3,4],[0,1,3]):
   assert layout['columns'][col]==dict(kind='load',state=1,index=corner,pivot=2)
   v=B[:,col].tocoo();assert {int(i):float(x) for i,x in zip(v.row,v.data)}=={12+corner:1.,14:-1.}
  E=sparse.load_npz(D/'zero_face_preparation_572/equality.npz').tocsr();ids=np.load(D/'zero_face_preparation_572/equality_original_positive_ids.npy')
  assert E.shape==(69617,78791) and E.has_canonical_format and np.array_equal(ids[:50],331495+np.arange(50))
  def row(i):
   z=np.asarray(E[i,[2,3,4]].toarray()).ravel();assert np.isfinite(z).all();return list(map(lambda x:F(float(x)),z)),[float(x).hex() for x in z]
  h0,x0=row(1378);h1,x1=row(1379)
  n=[h0[1]*h1[2]-h0[2]*h1[1],h0[2]*h1[0]-h0[0]*h1[2],h0[0]*h1[1]-h0[1]*h1[0]]
  prefix=[];failure=None;count=0
  if any(n):
   with (out/'row_prefix.jsonl').open('x') as f:
    for offset in range(50):
     if offset in [12,13]:continue
     check(START);i=1366+offset;r,hx=row(i);defect=sum((a*b for a,b in zip(r,n)),F(0));count+=3
     q=dict(row=i,offset=offset,original_positive_id=int(ids[offset]),coefficients=list(map(pair,r)),binary64_hex=hx,defect=pair(defect));f.write(json.dumps(q,separators=(',',':'))+'\n');prefix.append(i);progress=dict(rows_checked=len(prefix),coefficient_products=count)
     if defect:
      failure=q;break
  else:(out/'row_prefix.jsonl').write_text('')
  result='exact_local_pressure_quotient_rejected' if failure else ('necessary_local_premise_only_passed' if any(n) else 'fixed_moment_minor_inconclusive')
  cert=dict(state=1,block=0,foot='left',original_columns=[2,3,4],corners=[0,1,3],conserved_pivot_corner=2,moment_rows=[1378,1379],moment_coefficients=[list(map(pair,h0)),list(map(pair,h1))],moment_binary64_hex=[x0,x1],signed_cofactors=list(map(pair,n)),checked_rows=prefix,first_nonzero=failure)
  (out/'certificate.json').write_text(json.dumps(cert,indent=2));check(START)
  r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,preparation_only=True,result=result,proposed_quotient_certified=False,exact_obstruction_found=failure is not None,local_rank=3 if failure else (2 if any(n) else None),cofactor_products=6,row_products=count,rows_checked=len(prefix),exact_inverse_calls=0,linear_system_solves=0,lp_calls=0,primal_points=0,model_calls=0,physical_admission=False,scientific_elapsed_s=time.monotonic()-START,peak_memory_measured=False,specification=SPEC)
  (out/'preparation.json').write_text(json.dumps(r,indent=2));check(START)
  (out/'SHA256.json').write_text(json.dumps({p.name:sha(p,START) for p in out.iterdir() if p.is_file()},indent=2));check(START)
  tm=dict(through_output_manifest_elapsed_s=time.monotonic()-START,limit_s=240,excludes='final tiny timing record/stdout/shutdown; not fullyinstrumented processwall')
  (out/'completion_timing.json').write_text(json.dumps(tm,indent=2));check(START)
  print(json.dumps({k:v for k,v in r.items() if k!='specification'},indent=2),flush=True)
 except BaseException as e:
  (out/'terminal_failure.json').write_text(json.dumps(dict(exception=type(e).__name__,message=str(e),progress=progress,elapsed_s=time.monotonic()-START,terminal=True,restart_allowed=False),indent=2));raise
if __name__=='__main__':main()

"""Independent integer determinant certificate audit; no producer imports."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,math,itertools,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(Path(p).read_text())
def check():
 if time.monotonic()-START>240:raise TimeoutError('240s independent pressure audit cap')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b);check()
 return h.hexdigest()
def determinant(A):
 # Integer row denominators, Leibniz permutations, exact integer result.
 n=len(A);den=[math.lcm(*(a.denominator for a in r)) for r in A];Z=[[a.numerator*(den[i]//a.denominator) for a in r] for i,r in enumerate(A)];total=0;products=0
 for p in itertools.permutations(range(n)):
  sign=(-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n));v=Z[0][p[0]]
  for i in range(1,n):v*=Z[i][p[i]];products+=1
  total+=sign*v
 return F(total,math.prod(den)),products
def main():
 src=D/'pressure_quotient_preparation_602';out=D/'pressure_quotient_audit_603';assert not out.exists()
 ex=read(D/'pressure_quotient_preparation_exit_602.json');assert ex['actual_exit']==0 and ex['main_manifest_sha256']==sha(src/'SHA256.json') and ex['completion_timing_sha256']==sha(src/'completion_timing.json')
 out.mkdir()
 try:
  for p,h in read(D/'pressure_quotient_launch_602.json')['source_input_sha256'].items():assert sha(p)==h,p
  for p,h in read(src/'SHA256.json').items():assert sha(src/p)==h,p
  c=read(src/'certificate.json');r=read(src/'preparation.json');assert r['passed'] and not r['proposed_quotient_certified']
  layout=read(D/'gap_coordinate_preparation_v3_466/layout.json');mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsc()
  assert np.flatnonzero(~mesh['fixed'])[0]==1 and not np.any(mesh['inactive'][1,:4]);assert c['original_columns']==[2,3,4] and c['moment_rows']==[1378,1379] and c['state']==1 and c['block']==0 and c['foot']=='left' and c['corners']==[0,1,3] and c['conserved_pivot_corner']==2
  for col,j in zip([2,3,4],[0,1,3]):
   assert layout['columns'][col]==dict(kind='load',state=1,index=j,pivot=2);lo,hi=B.indptr[col:col+2];assert dict(zip(map(int,B.indices[lo:hi]),map(float,B.data[lo:hi])))=={12+j:1.,14:-1.}
  E=sparse.load_npz(D/'zero_face_preparation_572/equality.npz').tocsr();ids=np.load(D/'zero_face_preparation_572/equality_original_positive_ids.npy');assert E.shape==(69617,78791) and E.has_canonical_format and np.array_equal(ids[:50],331495+np.arange(50))
  def row(i):return [F(float(E[i,j])) for j in [2,3,4]]
  H=[row(1378),row(1379)];assert H==[[F(*v) for v in z] for z in c['moment_coefficients']]
  assert [[float(v).hex() for v in z] for z in H]==c['moment_binary64_hex']
  n=[];minor_products=0
  for j in range(3):
   d,count=determinant([[v for k,v in enumerate(z) if k!=j] for z in H]);n.append((-1)**j*d);minor_products+=count
  assert n==[F(*v) for v in c['signed_cofactors']]
  assert all(sum((a*b for a,b in zip(z,n)),F(0))==0 for z in H)
  entries=[json.loads(line) for line in (src/'row_prefix.jsonl').read_text().splitlines()];expected=[1366+i for i in range(50) if i not in [12,13]]
  assert [z['row'] for z in entries]==expected[:len(entries)]==c['checked_rows'];flags=[];rank_defect=None;det_products=0
  for k,z in enumerate(entries):
   check();i=z['row'];v=row(i);assert v==[F(*x) for x in z['coefficients']] and [float(x).hex() for x in v]==z['binary64_hex'];assert z['offset']==i-1366 and z['original_positive_id']==int(ids[i-1366])
   defect,count=determinant(H+[v]);det_products+=count;assert defect==F(*z['defect'])
   if defect:assert k==len(entries)-1 and z==c['first_nonzero'];rank_defect=defect
   flags.append(1)
  if rank_defect is not None:assert r['result']=='exact_local_pressure_quotient_rejected' and r['local_rank']==3 and r['exact_obstruction_found']
  elif any(n):assert len(entries)==48 and r['result']=='necessary_local_premise_only_passed' and not r['exact_obstruction_found']
  else:assert not entries and r['result']=='fixed_moment_minor_inconclusive' and not r['exact_obstruction_found']
  assert r['cofactor_products']==6 and r['row_products']==3*len(entries) and r['rows_checked']==len(entries)
  np.save(out/'row_prefix_verified.npy',np.array(flags,dtype=np.uint8));np.save(out/'moment_kernel_verified.npy',np.ones(2,dtype=np.uint8))
  q=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,result=r['result'],exact_local_rank=r['local_rank'],exact_obstruction_found=rank_defect is not None,integer_minor_products=minor_products,integer_determinant_products=det_products,moment_kernel_products=6,rows_verified=len(entries),copy_error_exact=0,complete_reduction=False,primal_witness=False,physical_admission=False,old_proofs_replayed=False,producer_imported=False,scientific_elapsed_s=time.monotonic()-START,peak_memory_measured=False)
  (out/'independent_audit.json').write_text(json.dumps(q,indent=2));check()
  (out/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in out.iterdir() if p.is_file()},indent=2));check()
  (out/'completion_timing.json').write_text(json.dumps(dict(through_output_manifest_elapsed_s=time.monotonic()-START,limit_s=240,excludes='final tiny timing record/stdout/shutdown; not fullyinstrumented processwall'),indent=2));check();print(json.dumps(q,indent=2),flush=True)
 except BaseException as e:
  (out/'terminal_failure.json').write_text(json.dumps(dict(exception=type(e).__name__,message=str(e),elapsed_s=time.monotonic()-START,terminal=True,restart_allowed=False),indent=2));raise
if __name__=='__main__':main()

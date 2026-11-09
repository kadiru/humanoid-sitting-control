"""Independent binary64 rounding-cell and exact error-support audit; no producer import."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime,struct,math,itertools
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def limit():
 if time.monotonic()-START>240:raise TimeoutError('240s independent graph-encoding audit cap')
def frac(v):
 assert len(v)==2 and type(v[0]) is int and type(v[1]) is int and v[1]>0
 q=F(*v);assert q.numerator==v[0] and q.denominator==v[1];return q
def bits(v):return struct.unpack('>Q',struct.pack('>d',v))[0]
def adjacent(v,up):
 assert math.isfinite(v)
 if v==0:raw=1 if up else (1<<63)|1
 else:raw=bits(v)+(1 if ((v>0)==up) else -1)
 w=struct.unpack('>d',struct.pack('>Q',raw))[0];assert math.isfinite(w)
 return w
def nearest(n,d,v):
 assert math.isfinite(v)
 vn,vd=v.as_integer_ratio();delta=n*vd-vn*d
 if not delta:return delta,vd,False
 q=F(n,d);center=F(v);left=(F(adjacent(v,False))+center)/2;right=(center+F(adjacent(v,True)))/2
 assert left<=q<=right
 if q==left or q==right:assert bits(v)&1==0
 return delta,vd,True
def same(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 if a.dtype.kind=='f':assert np.array_equal(a.view(np.uint64),b.view(np.uint64))
 else:assert np.array_equal(a,b)
def same_csr(a,b):
 assert a.shape==b.shape;same(a.indptr,b.indptr);same(a.indices,b.indices);same(a.data,b.data)
def main():
 assert read(D/'graph_encoding_preparation_exit_580.json')['actual_exit']==0
 for p,h in read(D/'graph_encoding_launch_580.json')['source_input_sha256'].items():assert digest(p)==h,p
 outsrc=D/'graph_encoding_preparation_580';src=D/'affine_output_preparation_578'
 for p,h in read(outsrc/'SHA256.json').items():assert digest(outsrc/p)==h,p
 assert read(D/'affine_output_audit_579/independent_audit.json')['passed'] and read(D/'box_support_audit_547/independent_audit.json')['passed']
 A=sparse.load_npz(outsrc/'matrix.npz').tocsr();E=sparse.load_npz(outsrc/'retained_equality.npz').tocsr();a=dict(np.load(outsrc/'arrays.npz'));original=dict(np.load(src/'arrays.npz'));maps=dict(np.load(outsrc/'maps.npz'));source_maps=dict(np.load(src/'maps.npz'))
 assert A.shape==(265967,31016) and A.nnz==2580552 and A.has_canonical_format and set(a)=={'rhs','cost','bounds','equality_rhs'}
 for k in source_maps:same(maps[k],source_maps[k])
 same_csr(E,sparse.load_npz(src/'retained_equality.npz').tocsr());assert E.shape==(21842,31016) and E.nnz==197194
 same(a['cost'],original['cost']);same(a['equality_rhs'],original['equality_rhs']);same(np.load(outsrc/'original_bounds.npy'),original['bounds']);same(np.load(outsrc/'original_row_ids.npy'),np.load(src/'original_row_ids.npy'))
 endpoints=read(D/'box_support_preparation_546/primary/endpoints.json');exponents=np.load(outsrc/'box_exponents.npy');assert exponents.shape==(31015,)
 bounds=a['bounds'];assert bounds.shape==(31016,2) and np.isfinite(bounds).all();columns=maps['retained_columns'];domain=[];boxflags=np.zeros(31016,dtype=np.uint8)
 for j,col in enumerate(columns):
  if j%128==0:limit()
  lower,upper=map(F,endpoints[int(col)]);lo,hi=map(lambda v:F(float(v)),bounds[j]);assert lo<=lower<=upper<=hi
  if j<31015:
   assert lo==-hi and hi>=0
   radius=max(abs(lower),abs(upper))
   if not radius:assert hi==0 and int(exponents[j])==-2147483648
   else:
    n,d=hi.numerator,hi.denominator;assert n>0 and n&(n-1)==0 and d&(d-1)==0
    assert hi/2<radius<=hi and int(exponents[j])==n.bit_length()-d.bit_length()
   assert np.isneginf(original['bounds'][j,0]) and np.isposinf(original['bounds'][j,1])
  else:same(bounds[j],original['bounds'][j]);assert col==78790 and lo==0
  domain.append((lo,hi));boxflags[j]=1
 rowflags=np.zeros(265967,dtype=np.uint8);coeffflags=np.zeros(A.nnz,dtype=np.uint8);floorflags=np.zeros(265967,dtype=np.uint8)
 rounded=0;zeros=0;products=0;rhs_down=0;nonzero_M=0;maxM=F(0);minM=F(0);maxdown=F(0);maxbits=0;slot=0
 with (src/'rational_inequalities.jsonl').open() as f,(outsrc/'rounding_certificates.jsonl').open() as log:
  for i,(line,certline) in enumerate(itertools.zip_longest(f,log)):
   if i%128==0:limit()
   assert line is not None and certline is not None and i<265967
   row,terms,b,_=json.loads(line);ci,ms,ts,ds=json.loads(certline);assert row==ci==i
   start,end=map(int,A.indptr[i:i+2]);assert start==slot and end-start==len(terms)
   M=F(0)
   for k,(j,n,d) in enumerate(terms,start):
    assert int(A.indices[k])==j
    # Integer cross-difference is independent of producer Fraction subtraction.
    v=float(A.data[k]);delta,vd,inexact=nearest(n,d,v);rounded+=int(inexact);zeros+=int(v==0)
    endpoint=domain[j][1 if delta>0 else 0]
    M+=F(delta*endpoint.numerator,d*vd*endpoint.denominator);products+=1;coeffflags[k]=1
   slot=end;threshold=frac(b)-M;v=float(a['rhs'][i]);assert math.isfinite(v)
   down=threshold-F(v);assert down>=0 and threshold<F(adjacent(v,True))
   assert frac(ms)==M and frac(ts)==threshold and frac(ds)==down
   rhs_down+=int(down>0);nonzero_M+=int(M!=0);maxM=max(maxM,M);minM=min(minM,M);maxdown=max(maxdown,down)
   for x in [M,threshold,down]:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
   rowflags[i]=1;floorflags[i]=1
 assert i+1==265967 and slot==A.nnz and np.all(rowflags) and np.all(coeffflags) and np.all(floorflags) and np.all(boxflags);limit()
 q=dict(rows=265967,columns=31016,stored_coefficient_slots=int(A.nnz),rounded_zero_coefficients=zeros,nearest_coefficient_encodings=int(A.nnz),inexact_coefficient_encodings=rounded,exact_error_support_products=products,nonzero_error_products=rounded,compensated_RHS_floor_encodings=265967,strict_RHS_floor_roundings=rhs_down,rows_with_nonzero_error_support=nonzero_M,minimum_exact_error_support=str(minM),maximum_exact_error_support=str(maxM),maximum_RHS_downward_rounding=str(maxdown),largest_certificate_fraction_bits=maxbits,native_outer_box_encodings=31015,scalar_bounds_unchanged=True,original_scalar_column=78790,reduced_scalar_column=31015,equality_rows=21842,equality_nonzeros=197194,original_inequalities_removed=0,original_equalities_copied=True,real_encoded_set_inner_subset=True,real_encoded_set_equivalence_claimed=False,old_endpoint_proofs_repeated=0,graph_point_lifts=0)
 assert q==read(outsrc/'preparation.json')['primary']
 out=D/'graph_encoding_audit_581';out.mkdir(exist_ok=False)
 for name,arr in [('outer_box_verified',boxflags),('nearest_coefficients_verified',coeffflags),('row_error_support_verified',rowflags),('RHS_floor_verified',floorflags)]:np.save(out/(name+'.npy'),arr)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,preparation_only=True,primary=q,independent_nearest_coefficient_certificates=int(A.nnz),independent_integer_error_support_products=products,independent_RHS_floor_certificates=265967,independent_outer_box_enclosures=31015,exact_copy_error=0,elapsed_s=time.monotonic()-START,producer_import=False,numerical_optimizer_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,primal_vectors_constructed_or_evaluated=0,analytic_dual_vectors_constructed=0,graph_point_lifts=0,physical_admission=False,execution_recovery=False,old_proofs_replayed=False,scope='Encoded REAL feasible set is sufficient INNER SUBSET of578 exact rational zero-face projection. Explicit outer coordinate box plus inward coefficient-error/RHS compensation. No wholeLP/finite/tolerance-equivalence,primalwitness,universalfinitegraphlift,performance orphysicalclaim. No complete numerical algorithm/wholepath wrapper.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

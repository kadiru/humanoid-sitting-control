"""Independent binary64 rounding, directed support/RHS, exactE and projected-domain audit."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime,struct,math,itertools
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
PROGRESS={};FLAGS={}
def limit():
 if time.monotonic()-START>240:raise TimeoutError('240s independent balance-encoding audit cap')
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):limit();h.update(b)
 return h.hexdigest()
def frac(v):
 assert len(v)==2 and type(v[0]) is int and type(v[1]) is int and v[1]>0
 q=F(*v);assert [q.numerator,q.denominator]==v;return q
def bits(v):return struct.unpack('>Q',struct.pack('>d',v))[0]
def adjacent(v,up):
 assert math.isfinite(v)
 if v==0:raw=1 if up else (1<<63)|1
 else:raw=bits(v)+(1 if ((v>0)==up) else -1)
 w=struct.unpack('>d',struct.pack('>Q',raw))[0];assert math.isfinite(w);return w
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

def run(out):
 ex=read(D/'balance_encoding_preparation_exit_596.json');assert ex['actual_exit']==0
 for p,h in read(D/'balance_encoding_launch_596.json')['source_input_sha256'].items():assert digest(p)==h,p
 src=D/'balance_block_preparation_594';enc=D/'balance_encoding_preparation_596'
 for p,h in read(enc/'SHA256.json').items():assert digest(enc/p)==h,p
 assert digest(enc/'completion_timing.json')==ex['completion_timing_sha256'] and digest(enc/'SHA256.json')==ex['main_manifest_sha256']
 timing=read(enc/'completion_timing.json');assert timing['checked_limit_passed'] and timing['through_output_manifest_elapsed_s']<=240
 assert read(D/'balance_block_audit_595/independent_audit.json')['passed'] and read(D/'foot_encoding_audit_589/independent_audit.json')['passed']
 a=dict(np.load(enc/'arrays.npz'));old=dict(np.load(src/'arrays.npz'));maps=dict(np.load(enc/'maps.npz'));original_maps=dict(np.load(src/'maps.npz'))
 assert set(a)=={'rhs','equality_rhs','bounds','cost'} and set(maps)==set(original_maps)
 for k in original_maps:same(maps[k],original_maps[k])
 same(a['cost'],old['cost']);same(np.load(enc/'original_bounds.npy'),old['bounds']);same(np.load(enc/'original_row_ids.npy'),np.load(src/'original_row_ids.npy'))
 existing=dict(np.load(D/'foot_encoding_preparation_588/arrays.npz'));em=dict(np.load(D/'foot_encoding_preparation_588/maps.npz'))
 selected=maps['retained_columns'];oc=maps['original_columns'];assert selected.shape==oc.shape==(10541,) and selected[-1]==14635 and oc[-1]==78790
 same(em['original_columns'][selected],oc);same(a['bounds'],existing['bounds'][selected]);same(a['bounds'][-1],old['bounds'][-1])
 assert np.isfinite(a['bounds']).all() and np.isneginf(old['bounds'][:-1,0]).all() and np.isposinf(old['bounds'][:-1,1]).all()
 domain=[(F(float(lo)),F(float(hi))) for lo,hi in a['bounds']];assert all(lo<=hi for lo,hi in domain)
 A=sparse.load_npz(enc/'inequality_matrix.npz').tocsr();E=sparse.load_npz(enc/'equality_matrix.npz').tocsr()
 m=265967;nnz=6102946;assert A.shape==(m,10541) and A.nnz==nnz and A.has_canonical_format and a['rhs'].shape==(m,)
 flags=np.zeros(m,dtype=np.uint8);coeff=np.zeros(nnz,dtype=np.uint8);FLAGS.update(inequality_rows_verified=flags,inequality_coefficients_verified=coeff)
 counts=dict(rows=m,coefficient_encodings=nnz,inexact_coefficients=0,rounded_zero_coefficients=0,error_endpoint_products=0,rhs_floor_encodings=m,strict_rhs_floors=0,nonzero_error_rows=0,maximum_fraction_bits=0)
 PROGRESS.update(stage='inequalities',completed_coefficients=0,completed_inequalities=0,completed_equalities=0,statistics=counts)
 maxsupport=F(0);maxdown=F(0);slot=0
 with (src/'rational_inequalities.jsonl').open() as f,(enc/'inequality_certificates.jsonl').open() as g:
  for i,(line,record) in enumerate(itertools.zip_longest(f,g)):
   if i%32==0:limit()
   assert line is not None and record is not None and i<m
   row,terms,b=json.loads(line);cert=json.loads(record);assert row==i and cert[:2]==[i,row]
   begin,end=map(int,A.indptr[i:i+2]);assert begin==slot and end-begin==len(terms);support=F(0)
   for k,(j,n,d) in enumerate(terms,begin):
    assert int(A.indices[k])==j
    v=float(A.data[k]);num,vd,inexact=nearest(n,d,v);counts['inexact_coefficients']+=int(inexact);counts['rounded_zero_coefficients']+=int(v==0)
    lo,hi=domain[j];endpoint=hi if num>0 else lo
    support+=F(num*endpoint.numerator,d*vd*endpoint.denominator);counts['error_endpoint_products']+=1;coeff[k]=1;PROGRESS['completed_coefficients']=k+1
   v=float(a['rhs'][i]);assert math.isfinite(v);slot=end
   threshold=frac(b)-support;down=threshold-F(v);assert down>=0 and threshold<F(adjacent(v,True))
   assert len(cert)==5 and [frac(x) for x in cert[2:]]==[support,threshold,down]
   counts['strict_rhs_floors']+=int(down!=0);counts['nonzero_error_rows']+=int(support!=0);maxsupport=max(maxsupport,abs(support));maxdown=max(maxdown,down)
   for x in [support,threshold,down]:counts['maximum_fraction_bits']=max(counts['maximum_fraction_bits'],abs(x.numerator).bit_length(),x.denominator.bit_length())
   flags[i]=1;PROGRESS['completed_inequalities']=i+1
 assert i+1==m and slot==nnz and flags.all() and coeff.all();limit()
 counts.update(maximum_exact_error_support=str(maxsupport),maximum_directed_rhs_slack=str(maxdown))
 PROGRESS['stage']='exact_equalities';ef=np.zeros(1367,dtype=np.uint8);ecf=np.zeros(1,dtype=np.uint8);FLAGS.update(exact_equality_rows_verified=ef,equality_coefficients_verified=ecf)
 assert E.shape==(1367,10541) and E.nnz==1 and E.has_canonical_format and a['equality_rhs'].shape==(1367,)
 with (src/'rational_equalities.jsonl').open() as f,(enc/'equality_certificates.jsonl').open() as g:
  for i,(line,record) in enumerate(itertools.zip_longest(f,g)):
   if i%32==0:limit()
   assert line is not None and record is not None and i<1367
   row,terms,b=json.loads(line);cert=json.loads(record);assert row==int(maps['source578_equality_rows'][i]) and cert==[i,row,len(terms),True]
   assert terms==([] if i<1366 else [[10540,1,1]]) and b==[0,1]
   start,end=map(int,E.indptr[i:i+2]);assert end-start==len(terms)
   for k,(j,n,d) in enumerate(terms,start):
    assert int(E.indices[k])==j;vn,vd=float(E.data[k]).as_integer_ratio();assert n*vd==vn*d;ecf[k]=1
   bn,bd=b;vn,vd=float(a['equality_rhs'][i]).as_integer_ratio();assert bn*vd==vn*bd;ef[i]=1;PROGRESS['completed_equalities']=i+1
 assert i+1==1367 and ef.all() and ecf.all();limit()
 q=dict(columns=10541,original_scalar_column=78790,reduced_scalar_column=10540,projected_existing_box_coordinates=10541,new_endpoint_encodings=0,inequality=counts,equality=dict(rows=1367,coefficient_encodings=1,rhs_encodings=1367,inexact_coefficients=0,inexact_rhs=0,error_endpoint_products=0,coefficient_rhs_encoding_error_exactly_zero=True),original_inequalities_removed=0,original_equalities_removed=0,equalities_exact_binary64=True,complete_system_is_equality_surrogate=False,complete_system_real_inner_subset_claimed=True,graph_point_lifts=0,old_proofs_repeated=False)
 assert q==read(enc/'preparation.json')['primary'];FLAGS['domain_original_columns_verified']=oc
 for name,arr in FLAGS.items():np.save(out/(name+'.npy'),arr);limit()
 PROGRESS['stage']='complete'
 return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,preparation_only=True,primary=q,exact_copy_error=0,scientific_elapsed_s=time.monotonic()-START,producer_completion_timing=timing,producer_import=False,numerical_optimizer_calls=0,numerical_linear_system_solves=0,exact_block_inverses=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,primal_vectors_constructed_or_evaluated=0,analytic_dual_vectors_constructed=0,graph_point_lifts=0,physical_admission=False,execution_recovery=False,old_proofs_replayed=False,peak_memory_measured=False,scope='ALLnew596 nearestcells/integererrorproducts/directedRHSfloors/exactE/domainassociation certified. Complete596 numericREALfeasibleset insideexplicitbox is a sufficientINNERsubset of594 exactzeroFACE,possiblyempty/proper. No tolerance/finiteequivalence/primalexistence/performance/physicalclaim;fullpoint/compositeonce-roundedoriginal checks mandatory.')
def main():
 out=D/'balance_encoding_audit_597';out.mkdir(exist_ok=False)
 try:
  r=run(out);(out/'independent_audit.json').write_text(json.dumps(r,indent=2));limit()
  (out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));limit()
  timing=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),through_output_manifest_elapsed_s=time.monotonic()-START,limit_s=240,checked_limit_passed=True,excludes='Final tiny completion_timing.json write/stdout/processshutdown; not fullyinstrumentedprocesswall.')
  (out/'completion_timing.json').write_text(json.dumps(timing,indent=2));limit();print(json.dumps(dict(result=r,completion=timing),indent=2),flush=True)
 except BaseException as e:
  for name,arr in FLAGS.items():np.save(out/(name+'.npy'),arr)
  failure=dict(passed=False,exception=type(e).__name__,message=str(e),elapsed_s=time.monotonic()-START,progress=PROGRESS,partial_prefix_is_not_certificate=True,terminal=True,restart_allowed=False,physical_admission=False)
  (out/'terminal_failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure,indent=2),flush=True);raise
if __name__=='__main__':main()

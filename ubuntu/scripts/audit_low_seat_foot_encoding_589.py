"""Independent rounding cells and signed AFFINE equality-defect certificates."""
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
 if time.monotonic()-START>240:raise TimeoutError('240s independent foot-encoding audit cap')
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
def audit_rows(src,enc,mat,rhs,domain,kind):
 m,nnz=(265967,1891613) if kind=='inequality' else (5462,52506)
 assert mat.shape==(m,14636) and mat.nnz==nnz and mat.has_canonical_format and rhs.shape==(m,)
 flags=np.zeros(m,dtype=np.uint8);coeff=np.zeros(nnz,dtype=np.uint8)
 q=dict(rows=m,coefficient_encodings=nnz,inexact_coefficients=0,rounded_zero_coefficients=0,error_endpoint_products=0,rhs_encodings=m,inexact_rhs=0,nonzero_error_rows=0,maximum_fraction_bits=0)
 maxrad=F(0);maxdown=F(0);excess=0;first=None;slot=0
 source_name='rational_inequalities.jsonl' if kind=='inequality' else 'rational_equalities.jsonl'
 with (src/source_name).open() as f,(enc/(kind+'_certificates.jsonl')).open() as g:
  for i,(line,record) in enumerate(itertools.zip_longest(f,g)):
   if i%64==0:limit()
   assert line is not None and record is not None and i<m
   row,terms,b=json.loads(line);cert=json.loads(record);assert cert[:2]==[i,row]
   if kind=='inequality':assert row==i
   begin,end=map(int,mat.indptr[i:i+2]);assert begin==slot and end-begin==len(terms)
   lower=F(0);upper=F(0)
   for k,(j,n,d) in enumerate(terms,begin):
    assert int(mat.indices[k])==j
    v=float(mat.data[k]);num,vd,inexact=nearest(n,d,v);q['inexact_coefficients']+=int(inexact);q['rounded_zero_coefficients']+=int(v==0)
    lo,hi=domain[j]
    # Independent integer cross-difference and explicit endpoint products.
    if kind=='inequality':
     endpoint=hi if num>0 else lo;upper+=F(num*endpoint.numerator,d*vd*endpoint.denominator);q['error_endpoint_products']+=1
    else:
     e1=F(num*lo.numerator,d*vd*lo.denominator);e2=F(num*hi.numerator,d*vd*hi.denominator)
     lower+=min(e1,e2);upper+=max(e1,e2);q['error_endpoint_products']+=2
    coeff[k]=1
   v=float(rhs[i]);assert math.isfinite(v);slot=end
   if kind=='inequality':
    threshold=frac(b)-upper;down=threshold-F(v);assert down>=0 and threshold<F(adjacent(v,True))
    assert len(cert)==5 and [frac(x) for x in cert[2:]]==[upper,threshold,down]
    q['inexact_rhs']+=int(down!=0);q['nonzero_error_rows']+=int(upper!=0);maxrad=max(maxrad,abs(upper));maxdown=max(maxdown,down);vals=[upper,threshold,down]
   else:
    bn,bd=b;num,vd,inexact=nearest(bn,bd,v);delta=F(num,bd*vd);low=lower-delta;high=upper-delta;assert low<=high
    radius=max(abs(low),abs(high));assert len(cert)==6 and [frac(x) for x in cert[2:]]==[low,high,delta,radius]
    q['inexact_rhs']+=int(inexact);q['nonzero_error_rows']+=int(radius!=0);maxrad=max(maxrad,radius);vals=[low,high,delta,radius]
    if radius>F(1e-9):
     excess+=1
     if first is None:first=i
   for x in vals:q['maximum_fraction_bits']=max(q['maximum_fraction_bits'],abs(x.numerator).bit_length(),x.denominator.bit_length())
   flags[i]=1
 assert i+1==m and slot==nnz and np.all(flags) and np.all(coeff);limit()
 q.update(maximum_exact_error_radius=str(maxrad),maximum_directed_rhs_slack=str(maxdown),equality_defect_radius_above_1e_9=excess,first_equality_radius_above_1e_9=first)
 return q,flags,coeff
def main():
 assert read(D/'foot_encoding_preparation_exit_588.json')['actual_exit']==0
 for p,h in read(D/'foot_encoding_launch_588.json')['source_input_sha256'].items():assert digest(p)==h,p
 src=D/'foot_block_preparation_586';enc=D/'foot_encoding_preparation_588'
 for p,h in read(enc/'SHA256.json').items():assert digest(enc/p)==h,p
 assert read(D/'foot_block_audit_587/independent_audit.json')['passed'] and read(D/'graph_encoding_audit_581/independent_audit.json')['passed']
 a=dict(np.load(enc/'arrays.npz'));old=dict(np.load(src/'arrays.npz'));maps=dict(np.load(enc/'maps.npz'));original_maps=dict(np.load(src/'maps.npz'))
 assert set(a)=={'rhs','equality_rhs','bounds','cost'}
 for k in original_maps:same(maps[k],original_maps[k])
 same(a['cost'],old['cost']);same(np.load(enc/'original_bounds.npy'),old['bounds']);same(np.load(enc/'original_row_ids.npy'),np.load(src/'original_row_ids.npy'))
 b580=dict(np.load(D/'graph_encoding_preparation_580/arrays.npz'));m580=dict(np.load(D/'graph_encoding_preparation_580/maps.npz'))
 selected=maps['retained_columns'];oc=maps['original_columns'];assert selected.shape==oc.shape==(14636,) and selected[-1]==31015 and oc[-1]==78790
 same(m580['retained_columns'][selected],oc);same(a['bounds'],b580['bounds'][selected]);same(a['bounds'][-1],old['bounds'][-1])
 assert np.isfinite(a['bounds']).all() and np.isneginf(old['bounds'][:-1,0]).all() and np.isposinf(old['bounds'][:-1,1]).all()
 domain=[(F(float(lo)),F(float(hi))) for lo,hi in a['bounds']];assert all(lo<=hi for lo,hi in domain)
 A=sparse.load_npz(enc/'inequality_matrix.npz').tocsr();E=sparse.load_npz(enc/'equality_matrix.npz').tocsr()
 ac,af,acf=audit_rows(src,enc,A,a['rhs'],domain,'inequality');ec,ef,ecf=audit_rows(src,enc,E,a['equality_rhs'],domain,'equality')
 q=dict(columns=14636,original_scalar_column=78790,reduced_scalar_column=14635,projected_existing_box_coordinates=14636,new_endpoint_encodings=0,inequality=ac,equality=ec,original_inequalities_removed=0,original_equalities_removed=0,equalities_copied_binary64=False,complete_system_is_equality_surrogate=True,complete_system_real_inner_subset_claimed=False,inequality_inner_implication_only=True,graph_point_lifts=0,old_proofs_repeated=False)
 assert q==read(enc/'preparation.json')['primary'];limit()
 out=D/'foot_encoding_audit_589';out.mkdir(exist_ok=False)
 for name,arr in [('inequality_rows_verified',af),('inequality_coefficients_verified',acf),('equality_defects_verified',ef),('equality_coefficients_verified',ecf),('domain_original_columns_verified',oc)]:np.save(out/(name+'.npy'),arr)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,preparation_only=True,primary=q,exact_copy_error=0,elapsed_s=time.monotonic()-START,producer_import=False,numerical_optimizer_calls=0,numerical_linear_system_solves=0,exact_block_inverses=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,primal_vectors_constructed_or_evaluated=0,analytic_dual_vectors_constructed=0,graph_point_lifts=0,physical_admission=False,execution_recovery=False,old_proofs_replayed=False,scope='ALLnew588 rounding cells,A directed support/RHS andE signed AFFINE defect intervals certified;complete numeric system is EQUALITYSURROGATE,NOTexactREALinner/equivalent feasible set. Only inequalityblock has sufficientinner implication inside explicitprojectedbox. No primalfeasibility/exactequality/performance/physicalclaim. Futurepoint rationalE/fulloriginal checks mandatory.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

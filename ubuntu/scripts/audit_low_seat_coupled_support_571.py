"""Independent exact conic reconstruction. Imports no producer or optimizer."""
import time,resource
T0=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction
import json,hashlib,datetime
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
 if time.monotonic()-T0>240:raise TimeoutError('240s independent coupled-support audit cap')
def rational_row(matrix,i):
 start,stop=matrix.indptr[i:i+2]
 return dict(zip(map(int,matrix.indices[start:stop]),[Fraction(float(v)) for v in matrix.data[start:stop]]))
def same(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 if a.dtype.kind=='f':assert np.array_equal(a.view(np.uint64),b.view(np.uint64))
 else:assert np.array_equal(a,b)
def csr_same(a,b):
 assert a.shape==b.shape;same(a.indptr,b.indptr);same(a.indices,b.indices);same(a.data,b.data)
def audit(role,out):
 source=D/'box_support_preparation_546'/role;new=D/'coupled_support_preparation_570'/role
 A=sparse.load_npz(source/'matrix.npz').tocsr();z=dict(np.load(source/'arrays.npz'));ids=np.load(source/'original_row_ids.npy');src516=np.load(source/'retained_source_positions.npy');owners=np.load(source/'bound_owners.npy')
 S=sparse.load_npz(D/'exact_row_quotient_preparation_516'/role/'original/matrix.npz').tocsr();sa=dict(np.load(D/'exact_row_quotient_preparation_516'/role/'original/arrays.npz'))
 rows=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');g=rows['groups'][np.flatnonzero(rows['keep'])]
 extra=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];joint={10540+i:int(v)%75-39 for i,v in enumerate(extra) if 39<=int(v)%75<=54}
 target_names=set(('old_window_upper','old_window_lower','mesh_window_upper','mesh_window_lower','clock_target_upper','clock_target_lower','old_ramp_upper','old_ramp_lower'))
 rate_names={'mesh_rate_increasing','mesh_rate_decreasing'};pairs={};eligible=[];protected=[]
 for i,rid in enumerate(ids.tolist()):
  limit()
  if rid>=len(g):continue
  name=str(g[rid]);lo,hi=A.indptr[i:i+2];columns=[int(x) for x in A.indices[lo:hi] if x<78790]
  if name in rate_names and len(columns)==2:
   assert columns[0] in joint and columns[1] in joint and joint[columns[0]]==joint[columns[1]]
   p=rational_row(A,i);assert p[columns[0]]+p[columns[1]]==0
   pairs.setdefault(frozenset(columns),[]).append(i);protected.append(i)
  if name in target_names and len(columns)>=2:
   assert all(c in joint for c in columns) and len({joint[c] for c in columns})==1;eligible.append(i)
 chosen={}
 for i in eligible:
  limit();a=rational_row(A,i);columns=sorted(c for c in a if c<78790);possible=[]
  for left in range(len(columns)):
   for right in range(left+1,len(columns)):
    pair=(columns[left],columns[right])
    for k in pairs.get(frozenset(pair),[]):
     p=rational_row(A,k)
     if (a[pair[0]]>0)==(p[pair[0]]>0) or (a[pair[1]]>0)==(p[pair[1]]>0):possible.append(k)
  if possible:chosen[i]=sorted(possible,key=lambda k:int(ids[k]))[0]
 decision=np.zeros(A.shape[0],dtype=np.uint8);decision[eligible]=3;seen=[];evaluations=0;native_terms=0;conic_coefficient_products=0;conic_rhs_products=0;retained516=set(src516.tolist());maxbits=0
 with (new/'certificates.jsonl').open() as f:
  for line in f:
   limit();i,k,tests,best=json.loads(line);assert chosen[i]==k;seen.append(i);a=rational_row(A,i);p=rational_row(A,k)
   knots=set()
   for col,v in p.items():
    x=a.get(col,Fraction(0))/v
    if x>0:knots.add(x)
   expected=sorted(knots);assert [Fraction(test[0]) for test in tests]==expected
   exact_slacks=[]
   for lam,saved_slack in tests:
    limit();lam=Fraction(lam);assert lam>0
    combined={j:lam*v for j,v in p.items()};rhs=lam*Fraction(float(z['rhs'][k]));conic_coefficient_products+=len(p);conic_rhs_products+=1
    for j in sorted(set(a)|set(p)):
     residual=a.get(j,Fraction(0))-combined.get(j,Fraction(0))
     if residual==0:continue
     side=1 if residual>0 else 0;owner=int(owners[j,side]);assert owner!=-2
     if owner==-1:
      base=Fraction(1 if side else -1);constant=Fraction(float(z['bounds'][j,side]))*(1 if side else -1)
     else:
      assert owner in retained516
      rr=rational_row(S,owner);assert len(rr)==1 and j in rr;base=rr[j];constant=Fraction(float(sa['rhs'][owner]))
     multiplier=residual/base;assert multiplier>0
     combined[j]=combined.get(j,Fraction(0))+multiplier*base;rhs+=multiplier*constant
     native_terms+=1;conic_coefficient_products+=1;conic_rhs_products+=1
    assert {j:v for j,v in combined.items() if v}==a
    slack=Fraction(float(z['rhs'][i]))-rhs;assert slack==Fraction(saved_slack);exact_slacks.append(slack);evaluations+=1;maxbits=max(maxbits,slack.numerator.bit_length(),slack.denominator.bit_length())
   independently_best=exact_slacks.index(max(exact_slacks));assert best==independently_best
   decision[i]=2 if exact_slacks[best]>=0 else 1
 assert seen==list(chosen) and len(set(seen))==len(seen)
 assert not np.any(decision[protected]);same(decision,np.load(new/'decisions.npy'))
 keep=np.flatnonzero(decision!=2);same(keep,np.load(new/'retained_546_positions.npy'));same(src516[keep],np.load(new/'source_516_positions.npy'));same(ids[keep],np.load(new/'original_row_ids.npy'))
 csr_same(A[keep].tocsr(),sparse.load_npz(new/'matrix.npz').tocsr());csr_same(sparse.load_npz(source/'equality.npz'),sparse.load_npz(new/'equality.npz'))
 nz=dict(np.load(new/'arrays.npz'));assert set(z)==set(nz)
 for key in z:same(z[key][keep] if key=='rhs' else z[key],nz[key])
 np.save(out/f'{role}_verified_decisions.npy',decision);np.save(out/f'{role}_verified_retained.npy',keep)
 expected_summary=dict(source_shape=[int(x) for x in A.shape],source_nonzeros=int(A.nnz),reduced_shape=[int(len(keep)),int(A.shape[1])],reduced_nonzeros=int(A[keep].nnz),protected_rate_premises=int(len(protected)),eligible_targets=int(len(eligible)),selected_targets=int(len(chosen)),unmatched_untested=int(len(eligible)-len(chosen)),new_positive_ratio_support_evaluations=int(evaluations),new_residual_endpoint_products=int(native_terms),proved_removed_rows=int(np.count_nonzero(decision==2)),insufficient_targets=int(np.count_nonzero(decision==1)),largest_fraction_bits=int(maxbits),zero_lambda_tests=0,new_endpoints=0,old_box_tests_repeated=0,variables_removed=0,cost_domain_equality_changed=False)
 assert expected_summary==read(D/'coupled_support_preparation_570/preparation.json')['systems'][role]
 return dict(**expected_summary,independent_positive_conic_coefficient_products=int(conic_coefficient_products),independent_positive_conic_rhs_products=int(conic_rhs_products),exact_copy_error=0,all_certificates_checked=True,all_premises_retained=True,old_endpoint_tightness_proof_replayed=False)
def main():
 assert read(D/'coupled_support_preparation_exit_570.json')['actual_exit']==0
 for p,h in read(D/'coupled_support_launch_570.json')['source_input_sha256'].items():assert digest(p)==h,p
 for p,h in read(D/'coupled_support_preparation_570/SHA256.json').items():assert digest(D/'coupled_support_preparation_570'/p)==h,p
 limit();out=D/'coupled_support_audit_571';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  systems[role]=audit(role,out);print(json.dumps(dict(role=role,**systems[role])),flush=True)
 limit();r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,physical_admission=False,preparation_only=True,systems=systems,elapsed_s=time.monotonic()-T0,production_import=False,numerical_optimizer_calls=0,linear_system_solves=0,robot_lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,old_scientific_calculations_repeated=False,new_candidate=False,execution_recovery=False)
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(passed=True,elapsed_s=r['elapsed_s'])),flush=True)
if __name__=='__main__':main()

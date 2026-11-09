"""Independent exact pair/projection audit; no production projection import or solver."""
from pathlib import Path
from fractions import Fraction as Q
import json,hashlib,datetime,itertools
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'torque_projection_preparation_522';O=D/'torque_projection_audit_523'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def entries(M,i):
 start,end=M.indptr[i:i+2];cols=M.indices[start:end];assert len(np.unique(cols))==len(cols)
 return {int(j):Q(float(v)) for j,v in zip(cols,M.data[start:end]) if v!=0}
def same(A,B):
 A=A.copy().tocsr();B=B.copy().tocsr();A.sort_indices();B.sort_indices();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def main():
 assert read(D/'torque_projection_preparation_exit_522.json')['actual_exit']==0
 for p,h in read(D/'torque_projection_launch_522.json')['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed']
 O.mkdir(exist_ok=False);results={}
 ex=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];expected_el=np.array([10540+i for i,j in enumerate(ex) if 55<=int(j)%75<=70])
 for role in ['primary','witness']:
  src=D/'exact_row_quotient_preparation_516'/role/'original';A=sparse.load_npz(src/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'));B=sparse.load_npz(P/role/'matrix.npz').tocsr();b=dict(np.load(P/role/'arrays.npz'));led=dict(np.load(P/role/'projection_ledger.npz'));el=led['eliminated_columns'];ret=led['retained_columns'];unt=led['unchanged_rows'];pairs=led['pairs']
  assert np.array_equal(el,expected_el) and len(el)==21840;assert np.array_equal(ret,np.setdiff1d(np.arange(A.shape[1]),el));assert np.all(a['cost'][el]==0) and np.all(a['bounds'][el,0]==-np.inf) and np.all(a['bounds'][el,1]==np.inf)
  incidence=A[:,el].tocsr();assert np.all(np.diff(incidence.indptr)<=1);assert np.array_equal(unt,np.flatnonzero(np.diff(incidence.indptr)==0))
  same(B[:len(unt)],A[unt][:,ret]);assert np.array_equal(b['rhs'][:len(unt)],a['rhs'][unt]);assert np.array_equal(b['cost'],a['cost'][ret]) and np.array_equal(b['bounds'],a['bounds'][ret]) and np.array_equal(b['equality_rhs'],a['equality_rhs'])
  E=sparse.load_npz(D/'lifted_feasibility_preparation_490'/role/'equality.npz');assert E[:,el].nnz==0;same(sparse.load_npz(P/role/'equality.npz'),E[:,ret]);assert np.array_equal(np.load(P/role/'source_row_ids.npy'),np.load(src.parent/'row_ids.npy'))
  C=A.tocsc();rowcache={};cursor=0;dest=len(unt);flags=np.zeros(len(pairs),dtype=np.bool_);coefs=0;statuses=[0,0,0]
  inverse={int(c):i for i,c in enumerate(ret)}
  for c in el:
   start,end=C.indptr[c:c+2];rr=C.indices[start:end];vv=C.data[start:end];lower=rr[vv<0];upper=rr[vv>0];assert len(lower)>0 and len(upper)>0
   rowcache={int(i):entries(A,int(i)) for i in rr}
   for low,high in itertools.product(lower,upper):
    col,li,ui,output,status,power=map(int,pairs[cursor]);assert (col,li,ui)==(int(c),int(low),int(high))
    L=rowcache[li];U=rowcache[ui];al=L[col];au=U[col];assert al<0<au
    # Independent endpoint subtraction, then positive denominator product.
    factor=au*(-al);rest={j:(U.get(j,Q(0))/au-L.get(j,Q(0))/al)*factor for j in set(L)|set(U)};assert rest[col]==0;rest={j:q for j,q in rest.items() if q};rhs=(Q(float(a['rhs'][ui]))/au-Q(float(a['rhs'][li]))/al)*factor
    assert all(j in inverse for j in rest)
    maximum=Q(0);bounded=True
    for j,q in rest.items():
     endpoint=a['bounds'][j,1 if q>0 else 0]
     if not np.isfinite(endpoint):bounded=False;break
     maximum+=q*Q(float(endpoint))
    expect=1 if not rest and rhs>=0 else 2 if bounded and maximum<=rhs else 0;assert status==expect;statuses[status]+=1
    if status:
     assert output==-1 and power==0
    else:
     assert output==dest;scale=Q(2)**power;largest=max([abs(rhs)]+[abs(q) for q in rest.values()])*scale;assert Q(1,2)<largest<=1
     got=entries(B,output);assert set(got)=={inverse[j] for j in rest}
     for j,q in rest.items():
      rounded=float(q*scale);assert np.isfinite(rounded) and rounded!=0;assert got[inverse[j]]==Q(rounded);coefs+=1
     assert b['rhs'][output]==float(rhs*scale) and (rhs==0 or b['rhs'][output]!=0);dest+=1
    flags[cursor]=True;cursor+=1
  np.savez_compressed(O/(role+'_pair_proof.npz'),verified_pairs=flags,source_rows=unt,eliminated_columns=el)
  assert cursor==len(pairs) and dest==B.shape[0] and flags.all();results[role]=dict(pairs_verified=cursor,generated_coefficients_verified=coefs,kept_pairs=statuses[0],constant_tautologies=statuses[1],variable_bound_implications=statuses[2],matrix_rhs_copy_error=0,rounded_exact_pair_error=0,shape=list(B.shape),nnz=B.nnz,source_quotient_proof_inherited_not_repeated=True)
  print(json.dumps(dict(role=role,**results[role])),flush=True)
 cases=read(P/'new_projection_examples.json');c0,c1=cases;assert c0['source']==[[-2,1,1],[3,-2,2]];assert 3*(-2)+2*3==0 and 3*1+2*(-2)==-1 and 3*1+2*2==7
 assert c0['at_y0']==['-1/2','2/3','1/12'] and c0['at_y_minus7_singleton']=='-4' and c0['at_y_minus8_empty']
 assert Q(-9,2)>Q(-14,3);assert c1['source']==[[3,1],[-3,-1]] and Q(c1['exact_singleton'])==Q(1,3) and Q(c1['once_rounded'])!=Q(1,3) and Q(c1['rounding_error'])==Q(c1['once_rounded'])-Q(1,3) and not c1['binary64_exact_lift_exists']
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_only=True,systems=results,new_examples_verified=2,original_variables_removed=21840,remaining_direction_coordinates=56950,exact_real_projection_proved=True,rounded_lp_exact_equivalence_claimed=False,binary64_lift_always_exists=False,prior517_proof_repeated=False,model_calls=0,lp_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

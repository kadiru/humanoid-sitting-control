"""Exact binary64 feasible-set-preserving singleton-row absorption; no candidate rounding or LP."""
from pathlib import Path
from fractions import Fraction as F
import struct,json,hashlib,time,sys
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(scope='One new finite-domain structural reduction from saved546 exact endpoints, not another bound propagation pass or solver retry.',
 theorem='Let B=[L,U] be saved546 exact rational unconditional endpoints. Define l64=min{finite binary64 v:v>=L},u64=max{finite binary64 v:v<=U}. Replace ALL singleton rows by coordinate bounds[l64,u64],retain all other546 rows,cost,E,RHS. For finite binary64 vectors,original516 feasibility iff new feasibility,using inherited517/547 proofs. Over real vectors the new polytope is an inner subset,NOT generally equal to the original real LP. No statement about equality of tolerance-feasible sets,real optima or original dual certificates.',
 derivation='For |q|=n/d>0 find e=floor(log2(n/d)) by integer bit lengths and exact comparison;k=max(-1074,e-52). Divide n/d by2^k using integers; choose floor or ceil quotient,reverse direction for negative q. Encode the integer significand and exponent directly into IEEE binary64 bits; carry at2^53 exactly. Zero canonical+0 but both signed-zero candidate encodings remain mathematically feasible. No nearest-float guess,nextafter,bisection or candidate adjustment.',
 reuse='Load immutable546 endpoints/owners and reduced matrix plus547 proof byhash. NEVER reconstruct endpoints,tighten/propagate bounds,rerun mixed-row supports or517/547 proof. Remove only exact singleton rows;constants/mixed rows unchanged. All78791 coordinates retained. Both consistency signs/frozen units and all original criteria remain imposed through explicit mixed rows and source-implied bounds.',
 audit='Independent553 imports no producer. It uses adjacent IEEE bit encodings to prove lower predecessor<L<=lower64 and upper64<=U<upper successor for EVERY endpoint. Exact positive-conic bound support proves every removed singleton row. It checks original finite bounds,new domain/row IDs/CSR/RHS/cost/E copies and inheritedproof hashes. No LP or synthetic fixture.',
 limits='No robot direction,scalar/torque lift,finite path encoding,model/root/map/physics. No universal float lift: if l64>u64 the interval has no finite binary64 member and preparation rejects; no old1/3fixture repeated. No speed/conditioning/feasibility gain from storage counts. No numerical residual amplification or original KKT guarantee.',
 future='All returned native+scalar solver coordinates must stay bitwise unchanged. FULL new/546/516/FULL490 primal/equality/bound/KKT gates remain. Rounded bound multipliers would need an explicit original-premise dual lift plus full original stationarity/gap; a stronger real bound is NOT automatically an original-real dual certificate. Any changedstate/trust/ACTUALceiling needs fresh valid assembly/proofs;no illustrative witnessceiling reuse. No complete554 numerical algorithm/wrapper/budget exists.',
 budgets=dict(preparation_wall_s=240,independent_audit_wall_s=240,address_space_GiB=40,peak_memory_measured=False,checks_between_operations=True,LP_budget_unchanged_s=45),
 references=['https://docs.python.org/3.12/library/struct.html#format-characters','https://docs.python.org/3.12/library/stdtypes.html#float.as_integer_ratio'],physical_admission=False)
def check(start):
 if time.monotonic()-start>240:raise RuntimeError('240s symbolic preparation deadline; no continuation')
def directed(q,ceil):
 assert isinstance(q,F)
 if not q:return 0.,(0,-1074,0)
 sign=1 if q>0 else -1;n=abs(q.numerator);d=q.denominator;e=n.bit_length()-d.bit_length()
 if (n<d<<e) if e>=0 else (n<<-e<d):e-=1
 k=max(-1074,e-52);num,den=(n,d<<k) if k>=0 else (n<<-k,d)
 m,rem=divmod(num,den);up=ceil if sign>0 else not ceil
 if up and rem:m+=1
 if m==2**53:m//=2;k+=1
 if not m:bits=0
 elif k==-1074 and m<2**52:bits=m
 else:
  ex=k+52+1023;assert 1<=ex<=2046 and 2**52<=m<2**53,'unsupported finite binary64 endpoint range'
  bits=(ex<<52)|(m-2**52)
 if sign<0 and m:bits|=1<<63
 v=struct.unpack('>d',bits.to_bytes(8,'big'))[0];assert np.isfinite(v)
 exact=F(sign*m)*(F(2)**k);assert F(v)==exact and (F(v)>=q if ceil else F(v)<=q)
 return v,(sign,k,m)
def derive(role,out,start):
 src=D/'box_support_preparation_546'/role;A=sparse.load_npz(src/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'));E=sparse.load_npz(src/'equality.npz');ep=read(src/'endpoints.json');owners=np.load(src/'bound_owners.npy');ids=np.load(src/'original_row_ids.npy');oldkeep=np.load(src/'retained_source_positions.npy')
 assert A.shape[1]==78791 and A.has_canonical_format and np.all(A.data!=0) and len(ep)==A.shape[1] and np.all(owners!=-2)
 assert sys.float_info.radix==2 and sys.float_info.mant_dig==53 and sys.float_info.max_exp==1024 and sys.float_info.min_exp==-1021 and np.dtype('float64').itemsize==8
 bounds=np.empty((A.shape[1],2));enc=np.empty((A.shape[1],2,3),dtype=np.int64);changed=0;worst=F(0);bitmax=0
 with (out/'rounding_ledger.jsonl').open('w') as log:
  for j,(ls,us) in enumerate(ep):
   check(start);lo,hi=F(ls),F(us);assert lo<=hi
   bounds[j,0],enc[j,0]=directed(lo,True);bounds[j,1],enc[j,1]=directed(hi,False)
   assert bounds[j,0]<=bounds[j,1],'empty binary64 interval: no preparation output/candidate accepted'
   dl=F(float(bounds[j,0]))-lo;du=hi-F(float(bounds[j,1]));assert dl>=0 and du>=0
   changed+=int(dl>0)+int(du>0);worst=max(worst,dl,du);bitmax=max(bitmax,dl.numerator.bit_length(),dl.denominator.bit_length(),du.numerator.bit_length(),du.denominator.bit_length())
   log.write(json.dumps([j,str(dl),str(du)],separators=(',',':'))+'\n')
 keep=np.flatnonzero(np.diff(A.indptr)!=1);removed=np.flatnonzero(np.diff(A.indptr)==1);B=A[keep].tocsr();check(start)
 sparse.save_npz(out/'matrix.npz',B);np.savez_compressed(out/'arrays.npz',**dict(a,rhs=a['rhs'][keep],bounds=bounds));np.save(out/'original_bounds.npy',a['bounds']);np.save(out/'encoding.npy',enc);np.save(out/'retained546_positions.npy',keep);np.save(out/'removed546_positions.npy',removed);np.save(out/'source516_positions.npy',oldkeep[keep]);np.save(out/'original490_row_ids.npy',ids[keep]);np.save(out/'bound_owners516.npy',owners)
 import shutil
 shutil.copy2(src/'equality.npz',out/'equality.npz')
 return dict(source_shape=list(A.shape),source_nonzeros=int(A.nnz),new_shape=list(B.shape),new_nonzeros=int(B.nnz),singleton_rows_absorbed=len(removed),constants_retained=int(np.count_nonzero(np.diff(A.indptr)==0)),mixed_rows_retained=int(np.count_nonzero(np.diff(A.indptr)>=2)),endpoint_encodings=2*A.shape[1],strict_inward_endpoint_changes=changed,exact_endpoints_unchanged=2*A.shape[1]-changed,maximum_exact_inward_shift=str(worst),largest_shift_fraction_bits=bitmax,native_variables_removed=0,scalar_variables_removed=0,finite_binary64_feasible_set_equal=True,real_polytope_inner_subset=True,real_polytope_equivalence_claimed=False,tolerance_feasible_set_equivalence_claimed=False,source_endpoint_proof_inherited=True,new_endpoint_construction=False,new_mixed_support_tests=0,new_matrix_coefficients=False,new_candidate=False)

"""Certified inward binary64 representation of the saved exact rational graph LP."""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib, time, struct, math, shutil
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(
 scope='Numerical INPUT preparation only for578 exact rational primary zero-face graph reduction. No LP/linear solve/candidate/graph-point lift/model/physical trial. Complete numerical algorithm and whole-path/return auditor remain absent.',
 initialization='Immutable578 rational inequalities,retainedE/cost/originalbounds/maps and579 reverse proof; immutable546 PRIMARY exact unconditional endpoints and547 proof. All earlier proofs inherited by hashes,not recomputed. No552 inward endpoints,old solver vectors or576 report replay.',
 domain='For each of31015 retained native coordinates choose R_j=least nonnegative power of2 >=max(abs(saved546 L_j),abs(saved546 U_j)),with R_j=0 if both zero. Explicit numerical bound [-R_j,R_j]. Scalar31015 retains original [0,U] and t=0 equality. These are OUTWARD enclosures of existing theorem endpoints,not tighter original bounds or new endpoint propagation. All exact578 feasible points lie inside this box by inherited proofs. Numeric bounds MUST be imposed; they cannot be inferred for an approximate point.',
 encoding='For each saved exact rational coefficient a_j,encode ONE nearest-even finite binary64 ahat_j,then compute exact e_j=a_j-Fraction(ahat_j). M=sum max(e_j*l_j,e_j*u_j) over the explicit binary64 box. Choose bhat=greatest finite binary64 <= b-M by direct integer IEEE encoding. No row scaling,normalization,coefficient threshold,row removal or adaptive tightening. Preserve all265967 rows and2580552 source coefficient slots,including any coefficient rounded to zero. Copy retained equality coefficients/RHS,cost,column/row maps bitwise.',
 theorem='For exact REAL u inside the explicit numeric box, a.u=ahat.u+e.u<=ahat.u+M. Hence ahat.u<=bhat and bhat<=b-M imply a.u<=b. Thus the REAL encoded feasible set is a SUFFICIENT INNER SUBSET of exact578 projection,which itself equals572 zero face projection. It need not equal either set and may be empty; no original optimum-zero/existence/tolerance-equivalence claim. No universal binary64 graph extension.',
 error_policy='Save exact M,b-M,and nonnegative downward RHS rounding slack for every row. Audit every new coefficient nearest-even enclosure by adjacent binary64 midpoint inequalities,each error support using integer residual numerators and signed endpoints,and each RHS floor by IEEE successor inequality. Exact compensated row implication holds only for exact satisfaction of numeric constraints and box; numerical residuals and graph rounding require independent full checks.',
 future_return='All31016 retained coordinates BITWISE backend values; scalar already exactly0(+/-0),never set/clip/repair/normalize. Reconstruct each47775 auxiliary output by exact original graph sum/division then roundONCE. Require finite values and full encoded/exact-rational-reduced/572/570/546/516/FULL490 primal/equality/bound/requiredstationarity/absolute-gap <=1e-9 inside1e-7. Original573 analytic lowerbound may be inherited unchanged; no newdualconstruction now. Exact/rational original residuals cannot be inferred from this representation theorem or numerical KKT.',
 wholepath='No complete numerical construction exists here. Before any LP freeze complete initialization/algorithm/resources/return and actual-state independent auditors. Primary NEVER mapped; same complete original direction determines modeledphi/actual(mu+theta)/2 ceiling/prediction,not assumed0fromt0. Interior uses separately frozen freshFULLoriginal490,not zero-facegraph. Newstate/trust needs valid fresh assembly/graph/encoding/input proofs; initial reuse only fullidentityhash.45sTOTAL fromMAINentry BEFOREimports/loads/preparation/backend/allKKT,40GiBchild/120sparent,24major48LP120maps5dyadics/trust/guards unchanged. All physical/runtime/retention/PPO gates remain.',
 resources='ONE producer240s andONE independent audit240s,frombefore scientificimports;40GiBaddressspace,checksbetweenoperations,peakUNMEASURED. Symbolic/numerical input preparation limits,NOT45s robot LP extension. Terminal stage on failure,no completed arithmetic replay.',
 audit='Independent581 imports NOproducer/optimizer. Reconstruct box inclusion/minimal power2 directly from saved endpoint comparisons. Verify nearest-even coefficient cells,integer residual support and directed RHS floor for every row. Check E/cost/maps/IDs/originalbounds bitwise and explicit bounds. Do not rerun578 substitution/579 reverse identities/547 endpoints/support/552 encodings/573 analyticdual proof.',
 physical_admission=False,complete_numerical_algorithm_exists=False,primal_witness_exists=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s graph encoding preparation cap')
def pair(q):return [q.numerator,q.denominator]
def floor64(q):
 # Direct directed encoding of a NEW compensated RHS,not552 endpoint encoding replay.
 if not q:return 0.0
 sign=1 if q>0 else -1;n=abs(q.numerator);d=q.denominator
 e=n.bit_length()-d.bit_length()
 if (n<d<<e) if e>=0 else (n<<-e<d):e-=1
 k=max(-1074,e-52);num,den=(n,d<<k) if k>=0 else (n<<-k,d)
 m,rem=divmod(num,den)
 if sign<0 and rem:m+=1
 if m==2**53:m//=2;k+=1
 if not m:bits=0
 elif k==-1074 and m<2**52:bits=m
 else:
  ex=k+52+1023;assert 1<=ex<=2046 and 2**52<=m<2**53,'compensated RHS outside finite encoding range'
  bits=(ex<<52)|(m-2**52)
 if sign<0 and m:bits|=1<<63
 v=struct.unpack('>d',bits.to_bytes(8,'big'))[0]
 assert math.isfinite(v) and F(v)<=q
 return v
def enclose(lo,hi):
 assert lo<=hi
 r=max(abs(lo),abs(hi))
 if not r:return 0.0,-2147483648
 n,d=r.numerator,r.denominator;e=n.bit_length()-d.bit_length()
 p=F(2)**e
 if p<r:e+=1;p*=2
 elif p/2>=r:e-=1;p/=2
 assert -1074<=e<=1023 and p/2<r<=p
 v=math.ldexp(1.0,e);assert F(v)==p
 return v,e
def build(out,start):
 src=D/'affine_output_preparation_578';maps=dict(np.load(src/'maps.npz'));old=dict(np.load(src/'arrays.npz'))
 columns=maps['retained_columns'];assert len(columns)==31016 and columns[-1]==78790
 endpoints=read(D/'box_support_preparation_546/primary/endpoints.json')
 bounds=old['bounds'].copy();exponents=np.empty(31015,dtype=np.int32)
 assert np.isneginf(bounds[:-1,0]).all() and np.isposinf(bounds[:-1,1]).all() and bounds[-1,0]==0 and np.isfinite(bounds[-1,1])
 for j,col in enumerate(columns[:-1]):
  if j%128==0:check(start)
  lo,hi=map(F,endpoints[int(col)]);rad,e=enclose(lo,hi);bounds[j]=[-rad,rad];exponents[j]=e
 scalar_lo,scalar_hi=map(F,endpoints[int(columns[-1])]);assert F(float(bounds[-1,0]))<=scalar_lo<=scalar_hi<=F(float(bounds[-1,1]))
 exact_bounds=[(F(float(lo)),F(float(hi))) for lo,hi in bounds]
 m=265967;nnz=2580552;data=np.empty(nnz);indices=np.empty(nnz,dtype=np.int32);indptr=np.zeros(m+1,dtype=np.int32);rhs=np.empty(m)
 slot=0;rounded=0;zeros=0;support_products=0;nonzero_errors=0;rhs_down=0;nonzero_M=0;maxM=F(0);minM=F(0);max_down=F(0);maxbits=0
 with (src/'rational_inequalities.jsonl').open() as f,(out/'rounding_certificates.jsonl').open('x') as log:
  for i,line in enumerate(f):
   if i%128==0:check(start)
   row,terms,b,_=json.loads(line);assert row==i and i<m
   M=F(0)
   for col,n,d in terms:
    a=F(n,d);v=float(a);assert math.isfinite(v);err=a-F(v)
    lo,hi=exact_bounds[col];M+=err*(hi if err>0 else lo);support_products+=1
    rounded+=int(err!=0);nonzero_errors+=int(err!=0);zeros+=int(v==0)
    data[slot]=v;indices[slot]=col;slot+=1
   threshold=F(*b)-M;v=floor64(threshold);down=threshold-F(v);assert down>=0
   rhs[i]=v;indptr[i+1]=slot;rhs_down+=int(down>0);nonzero_M+=int(M!=0);maxM=max(maxM,M);minM=min(minM,M);max_down=max(max_down,down)
   for x in [M,threshold,down]:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
   log.write(json.dumps([i,pair(M),pair(threshold),pair(down)],separators=(',',':'))+'\n')
 assert i+1==m and slot==nnz;check(start)
 A=sparse.csr_matrix((data,indices,indptr),shape=(m,31016));assert A.has_canonical_format
 sparse.save_npz(out/'matrix.npz',A);np.savez_compressed(out/'arrays.npz',rhs=rhs,cost=old['cost'],bounds=bounds,equality_rhs=old['equality_rhs'])
 np.save(out/'original_bounds.npy',old['bounds']);np.save(out/'box_exponents.npy',exponents)
 for name in ['maps.npz','original_row_ids.npy','retained_equality.npz']:shutil.copy2(src/name,out/name)
 check(start)
 return dict(rows=m,columns=31016,stored_coefficient_slots=nnz,rounded_zero_coefficients=zeros,nearest_coefficient_encodings=nnz,inexact_coefficient_encodings=rounded,exact_error_support_products=support_products,nonzero_error_products=nonzero_errors,compensated_RHS_floor_encodings=m,strict_RHS_floor_roundings=rhs_down,rows_with_nonzero_error_support=nonzero_M,minimum_exact_error_support=str(minM),maximum_exact_error_support=str(maxM),maximum_RHS_downward_rounding=str(max_down),largest_certificate_fraction_bits=maxbits,native_outer_box_encodings=31015,scalar_bounds_unchanged=True,original_scalar_column=78790,reduced_scalar_column=31015,equality_rows=21842,equality_nonzeros=197194,original_inequalities_removed=0,original_equalities_copied=True,real_encoded_set_inner_subset=True,real_encoded_set_equivalence_claimed=False,old_endpoint_proofs_repeated=0,graph_point_lifts=0)

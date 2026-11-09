"""Exact existential elimination of native auxiliary torque increments, saved arrays only."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(scope='NEW exact Fourier-Motzkin existential projection of21840 native auxiliary TORQUE increments, not physical torque controls or a new basis. Preparation only.',
 source='Immutable516 original-coordinate representative matrices and517 exact REAL-system implication proof, inherited after hashes. Full490 remains mandatory future witness check. No504 transform,old quotient grouping/proof,model/root/AD/LP or finite map is repeated.',
 coordinates='Each1365nonfixed state has16 auxiliary torque outputs. Delete only their21840 LP columns, retaining56950 original direction coordinates plus epigraph/eta. ActualX,commands,COM,loads,gaps,all physical controls remain. Sourceactual472 and490canonicalauxiliary unchanged.',
 structure='Verify EVERY source row has at most ONE eliminated torque coordinate; eliminated columns have zero objective, no equality occurrence and no finite direct variable bound. All their finite limits remain source inequality rows. Each column has nonempty positive and negative incidence. No numerical threshold.',
 derivation='For a negative row aL*x+rL*y<=bL and positive row aU*x+rU*y<=bU, multiply the former by aU>0 and latter by -aL>0. The sum eliminates x exactly: (aU*rL-aL*rU)*y<=aU*bL-aL*bU. Retain EVERY lower/upper pair. Rows with no eliminated variable are copied. Because no source row couples eliminated variables, all such pair inequalities are necessary and sufficient for simultaneous real interval lifts. This is variable elimination, not another cut queue or coordinate rename.',
 proof='For fixed retained y, each source negative row gives a lower endpoint and each positive row an upper endpoint. All pair rows hold iff max(lower)<=min(upper). Choosing any member reconstructs each auxiliary. This preserves BOTH signs and unit scales of torque consistency existentially, not by deleting mechanics criteria. Commands and all other consistency remain explicit. All original criteria are explicit, projected, or inherited exact implications.',
 exact='All source binary64 coefficients/RHS are exact Fractions. Products and sums are exact. Only EXACT zero coefficients cancel. A constant nonnegative-RHS pair is tautological. A pair implied by unchanged retained VARIABLE bounds is proved by its exact maximum over those bounds. Every other pair, including contradictions, is kept; no duplicate/proportional regrouping.',
 storage='Each generated nontrivial pair receives a positive power-of-two scale bringing its largest coefficient/RHS magnitude to(0.5,1]. Scale is exact; each resulting matrix coefficient/RHS rounds ONCE tobinary64. Nonzero-to-zero underflow or nonfinite output rejects. Pair provenance and scale permit independent exact reconstruction. Copied source rows are unchanged. This is exact-real projection with a rounded numerical representation, NOT bitwise equivalence of floating LPs or a changed physical/LP tolerance.',
 reconstruction='A future reduced solver vector keeps every retained coordinate unchanged. Eliminated auxiliary directions would be reconstructed from EXACT original source lower/upper endpoint intersections, with rounding explicitly recorded and exhaustive FULL490 inequality/equality/bound checking at unchanged1e-7. Empty intersections reject. A real singleton need not contain any binary64 number; this limitation is explicit. No clipping or repair of solver-retained values; no claim of a universally exact binary64 lift. Full original intervaleta>0 and immutable490finiteencoding remain separate before any actual mechanics.',
 nonlinear='Auxiliary torque is not realized torque. The eventual full original direction must participate in SAME modeled merit/prediction and finite encoded delta; actual newX needs fresh mechanics/J. Allactualsignednormal/hover/kinematic/geometry/filter/runtime/physical/PPO gates remain. No15-root postprocessing or force-only construction.',
 reference='Primary and witness are SAVED assembly systems only. Witness .09895564659603738 is illustrative, not a solved ceiling. Any changed full LP requires new incidence/projection/proof. No run wrapper,budget,solverperformance or candidate is established.',
 fixtures='Two NEW small existential-projection arithmetic examples: feasible/empty interval distinction and non-binary64 real singleton1/3. Not full-coordinate finite maps,model/derivative/old quotient fixtures or candidate seeds.',
 lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
def row(A,i):
 p,q=A.indptr[i:i+2];out={}
 for c,v in zip(A.indices[p:q],A.data[p:q]):
  c=int(c);assert c not in out
  if v!=0:out[c]=F(float(v))
 return out
def max_at_bounds(r,bounds):
 s=F(0)
 for c,v in r.items():
  end=bounds[c,1 if v>0 else 0]
  if not np.isfinite(end):return None
  s+=v*F(float(end))
 return s
def dyadic_scale(r,b):
 m=max([abs(b)]+[abs(v) for v in r.values()]);assert m>0
 e=m.numerator.bit_length()-m.denominator.bit_length();p=F(2)**e
 if m>p:e+=1
 return -e
def project(A,arr,eliminated,out):
 A=A.tocsr();A.sort_indices();assert np.all(np.isfinite(A.data)) and np.all(np.isfinite(arr['rhs']))
 n=A.shape[1];ret=np.setdiff1d(np.arange(n),eliminated);remap=np.full(n,-1,dtype=np.int64);remap[ret]=np.arange(len(ret))
 assert np.all(arr['cost'][eliminated]==0) and np.all(np.isneginf(arr['bounds'][eliminated,0])) and np.all(np.isposinf(arr['bounds'][eliminated,1]))
 hit=A[:,eliminated].tocsr();assert np.all(np.diff(hit.indptr)<=1)
 unchanged=np.flatnonzero(np.diff(hit.indptr)==0);C=A.tocsc();base=A[unchanged][:,ret].tocsr();data=base.data.tolist();indices=base.indices.tolist();indptr=base.indptr.tolist();rhs=arr['rhs'][unchanged].tolist();records=[];coefficient_count=0
 for number,c in enumerate(eliminated):
  p,q=C.indptr[c:c+2];ids=C.indices[p:q];vals=C.data[p:q];low=ids[vals<0];up=ids[vals>0];assert len(low) and len(up)
  rows={int(i):row(A,int(i)) for i in ids}
  for li in low:
   for ui in up:
    li=int(li);ui=int(ui);l=rows[li];u=rows[ui];al=l[int(c)];au=u[int(c)];assert al<0<au
    g={j:au*l.get(j,F(0))-al*u.get(j,F(0)) for j in l.keys()|u.keys()};assert g[int(c)]==0;g={j:v for j,v in g.items() if v};assert all(remap[j]>=0 for j in g)
    b=au*F(float(arr['rhs'][li]))-al*F(float(arr['rhs'][ui]));maximum=max_at_bounds(g,arr['bounds']);status=1 if not g and b>=0 else 2 if maximum is not None and maximum<=b else 0
    dest=-1;power=0
    if not status:
     power=dyadic_scale(g,b);scale=F(2)**power;dest=len(rhs)
     for j,v in sorted(g.items()):
      f=float(v*scale);assert math.isfinite(f) and f!=0;indices.append(int(remap[j]));data.append(f);coefficient_count+=1
     indptr.append(len(data));bf=float(b*scale);assert math.isfinite(bf) and (b==0 or bf!=0);rhs.append(bf)
    records.append((int(c),li,ui,dest,status,power))
  if number%4000==0:print(json.dumps(dict(role=out.name,variables=number,pairs=len(records),rows=len(rhs))),flush=True)
 B=sparse.csr_matrix((np.asarray(data),np.asarray(indices),np.asarray(indptr)),shape=(len(rhs),len(ret)));sparse.save_npz(out/'matrix.npz',B)
 np.savez_compressed(out/'arrays.npz',rhs=np.asarray(rhs),cost=arr['cost'][ret],bounds=arr['bounds'][ret],equality_rhs=arr['equality_rhs'])
 np.savez_compressed(out/'projection_ledger.npz',pairs=np.asarray(records,dtype=np.int64),unchanged_rows=unchanged,retained_columns=ret,eliminated_columns=eliminated)
 return dict(source_shape=list(A.shape),source_nnz=A.nnz,shape=list(B.shape),nnz=B.nnz,eliminated_columns=len(eliminated),copied_rows=len(unchanged),pair_rows=len(records),kept_pairs=sum(r[4]==0 for r in records),constant_tautologies=sum(r[4]==1 for r in records),bound_implied_pairs=sum(r[4]==2 for r in records),generated_coefficients=coefficient_count)
def arithmetic_examples():
 # Local exact inequalities only; neither example is an optimization input.
 lo=lambda y:F(y-1,2);hi=lambda y:F(2+2*y,3)
 first=dict(source=[[-2,1,1],[3,-2,2]],projected_coefficient=-1,projected_rhs=7,at_y0=[str(lo(0)),str(hi(0)),str((lo(0)+hi(0))/2)],at_y_minus8_empty=lo(-8)>hi(-8),at_y_minus7_singleton=str(lo(-7)))
 assert first['at_y_minus8_empty'] and lo(-7)==hi(-7)==-4 and first['at_y0'][2]=='1/12'
 z=F(1,3);f=float(z);err=F(f)-z
 second=dict(source=[[3,1],[-3,-1]],projected_zero_tautology=True,exact_singleton=str(z),once_rounded=f,rounding_error=str(err),binary64_exact_lift_exists=False)
 assert err!=0 and 3*z==1 and 3*F(f)!=1
 return [first,second]

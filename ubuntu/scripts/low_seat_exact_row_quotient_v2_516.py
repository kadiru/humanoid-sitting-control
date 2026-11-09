"""Exact positive-ray dominance quotient of saved binary64 inequalities.

Preparation algebra only. No LP, transformed-basis rebuild, or actual state.
"""
from pathlib import Path
from fractions import Fraction
from math import gcd
from functools import reduce
import json, hashlib
import numpy as np
from scipy import sparse

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
SPEC=dict(
 scope='NEW exact positive-ray dominance quotient of BOTH saved original490 and mixed504 full LP inequality systems. Preparation only, no solver or candidate.',
 rationale='512 obtained only a rejected partial master and then stopped. Instead of continuing its cuts or changing their selection, prove a fixed subset equivalent to every full inequality BEFORE a solve. No512 vector or cut queue is an input. Existing basis, unit scales and all row coefficients remain immutable.',
 signature='Treat every finite binary64 coefficient as its exact dyadic rational. Clear power-of-two denominators and divide the integer coefficient vector by its positive integer gcd. Retain coefficient signs, column indices and the epigraph/eta column. Do not flip the sign, threshold coefficients, or round normalized coefficients. Signature columns are sorted without changing source storage or summing coefficients; duplicate stored columns reject. Dictionary keys use exact integer tuple equality, not a hash digest alone.',
 dominance='Within a positive ray A_i=s_i*k, s_i>0, keep the row with smallest exact rational b_i/s_i; ties choose smallest source row ID. Then A_i=lambda_i*A_rep with lambda_i>0 and b_i-lambda_i*b_rep>=0 EXACTLY. This proves every eliminated inequality from one retained inequality for all real vectors. Negative rays are distinct. Both consistency signs remain either explicit or covered by their own exact implication proof.',
 zero='A zero-LHS row with nonnegative exact RHS is a tautology. A zero-LHS row with negative RHS is a contradiction and is retained individually; never conceal it or assert global physical infeasibility. Exact stored zeros can be ignored only when constructing a signature; copied source matrices remain unmodified.',
 paired='Construct independent quotients for original and transformed binary64 matrices/RHS. Keep the UNION of their representative row IDs and every contradiction. Thus every removed row has a retained implication witness in EACH representation. No claim that rounded504 equals original490 is required. A row dominant in only one representation cannot be deleted unless independently covered in the other.',
 proof='For either system, the selected rows are a subset of the original rows, so full feasibility implies subset feasibility. For every omitted nonzero row, positive scaling of its selected representative plus the nonnegative exact RHS slack proves the omitted row. Omitted zero rows are tautologies. Thus subset feasibility implies full feasibility in exact real arithmetic over the stored binary64 coefficients. The objective, variable bounds, all equalities and their RHS are copied unchanged. This is constraint reduction, not variable elimination or a new coordinate basis.',
 numerical='Do NOT multiply the solver tolerance by the implication scale or infer full binary64 feasibility from a reduced residual. Even tiny representative errors can amplify. Any future returned unedited transformed vector and reconstructed ORIGINAL t=S*z+a must pass EXHAUSTIVE original AND transformed full inequality/equality/bound checks at unchanged1e-7. Failure is rejection, not clipping, rounding, a waived sign, or an invitation to continue512. Exact original interval eta>0 remains mandatory before490 encoding.',
 ledger='Each source row stores its chosen representative ID (minus1 only for a proved tautology), first nonzero binary64 pivot, support length and representative RHS comparison result. Immutable source coefficients plus two saved binary64 pivots specify lambda as an EXACT rational, without a lossy decimal ratio. Independent audit verifies every omitted coefficient by exact rational cross-products and RHS implication; it does not rerun the gcd grouping algorithm.',
 full_domain='All331888final/331495retained path criteria,156647HARD/noepigraph,174848rate/window,68250two-sided unit-scaled consistency rows,92267finite intervals including11906fullboxes/80155trust/206drop and every pivot remain represented or exactly implied.1411actual normal/hover and1366homogeneous LP equalities remain. No original440 +/-.02box,8eps/4ULP/32ULP/4302^-23 reserve or unit scale changes.392constantcontact/timeidentity omission remains optimizer-only.',
 reference='ONLY immutable490 and504 saved complete primary/witness matrices, arrays and equalities. No old basis/inverse/incidence/model/root/map fixture or transform product is rebuilt. Original actualX/Y/J, canonical auxiliary and exact drops remain unchanged. The witness ceiling.09895564659603738 is an inherited optimistic ASSEMBLY fixture, NOT a solved result.',
 future='No518 algorithm, resource/LP budget, changed-state quotient lifecycle or run wrapper is frozen here. Any new state/ceiling would need its own valid full assembly and exact quotient certificates; this reference subset is not universal. Row count reductions do not demonstrate solver speed, timeout cause, a primal witness or mechanical feasibility.',
 admission='Original full witness certification, exact finite-domain certification and all actual signed rows, phase,normal1e-13,hover1e-8,unit-specific closure/geometry/contactframe/COM/equilibrium/clock/windows/work/filter/runtime/physics/PPO gates remain. Auxiliary or exact quotient success never admits execution.',
 lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()

def quotient(M,b):
 """Return an exact-ray implication ledger; no solve or input mutation."""
 assert M.format=='csr'
 assert np.all(np.isfinite(M.data)) and np.all(np.isfinite(b))
 groups={};winners=[];scores=[]
 group_ids=np.full(M.shape[0],-1,dtype=np.int64)
 pivots=np.zeros(M.shape[0]);lengths=np.zeros(M.shape[0],dtype=np.int32)
 contradictions=[]
 for i in range(M.shape[0]):
  start,end=M.indptr[i:i+2];data=M.data[start:end];cols=M.indices[start:end]
  nonzero=data!=0;vals=data[nonzero];cols=cols[nonzero]
  order=np.argsort(cols);cols=cols[order];vals=vals[order]
  assert np.all(np.diff(cols)>0), "Duplicate stored columns require a separate exact-sum contract"
  lengths[i]=len(vals)
  if not len(vals):
   if b[i]<0:contradictions.append(i)
   continue
  pivots[i]=vals[0]
  ratios=[float(x).as_integer_ratio() for x in vals]
  exponents=[d.bit_length()-1 for _,d in ratios];power=max(exponents)
  integers=[n << (power-e) for (n,_),e in zip(ratios,exponents)]
  common=reduce(gcd,integers,0);assert common>0
  primitive=tuple(n//common for n in integers)
  key=(tuple(int(c) for c in cols),primitive)
  scale=Fraction(common,1 << power)
  score=Fraction(float(b[i]))/scale
  g=groups.get(key)
  if g is None:
   g=len(winners);groups[key]=g;winners.append(i);scores.append(score)
  elif score<scores[g]:winners[g]=i;scores[g]=score
  group_ids[i]=g
 representatives=np.full(M.shape[0],-1,dtype=np.int64)
 nz=group_ids>=0;representatives[nz]=np.asarray(winners,dtype=np.int64)[group_ids[nz]]
 representatives[contradictions]=contradictions
 kept=np.unique(representatives[representatives>=0])
 return dict(representative=representatives,pivot=pivots,support_length=lengths,representatives=kept,
  tautology_rows=np.flatnonzero(representatives<0),contradiction_rows=np.asarray(contradictions,dtype=np.int64))

def new_arithmetic_case():
 # One new matrix implication fixture, not an old map/direction/candidate.
 m=np.array([[1.,2.,0.],[3.,6.,0.],[-1.,-2.,0.],[1.,np.nextafter(2.,np.inf),0.],
             [1.,2.,1.],[0.,0.,0.],[0.,0.,0.],[2.,4.,0.],
             [2.**-1022,2.**-1021,0.]])
 b=np.array([3.,6.,-1.,3.,3.,0.,-2.**-1000,4.,3.*2.**-1022])
 t=m.copy();t[7,1]=np.nextafter(4.,np.inf)
 return sparse.csr_matrix(m),sparse.csr_matrix(t),b

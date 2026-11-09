"""Exact coupled-rate plus inherited-box conic certificates. No numerical solver."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time,itertools
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
TARGETS={'old_window_upper','old_window_lower','mesh_window_upper','mesh_window_lower','clock_target_upper','clock_target_lower','old_ramp_upper','old_ramp_lower'}
RATES={'mesh_rate_increasing','mesh_rate_decreasing'}
SPEC=dict(
 scope='NEW exact coupled-halfspace/box support certificates for inverse-filter command-window rows of immutable546 ORIGINAL-native systems. Preparation only; no robot LP, numerical optimizer, path, candidate or physical admission.',
 distinction='546 used only unconditional singleton/direct-bound premises. This construction adds ONE mandatory retained original mixed whole-clock rate inequality to each selected target certificate, then completes the exact coefficient defect with the inherited box. It does not build or traverse a forest, compose paths, regroup positive rays, propagate or tighten endpoints, change coordinates, lift definitions, dualize the LP, or change a solver parameter.',
 inputs='Immutable546 original matrices, arrays, equality, original490 IDs, retained516 positions, exact endpoints and owners; 547/517 proofs inherited by hashes. Saved446 keep/group labels and490 extra-feature labels are read, not reassembled. Actual472 INITIAL X/Y/J and490 canonical auxiliary/exact drops remain unchanged.',
 selection='Protect ALL mesh_rate_increasing/decreasing rows and ALL non-target rows. A rate premise must have exactly two opposite nonzero native command coefficients for the same joint. A target is an original old_window, mesh_window, clock_target or old_ramp upper/lower row with at least two native command coordinates of one joint. Choose the SMALLEST original490-ID rate premise whose two native columns occur in the target and which has at least one same-sign native target coefficient. Choose ONCE without comparing implication outcomes; no alternative premise retry. Keep targets without such a premise untested.',
 theorem='For target a.w<=b, retained premise p.w<=q, and unchanged exact inherited endpoints L,U, EVERY lambda>=0 gives a.w <= lambda*q + sum_j e_j*(U_j if e_j>0 else L_j), e=a-lambda*p. The conic certificate is lambda times the retained mixed row plus positive signed-singleton/direct-bound multiples completing e exactly. A nonnegative exact slack b-bound proves implication. All premises remain explicitly retained, so removing only certified targets preserves the exact-real feasible set/objective over stored binary64 coefficients. All78791 original variables/cost/domain/E stay unchanged.',
 breakpoints='The upper-bound function is convex piecewise linear in lambda; residual signs can change only at a_j/p_j. Test the finite set of STRICTLY POSITIVE exact ratios for nonzero premise coefficients, including the original scalar if present, in ascending order. Do NOT test lambda=0: the completed546 box tests are not repeated. Save EVERY new ratio and exact slack. Select the largest tested slack, ties first/smallest ratio. No search beyond these ratios, no adaptive bound pass, no optimizer, and no claim of global optimal support or nonredundancy from negative results.',
 arithmetic='Fractions of stored binary64 source coefficients/RHS and saved rational546 endpoints. No coefficient rounding/threshold, endpoint encoding/construction, candidate repair, numerical linear solve or small numerical LP. New exact coupled-support evaluations and conic coefficient/RHS products are counted explicitly; these are algebraic sufficient-implication tests, not actual mechanics progress.',
 resources='ONE preparation of TWO distinct saved reference systems and ONE independent audit. Each240s total wall from before scientific imports, checks between operations,40GiB address space, peak unmeasured. These are symbolic preparation caps, not an extension of45s robot LP. First input/resource failure terminal; no certified partial reduction or adaptive retry.',
 audit='Independent571 imports no producer/optimizer. Independently reconstructs fixed premise selection and all exact positive ratio choices, then builds positive conic certificates using saved516 singleton owners/direct bounds, checking every residual coefficient and exact RHS/slack. Checks positive AND negative decisions and every unchanged retained CSR/RHS/cost/domain/E/ID. Inherits endpoint tightness and517/547 implication proofs by hashes, never replays them.',
 future='Any later complete numerical construction must be separately frozen. Returned78791 coordinates stay bitwise unchanged, with full reduced546/516/FULL490 primal/equality/bound and required KKT checks, exact original unscaled interval eta>0 before immutable490 finite encoding, fresh mechanics/J at newX and all actual/geometry/filter/runtime/72retention/PPO gates. Actual primary sets the actual interior ceiling; old reference ceiling remains illustrative only. Changed state/trust/actual ceiling requires new valid assembly/proofs within unchanged45s total.',
 limits='Exact-real implication does not bound floating residual amplification or establish finite/tolerance feasible-set equality, witness existence, performance, conditioning, solver deadline cause, physical progress or effort savings. No old internal/rejected vector is read.',
 physical_admission=False,new_solver=False,new_candidate=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(t):
 if time.monotonic()-t>240:raise TimeoutError('240s symbolic coupled-support cap; no certified partial reduction')
def row(A,i):
 b,e=A.indptr[i:i+2];return {int(j):F(float(v)) for j,v in zip(A.indices[b:e],A.data[b:e])}
def labels():
 z=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');groups=z['groups'][z['keep']]
 er=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];cmd={10540+i:(int(v)//75,int(v)%75-39) for i,v in enumerate(er) if 39<=int(v)%75<55}
 assert len(groups)==331495 and len(cmd)==21840
 return groups,cmd
def select(A,ids,groups,cmd,t):
 rates={};eligible=[];premises=[]
 for i,rid in enumerate(ids):
  check(t)
  if rid>=len(groups):continue
  g=str(groups[rid]);b,e=A.indptr[i:i+2];native=[int(j) for j in A.indices[b:e] if j!=78790]
  if g in RATES:
   if len(native)!=2:continue
   assert all(j in cmd for j in native) and cmd[native[0]][1]==cmd[native[1]][1]
   rr=row(A,i);assert rr[native[0]]==-rr[native[1]]
   rates.setdefault(tuple(native),[]).append(i);premises.append(i)
  if g in TARGETS and len(native)>=2:
   assert all(j in cmd for j in native) and len({cmd[j][1] for j in native})==1
   eligible.append(i)
 chosen={}
 for i in eligible:
  check(t);a=row(A,i);native=sorted(j for j in a if j!=78790);possible=[]
  for pair in itertools.combinations(native,2):
   for p in rates.get(pair,[]):
    pr=row(A,p)
    if any(a[j]*pr[j]>0 for j in pair):possible.append(p)
  if possible:chosen[i]=min(possible,key=lambda p:int(ids[p]))
 return eligible,premises,chosen
def derive(role,out,t):
 src=D/'box_support_preparation_546'/role;A=sparse.load_npz(src/'matrix.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));ids=np.load(src/'original_row_ids.npy');source=np.load(src/'retained_source_positions.npy');E=sparse.load_npz(src/'equality.npz')
 assert A.shape[1]==78791 and A.has_canonical_format and np.all(A.data!=0) and E.nnz==0
 ep=[[F(x) if x is not None else None for x in pair] for pair in read(src/'endpoints.json')];groups,cmd=labels();eligible,premises,chosen=select(A,ids,groups,cmd,t)
 decisions=np.zeros(A.shape[0],np.uint8);decisions[eligible]=3;total=0;terms=0;removed=[];maxbits=0
 with (out/'certificates.jsonl').open('w') as f:
  for i,p in chosen.items():
   check(t);a=row(A,i);pr=row(A,p);ratios=sorted({a.get(j,F(0))/v for j,v in pr.items() if a.get(j,F(0))*v>0});assert ratios
   tests=[]
   for lam in ratios:
    check(t);value=lam*F(float(arr['rhs'][p]));e={j:a.get(j,F(0))-lam*pr.get(j,F(0)) for j in a.keys()|pr.keys()}
    for j,v in sorted(e.items()):
     if not v:continue
     endpoint=ep[j][int(v>0)];assert endpoint is not None;value+=v*endpoint;terms+=1
    slack=F(float(arr['rhs'][i]))-value;tests.append([str(lam),str(slack)]);total+=1;maxbits=max(maxbits,slack.numerator.bit_length(),slack.denominator.bit_length())
   best=max(range(len(tests)),key=lambda k:F(tests[k][1]));positive=F(tests[best][1])>=0;decisions[i]=2 if positive else 1
   if positive:removed.append(i)
   f.write(json.dumps([int(i),int(p),tests,int(best)],separators=(',',':'))+'\n')
 keep=np.flatnonzero(decisions!=2);assert all(decisions[p]==0 for p in premises)
 sparse.save_npz(out/'matrix.npz',A[keep].tocsr());sparse.save_npz(out/'equality.npz',E)
 np.savez_compressed(out/'arrays.npz',rhs=arr['rhs'][keep],cost=arr['cost'],bounds=arr['bounds'],equality_rhs=arr['equality_rhs'])
 np.save(out/'retained_546_positions.npy',keep);np.save(out/'source_516_positions.npy',source[keep]);np.save(out/'original_row_ids.npy',ids[keep]);np.save(out/'decisions.npy',decisions)
 return dict(source_shape=[int(x) for x in A.shape],source_nonzeros=int(A.nnz),reduced_shape=[int(len(keep)),int(A.shape[1])],reduced_nonzeros=int(A[keep].nnz),protected_rate_premises=int(len(premises)),eligible_targets=int(len(eligible)),selected_targets=int(len(chosen)),unmatched_untested=int(len(eligible)-len(chosen)),new_positive_ratio_support_evaluations=int(total),new_residual_endpoint_products=int(terms),proved_removed_rows=int(len(removed)),insufficient_targets=int(len(chosen)-len(removed)),largest_fraction_bits=int(maxbits),zero_lambda_tests=0,new_endpoints=0,old_box_tests_repeated=0,variables_removed=0,cost_domain_equality_changed=False)

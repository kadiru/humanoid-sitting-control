"""Exact nonparallel multirow implication by support of an antecedent box. No solver."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(
 scope='NEW one-pass exact multirow box-support implication on immutable516 ORIGINAL representative LPs, including their original scalar. All78791 columns retained. Preparation only, no optimizer/candidate/path.',
 distinction='Unlike516 positive-ray grouping or536 command-forest paths, a mixed target row is dominated by a positive conic combination of independent native-coordinate bound premises. No ray regrouping, graph/path, coordinate substitution, epigraph removal, torque elimination or changed solver parameter.',
 initialization='Use only saved original matrix/RHS/cost/bounds/E and original-row IDs. Inherit517 implication provenance by hashes. No old internal/rejected/accepted solver vectors, model/J/root or initial assembly.',
 antecedents='Keep EVERY source row with <=1 stored nonzero explicitly. Each singleton a*x_j<=b supplies exact bound b/a, with its original sign; original finite variable bounds also supply premises. For each side choose the tightest exact endpoint, ties retain the first premise (direct bound before ascending source row). Do not propagate bounds using mixed rows.',
 theorem='For a mixed row a.x<=b and each nonzero a_j, use U_j for a_j>0 and L_j for a_j<0. If every endpoint exists and M=sum a_j*endpoint_j<=b EXACTLY, the row follows. Its conic certificate uses positive a_j/p_j times the chosen singleton p_j*x_j<=q_j, or |a_j| times the signed direct variable bound. The resulting coefficients equal a exactly and RHS equals M. All premises remain explicit; no circular implication.',
 converse='The reduced system is a subset of the source with identical cost/bounds/equalities; implication certificates restore every removed inequality, so the exact-real feasible set and objective are identical over STORED binary64 source coefficients. Native/scalar coordinates need no reconstruction. This does not assert equivalence of floating solver residuals, or any522 rounded projection.',
 arithmetic='Fractions of all stored binary64 coefficients/RHS/endpoints; no tolerances, coefficient thresholds, rounding, row scaling or box expansion. Keep constants/singletons without re-proving old tautologies. Every mixed row gets one decision: exact implied, exact box support insufficient, or missing endpoint untested. Save exact slack for every tested row including negative decisions.',
 resource='ONE preparation of TWO reference systems, each row visited once for bound extraction and once for support. ONE independent audit. Each program has240s wall checks between rows,40GiB address space. No LP iterations, old45s LP deadline extension, small fixtures or adaptive retries. A resource/input failure is terminal; partial artifacts are not certified reductions.',
 audit='Independent code imports no producer. Validate every selected endpoint against its premise and tightness against all unary premises, then construct exact positive conic RHS/coefficients separately for ALL mixed-row decisions. Verify retained CSR/RHS/IDs and unchanged domain/cost/E. No optimizer or repeated517 proof.',
 future='Any future solver needs a complete frozen wrapper/resources/auditor. All returned78791 coordinates unchanged and FULLsource/FULL490 numerical checks at unchanged thresholds before exact original interval eta>0/immutable490 finite encoding. Actual primary defines actual interior ceiling; reference witness ceiling stays illustrative. Changed state/trust/actual ceiling needs fresh valid original assembly/provenance and new implication certificates.',
 limits='No feasibility, convergence, conditioning, deadline-cause, speed, physical support or effort claim. Negative box tests prove only this sufficient implication fails, not global nonredundancy. All physical/filter/runtime/72retention/PPO gates unchanged.',
 physical_admission=False,new_solver=False,new_candidate=False)
def check_time(start):
 if time.monotonic()-start>240:raise TimeoutError('240s exact preparation/audit resource cap; no certified partial result')
def derive(A,arr,ids,out,start):
 assert A.has_canonical_format and A.has_sorted_indices and np.all(np.isfinite(A.data)) and np.all(A.data!=0)
 assert np.all(np.isfinite(arr['rhs'])) and arr['bounds'].shape==(A.shape[1],2)
 m,n=A.shape; endpoints=[[None,None] for _ in range(n)];owners=np.full((n,2),-2,dtype=np.int64)
 # owner -1 denotes the corresponding original direct bound, -2 absent.
 for j in range(n):
  for side in (0,1):
   q=arr['bounds'][j,side]
   assert not np.isnan(q)
   if np.isfinite(q):endpoints[j][side]=F(float(q));owners[j,side]=-1
 unary=np.flatnonzero(np.diff(A.indptr)==1)
 for row in unary:
  check_time(start);k=A.indptr[row];j=int(A.indices[k]);a=F(float(A.data[k]));side=int(a>0);q=F(float(arr['rhs'][row]))/a;old=endpoints[j][side]
  if old is None or (q<old if side else q>old):endpoints[j][side]=q;owners[j,side]=row
 for lo,hi in endpoints:
  if lo is not None and hi is not None and lo>hi:raise ValueError('Exact antecedent bounds inconsistent; no reduction/candidate issued')
 (out/'endpoints.json').write_text(json.dumps([[None if q is None else str(q) for q in pair] for pair in endpoints]))
 np.save(out/'bound_owners.npy',owners)
 decision=np.zeros(m,dtype=np.uint8);terms=0;maxbits=0;minimum=None;maximum=None
 # 0 protected <=1 entry;1 tested insufficient;2 exact implication;3 untested missing endpoint.
 with (out/'exact_decisions.jsonl').open('w') as f:
  for i in range(m):
   check_time(start);b,e=A.indptr[i:i+2]
   if e-b<=1:continue
   ep=[endpoints[int(j)][int(a>0)] for j,a in zip(A.indices[b:e],A.data[b:e])]
   if any(q is None for q in ep):decision[i]=3;continue
   value=sum((F(float(a))*q for a,q in zip(A.data[b:e],ep)),F(0));slack=F(float(arr['rhs'][i]))-value
   terms+=int(e-b);decision[i]=2 if slack>=0 else 1
   maxbits=max(maxbits,slack.numerator.bit_length(),slack.denominator.bit_length())
   minimum=slack if minimum is None else min(minimum,slack);maximum=slack if maximum is None else max(maximum,slack)
   f.write(json.dumps([i,str(slack)],separators=(',',':'))+'\n')
 keep=np.flatnonzero(decision!=2);assert np.all(decision[unary]==0)
 B=A[keep].tocsr();sparse.save_npz(out/'matrix.npz',B)
 np.savez_compressed(out/'arrays.npz',rhs=arr['rhs'][keep],cost=arr['cost'],bounds=arr['bounds'],equality_rhs=arr['equality_rhs'])
 np.save(out/'retained_source_positions.npy',keep);np.save(out/'original_row_ids.npy',ids[keep]);np.save(out/'decisions.npy',decision)
 return dict(source_shape=list(A.shape),source_nonzeros=int(A.nnz),reduced_shape=list(B.shape),reduced_nonzeros=int(B.nnz),unary_premises_retained=len(unary),constant_rows_retained=int(np.count_nonzero(np.diff(A.indptr)==0)),finite_endpoints=int(np.count_nonzero(owners!=-2)),selected_row_endpoints=int(np.count_nonzero(owners>=0)),selected_direct_endpoints=int(np.count_nonzero(owners==-1)),mixed_rows_tested=int(np.count_nonzero((decision==1)|(decision==2))),proved_removed_rows=int(np.count_nonzero(decision==2)),negative_box_support_tests=int(np.count_nonzero(decision==1)),missing_endpoint_untested=int(np.count_nonzero(decision==3)),exact_support_terms=terms,largest_saved_fraction_bits=maxbits,minimum_exact_slack=None if minimum is None else str(minimum),maximum_exact_slack=None if maximum is None else str(maximum),all_premise_rows_retained=True,native_variables_removed=0,scalar_variables_removed=0,source_coefficients_changed=False)

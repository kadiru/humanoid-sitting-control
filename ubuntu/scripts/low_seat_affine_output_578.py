"""Exact rational elimination of auxiliary output graphs on the saved zero face."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
N=78791;N0=10540;BLOCKS=1365;OLD_E=1366
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(
 scope='NEW exact rational dimension reduction of the saved572 PRIMARY zero face by eliminating ONLY35 direct auxiliary outputs per state:16 commands,16 torques,3 COM. No numerical solver, candidate, model, root or physical experiment.',
 selection='Fixed original490 layout: retained prior10540 controls and first15 physical-X coordinates in every50-column block, plus original scalar78790. Eliminated columns10540+50*b+[15..49]. Pivot equality rows1366+50*b+([14..45]+[47..49]). No pivot search, alternatives, graph traversal, conditioning choice or adaptive selection.',
 structure='Every selected equality must contain exactly ONE eliminated coordinate, its designated nonzero pivot d. Other terms are retained coordinates only. Every unselected equality must contain NO eliminated coordinate. Check all47775 pivots and complete auxiliary incidence exactly; retain all20475 foot/moment/gap equalities,1366 oldzeroE andscalarzero equality. No inversion, matrix factorization, mechanics solve or15-root. Physical root/leg X coordinates are retained as independent variables.',
 equations='For pivot d_j*z_j+r_j*u=h_j, define exact affine graph z_j=(h_j-r_j*u)/d_j. For each inequality a_u*u+sum a_j*z_j<=b, lambda_j=a_j/d_j, reduced coefficients a_u-sum lambda_j*r_j and RHS b-sum lambda_j*h_j. All rational operations use exact stored binary64 values via Fraction. Retain every source inequality in its original order,even constants/zeros; retain every unselected equality. No original condition discarded; eliminated-coordinate bounds must be unbounded and objective coefficients zero.',
 theorem='Every REAL solution u of exact rational reduced system has unique REAL extension by these affine graphs satisfying ALL572 equalities/inequalities/bounds/cost. Every572 feasible vector projects to reduced feasible u. Thus exact projection/objective equivalence to572 ZERO FACE only, which remains possiblyEMPTY/proper subset of original490. Not equivalence towholeoriginalLP, not existence/rank ofremaining system/originaloptimumzero/speed/conditioning evidence.',
 distinction='Substantive elimination of47775 auxiliary columns andtheir47775 affine equations; not a coordinate-basis rename. Unlike522 torque inequality Fourier-Motzkin projection, this uses exact affine equalities onalreadyrestrictedt0 andeliminates commands/torques/COM together. Unlike496 approximateinverse residualcoordinates there is no inverse or nativecoordinatebasis. Unlike15root,allphysicalXvariables stay. No552finitebounds/558definitions/564dualization/newface/oldinternalor rejectedvectors.',
 storage='Save exact rational sparse inequality rows and RHS as JSONL integer numerator/positive denominator pairs; save equality graph in original coordinates and retained originalcolumn/row maps. No rounded binary64 reduced matrix or RHS is generated. Negative lambda allowed because pivot premises are equalities,not one-sided conic inequalities. Zero rational coefficients omitted only from storage,not rows. Source572scalar/RHS/oppositepair andanalyticdual proofs inheritedhash,not replayed.',
 finite_limit='An exact graph value need not be binary64 even when retainedu isbinary64. NOuniversalfiniteextension/exactfinite-feasible-set/tolerance-equivalence claimed. A future numericalrepresentation requires a separately frozen rounding/error policy; no numericalmatrixor solver exists here. Any future retained31016 coordinates must remainBITWISEbackendvalues; eliminated47775 outputs must each be evaluated as exact graph sum/division and roundedONCE,neverclip/repair/normalize retainedor reconstructedcoordinates. Reject nonfinite/failed fulloriginal checks. No candidategraph evaluation is performed now.',
 future_certificate='Original573analytic lowerbound can remain unchanged for a successful already-zero scalar; inheritedonly,no newdualproof. FULLreduced rational/572/570/546/516/FULL490 primal/equality/bound/requiredKKT <=1e-9 insideunchanged1e-7; numericalKKT not exactoptimality/objectiveerrorbound. Do not infer original residuals from roundedreducedcoefficients or graph residuals. PrimaryNEVERmapped; SAMEcompleteoriginaldirection computes modeledphi,ACTUALinteriorceiling andprediction,not assumed0fromt0. Separate completefullpath/interior/newinput/actualauditor/resourcefreeze needed beforeanyLP.',
 resources='ONEproducer240s andONEindependentaudit240s,totalfrombefore scientificimports;40GiBaddressspace,checksbetweenoperations,peakUNMEASURED. Symbolic rational preparation,not extension of45srobotLP. No smallfixture/LP/linear-systemsolve/customfactor/mechanics/AD/root/geometry/finiteencoding/physics. Failures terminal stage; no completedcalculation replay.',
 audit='Independent579 imports NOproducer/optimizer. Check designatedpivot incidence/domain/cost andexact graphcopies; independently ADDsavedlambda times ORIGINALpivot equalityrows to every saved reducedrow andRHS, reconstructing ALLoriginal572 coefficients/RHS exactly. This is reverse reconstruction,not producer forwardsubstitution. Check every unselectedequality,retainedbounds/cost/IDs/maps andsourcehashes. Save verification arrays; inherited572573/559571547517 proofs not replayed.',
 physical_admission=False,numerical_algorithm_exists=False,rounded_matrix_exists=False,primal_witness_exists=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s exact affine-output preparation cap')
def pair(x):return [x.numerator,x.denominator]
def exact_row(A,i):
 b,e=A.indptr[i:i+2];return {int(j):F(float(v)) for j,v in zip(A.indices[b:e],A.data[b:e]) if v!=0}
def build(out,start):
 src=D/'zero_face_preparation_572';A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'))
 assert A.shape==(265967,N) and E.shape==(69617,N) and A.has_canonical_format and E.has_canonical_format
 elim=(N0+50*np.arange(BLOCKS)[:,None]+np.arange(15,50)).ravel();piv=(OLD_E+50*np.arange(BLOCKS)[:,None]+np.r_[np.arange(14,46),np.arange(47,50)]).ravel()
 keep=np.setdiff1d(np.arange(N),elim);eqkeep=np.setdiff1d(np.arange(E.shape[0]),piv);assert len(elim)==47775 and len(keep)==31016 and len(eqkeep)==21842
 assert np.all(a['cost'][elim]==0) and np.isneginf(a['bounds'][elim,0]).all() and np.isposinf(a['bounds'][elim,1]).all()
 loc={int(v):i for i,v in enumerate(keep)};eloc={int(v):i for i,v in enumerate(elim)};graph=[];gcoeff=0;maxbits=0
 with (out/'affine_graph.jsonl').open('x') as f:
  for j,(col,row) in enumerate(zip(elim,piv)):
   if j%128==0:check(start)
   terms=exact_row(E,int(row));ec=[k for k in terms if k in eloc];assert ec==[int(col)]
   d=terms.pop(int(col));assert d!=0;rhs=F(float(a['equality_rhs'][row]));assert all(k in loc for k in terms)
   graph.append((d,terms,rhs));gcoeff+=len(terms)+1
   f.write(json.dumps([int(col),int(row),pair(d),pair(rhs),[[k,*pair(v)] for k,v in terms.items()]],separators=(',',':'))+'\n')
 assert E[eqkeep][:,elim].nnz==0;ER=E[eqkeep][:,keep].tocsr();sparse.save_npz(out/'retained_equality.npz',ER)
 np.savez_compressed(out/'maps.npz',retained_columns=keep,eliminated_columns=elim,pivot_rows=piv,retained_equality_rows=eqkeep)
 np.savez_compressed(out/'arrays.npz',cost=a['cost'][keep],bounds=a['bounds'][keep],equality_rhs=a['equality_rhs'][eqkeep])
 np.save(out/'original_row_ids.npy',np.load(src/'original_row_ids.npy'))
 newnnz=0;removedocc=0;products=0;rhsproducts=0;divisions=0;constantrows=0
 with (out/'rational_inequalities.jsonl').open('x') as f:
  for i in range(A.shape[0]):
   if i%128==0:check(start)
   terms=exact_row(A,i);reduced={loc[k]:v for k,v in terms.items() if k in loc};rhs=F(float(a['rhs'][i]));multipliers=[]
   for k,v in terms.items():
    if k not in eloc:continue
    j=eloc[k];d,g,h=graph[j];lam=v/d;divisions+=1;removedocc+=1;multipliers.append([j,*pair(lam)])
    for q,c in g.items():
     z=loc[q];reduced[z]=reduced.get(z,F(0))-lam*c;products+=1
    rhs-=lam*h;rhsproducts+=1
   reduced={k:v for k,v in sorted(reduced.items()) if v};newnnz+=len(reduced);constantrows+=not bool(reduced)
   for v in list(reduced.values())+[rhs]:maxbits=max(maxbits,v.numerator.bit_length(),v.denominator.bit_length())
   f.write(json.dumps([i,[[k,*pair(v)] for k,v in reduced.items()],pair(rhs),multipliers],separators=(',',':'))+'\n')
 check(start)
 return dict(source_columns=N,reduced_columns=len(keep),eliminated_auxiliary_columns=len(elim),retained_physical_X_coordinates=15*BLOCKS,source_inequality_rows=int(A.shape[0]),reduced_inequality_rows=int(A.shape[0]),source_inequality_nonzeros=int(A.nnz),exact_reduced_inequality_nonzeros=newnnz,source_equality_rows=int(E.shape[0]),reduced_equality_rows=int(ER.shape[0]),source_equality_nonzeros=int(E.nnz),retained_equality_nonzeros=int(ER.nnz),graph_nonzeros=gcoeff,pivot_incidence_checks=len(elim),eliminated_inequality_occurrences=removedocc,exact_multiplier_divisions=divisions,exact_coefficient_products=products,exact_RHS_products=rhsproducts,constant_rows_retained=constantrows,largest_fraction_bits=maxbits,reduced_scalar_column=loc[78790],original_scalar_column=78790,rows_discarded=0,original_bounds_changed=False,rounded_coefficients_created=0,primal_graph_evaluations=0,old_proofs_repeated=0)

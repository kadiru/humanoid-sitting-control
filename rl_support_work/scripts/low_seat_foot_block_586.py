"""Exact bilateral-foot block elimination of saved zero-face leg directions."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(scope='NEW exact coupled six-joint leg block elimination from578 rational PRIMARY zero-face. Fixed two6x6 bilateral-foot blocks at each1365state; removes16380 physical leg-direction columns and16380 foot equality rows. This is a dimension reduction,not another78791-variable residual basis or nonlinear root correction.',selection='Use578 source columns10540+15*b+3+6*s+[0..5] and retained equality rows1366+15*b+6*s+[0..5],b0..1364,s0..1. Check all selected rows contain no eliminated coordinate outside their designated6-column block. Retain10540 priorcontrols,4095 rootxy/f directions,andscalar:14636 columns; scalaroriginal78790 isnew14635. No alternative block selection,conditioning search,rank relaxation or numerical pivot threshold.',algebra='Each selected block is B*z+C*v=h. Exact rational Gauss-Jordan computes K=B^-1 with deterministic first-nonzero row pivot; a zero pivot column terminates the whole preparation. Then G=-K*C,k=K*h,z=G*v+k. SaveK,G,k andpivots. This DOES construct2730 NEW exact6x6 leg-block inverses and solve their coefficient/right-hand-side systems; do not report zero factors/linear-systemsolves. It does not call a numerical solver,actual mechanics,J,AD,IK or nonlinear15-root.',substitution='Every source578 rational inequality and every unselected equality is substituted exactly: aReduced=a_v+sum a_z*G,bReduced=b-sum a_z*k. PreserveALL265967inequalityrows/order/IDs,includingconstantrows. All remaining moment/gap equations plusoldzeroE/scalarzero remain explicit. No threshold,rounding,scaling or rowdeletion. Eliminated leg coordinates must havezero cost andunbounded sourceDIRECTbounds; alltheir original trust/fullbox restrictions remain through substituted inequalities.',theorem='Over exact stored-coefficient REAL arithmetic,B*K=I proves every squareB invertible; B*G+C=0 andB*k=h prove the unique graph. Thus reducedREAL feasible set projects bijectively to578 viafootgraph,then to572zeroFACE viaold578auxgraph,withsameobjective. ZeroFACE can beempty/proper originalLPsubset. No originaloptimumzero/existence/remainingrank/conditioning/convergence/speed claim.',distinction='Unlike578 scalar auxiliary-output pivots,these are coupled6x6 physicalleg blocks. Unlike496 approximate50x50 residual-coordinate basis,these inverses areexact withindependentidentity certificates and16380 columns/equalities actually disappear. Retain3rootxy/f degrees/state,not theold15-variable rootproblem;no15rootdrift ornewnonlinearstate. Auxiliaryforces/torquesremainmodeloutputs,notauthority. No580reencoding/numericinnerinputor552inwarddomain isused.',storage='NewA andremainingE arecanonical exact rational JSONL. No roundednumericmatrix/RHS exists.578graph/substitutionproofs inheritedbyhash,not replayed; rationalrows are read as source and onlyNEW footblock substitution performed. Oldauxgraph notreevaluated at anypoint.580compensationcannotbereusedfor thischanged matrix/domain.',finite='No primalpoint formed. A future exact composite lift must evaluate newfootgraphs onbitwise backendretainedv,thenold578auxgraphs ontheEXACT unrounded lifted31016-vector,rounding each64155 eliminatedoriginalcoordinateONCE. Retained14636 coordinatesbitwiseunchanged,scalaralready0. No universalbinary64extension;fulloriginalchecks mandatory,no clip/repair/normalize. Future numericencoding/errorpolicy andindependentpointauditor need separatefreeze.',future='Require exact newreduced/578/572/570/546/516/FULL490 primal/equality/bound/requiredKKT<=1e-9 inside1e-7. Inherited573 analyticdual only,notnewproof. SAMEcompleteoriginaldirectioncomputesmodeledphi/ACTUALinteriorceiling/prediction,primaryNEVERmapped. Zero-facegraphs doNOTapplytoactualnonzerointeriorceiling;freshfulloriginal conservativebranch/newinputproofs beforeLPinside45sTOTALfromMAINentry. Exactoriginalunscaledeta>0before490encoder,newXfreshmechanics/J,allphysical/runtime/72retention/PPOgates unchanged.',resources='240s producer and240s independent audit TOTALfromBEFOREscientificimports/inputchecks;40GiBaddressspace/checksbetweenoperations,peakunmeasured. No numericalLP/smallfixture/primalpoint/physicaltrial. These are symboliccaps,not45srobotLPextension. Anyfailure/cap terminal,no completedcalculationreplay.',audit='Standalone587 importsNOproducer/optimizer anddoesnot refactor/invert. Verifyexact B*K=I,B*G+C=0,B*k=h forALL2730blocks. For everynewinequality/equality,compute signedlambda=a_z*K fromsource eliminatedcoefficients,ADDlambda times ORIGINAL578footErow tosavedreducedrow/RHS,andrecoverentiresourceexactly. Independentreverseidentity,not producerforward substitution. Checkmaps/cost/bounds/IDs/fullcoverage;oldproofsinheritednotreplayed.',physical_admission=False,numerical_algorithm_exists=False,rounded_input_exists=False,primal_witness_exists=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s exact footblock preparation cap')
def pair(x):return [x.numerator,x.denominator]
def row(M,i):
 b,e=M.indptr[i:i+2];return {int(j):F(float(v)) for j,v in zip(M.indices[b:e],M.data[b:e]) if v!=0}
def inverse(B,counts):
 n=6;Q=[B[i][:]+[F(int(i==j)) for j in range(n)] for i in range(n)];pivots=[]
 for j in range(n):
  p=next((i for i in range(j,n) if Q[i][j]!=0),None)
  if p is None:raise ArithmeticError('Singular designated6x6 foot block; no alternative')
  pivots.append(p);Q[j],Q[p]=Q[p],Q[j];d=Q[j][j]
  for t in range(12):
   if Q[j][t]:Q[j][t]/=d;counts['inverse_divisions']+=1
  for i in range(n):
   if i==j or Q[i][j]==0:continue
   m=Q[i][j]
   for t in range(12):
    if Q[j][t]:Q[i][t]-=m*Q[j][t];counts['inverse_products']+=1
 return [r[6:] for r in Q],pivots

def build(out,start):
 src=D/'affine_output_preparation_578';E=sparse.load_npz(src/'retained_equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'));oldmaps=dict(np.load(src/'maps.npz'))
 assert E.shape==(21842,31016) and E.has_canonical_format
 elim=np.array([10540+15*b+3+j for b in range(1365) for j in range(12)],dtype=np.int64);erows=np.array([1366+15*b+j for b in range(1365) for j in range(12)],dtype=np.int64)
 keep=np.setdiff1d(np.arange(31016),elim);eqkeep=np.setdiff1d(np.arange(21842),erows);assert len(keep)==14636 and len(eqkeep)==5462
 assert np.all(a['cost'][elim]==0) and np.isneginf(a['bounds'][elim,0]).all() and np.isposinf(a['bounds'][elim,1]).all()
 eset=set(map(int,elim));loc={int(j):i for i,j in enumerate(keep)};graph={};counts=dict(inverse_divisions=0,inverse_products=0,graph_coefficient_products=0,graph_RHS_products=0,substitution_coefficient_products=0,substitution_RHS_products=0)
 graphnnz=0;knnz=0;maxbits=0
 with (out/'block_graphs.jsonl').open('x') as f:
  for b in range(1365):
   check(start)
   for side in range(2):
    cols=[10540+15*b+3+6*side+j for j in range(6)];rows=[1366+15*b+6*side+j for j in range(6)];rr=[row(E,i) for i in rows]
    assert all(set(r)&eset<=set(cols) for r in rr)
    rest=sorted(set().union(*(set(r)-set(cols) for r in rr)));B=[[r.get(j,F(0)) for j in cols] for r in rr];C=[[r.get(j,F(0)) for j in rest] for r in rr];h=[F(float(a['equality_rhs'][i])) for i in rows]
    K,piv=inverse(B,counts);G=[];kv=[]
    for i in range(6):
     gi=[]
     for j in range(len(rest)):
      v=F(0)
      for t in range(6):
       if K[i][t] and C[t][j]:v-=K[i][t]*C[t][j];counts['graph_coefficient_products']+=1
      gi.append(v)
     v=F(0)
     for t in range(6):
      if K[i][t] and h[t]:v+=K[i][t]*h[t];counts['graph_RHS_products']+=1
     G.append(gi);kv.append(v);terms={rest[j]:x for j,x in enumerate(gi) if x};graph[cols[i]]=(terms,v);graphnnz+=len(terms)
    knnz+=sum(bool(x) for r in K for x in r)
    for x in [x for m in [K,G] for r in m for x in r]+kv:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
    f.write(json.dumps([2*b+side,cols,rows,rest,[[pair(x) for x in r] for r in K],[[pair(x) for x in r] for r in G],[pair(x) for x in kv],piv],separators=(',',':'))+'\n')
 np.savez_compressed(out/'maps.npz',retained_columns=keep,eliminated_columns=elim,pivot_rows=erows,retained_equality_rows=eqkeep,original_columns=oldmaps['retained_columns'][keep])
 np.savez_compressed(out/'arrays.npz',cost=a['cost'][keep],bounds=a['bounds'][keep]);np.save(out/'original_row_ids.npy',np.load(src/'original_row_ids.npy'))
 def transform(terms,rhs):
  reduced={loc[j]:v for j,v in terms.items() if j in loc}
  for j,v in terms.items():
   if j not in graph:continue
   g,k=graph[j]
   for c,x in g.items():reduced[loc[c]]=reduced.get(loc[c],F(0))+v*x;counts['substitution_coefficient_products']+=1
   rhs-=v*k;counts['substitution_RHS_products']+=1
  return {j:v for j,v in sorted(reduced.items()) if v},rhs
 stats={}
 for role in ['inequality','equality']:
  nnz=0;constant=0;total=0
  if role=='inequality':
   def source():
    with (src/'rational_inequalities.jsonl').open() as f:
     for line in f:
      i,t,h,_=json.loads(line);yield i,{j:F(n,d) for j,n,d in t},F(*h)
  else:
   def source():
    for i in eqkeep:yield int(i),row(E,int(i)),F(float(a['equality_rhs'][i]))
  with (out/('rational_inequalities.jsonl' if role=='inequality' else 'rational_equalities.jsonl')).open('x') as f:
   for count,(i,t,h) in enumerate(source()):
    if count%32==0:check(start)
    v,h=transform(t,h);nnz+=len(v);constant+=not bool(v);total+=1
    for x in list(v.values())+[h]:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
    f.write(json.dumps([i,[[j,*pair(x)] for j,x in v.items()],pair(h)],separators=(',',':'))+'\n')
  stats[role]=dict(rows=total,nonzeros=nnz,constant_rows=constant)
 assert stats['inequality']['rows']==265967 and stats['equality']['rows']==5462;check(start)
 return dict(source_columns=31016,reduced_columns=14636,eliminated_leg_columns=16380,retained_prior_controls=10540,retained_rootxy_f_coordinates=4095,exact_six_by_six_inversions=2730,exact_block_coefficient_solves=2730,rank_certificate_entries=2730*36,inverse_nonzeros=knnz,graph_coefficient_nonzeros=graphnnz,original_scalar_column=78790,source_scalar_column=31015,reduced_scalar_column=14635,all_inequalities_retained=True,conditions_discarded=0,statistics=stats,operations=counts,largest_fraction_bits=maxbits,rounded_coefficients_created=0,primal_graph_values=0,old_proofs_repeated=False)

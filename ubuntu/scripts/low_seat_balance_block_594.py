"""Exact saved-coefficient moment/gap block condensation of immutable586."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(
 scope='NEW exact dimension reduction of586 rational PRIMARY zeroFACE. Fixed1365 coupled3x3 moment/gap blocks eliminate4095 remaining physical rootxy/f direction columns/equalities, leaving10540 priorcontrols plus originalscalar78790. No actualstate/model/root evaluation.',
 selection='Source586 columns10540+3*b+[0,1,2], equality FILE POSITIONS1366+3*b+[0,1,2], b0..1364. Each equality record keeps its original578 row label. Verify selected rows contain no eliminated coordinate outside their designated3-column block. Eliminated columns must havezero cost andunbounded ORIGINALdirectbounds. Fixed selection, deterministic firstnonzero exactrowpivot; singularity/incidencefailure terminates entire preparation, no alternateblock/rankthreshold/fallback.',
 algebra='For saved rational B*z+C*v=h, construct NEW K=B^-1 by exactFraction Gauss-Jordan, G=-K*C,k=K*h. This DOES perform1365 new exact3x3 inversions/coefficient-system solves if complete. It does NOT repeat586 inverses, call a numericallinear solver, or perform nonlinear15-root/IK/retraction/drift. Priorcontrols remain independent, physical rootxy/f and leg directions become uniquely determined graph outputs, not eliminated physical requirements.',
 substitution='For EVERY265967 source586 inequality andEVERY unselected equality, use aReduced=a_v+sum a_z*G,bReduced=b-sum a_z*k. Preserve all rows/order/sourceIDs, includingconstantrows, all original/inward/trust/box/consistencysigns/frozenunits through exact substitution. No threshold/scaling/rounding/normalization/rowdeletion. Remaining equality count1367 includes1366oldzero rows andscalarzero, subject to checked source layout.',
 theorem='Independent ALLsquareB*K=I certifies invertibility; B*G+C=0 andB*k=h certify uniqueREALgraph. Exact reducedREALfeasibility lifts uniquely to586, then inherited586footgraph to578 and578auxgraph to572ZERO FACE withsameobjective. Converse projection holds. ZeroFACE mayempty/proper originalLPsubset. No originaloptimumzero/primalexistence/remainingrank/conditioning/speed/convergence/finiteextension claim.',
 distinction='Substantive additional column/equality elimination of the coupled moment/gap constraints after foot elimination, not a renamed face, basis, rowgroup, same-matrix encoding, backend/presolve/crossover retry or old15-variable nonlinearroot. No previous solver/internal/rejected vector selected.',
 storage='Canonical rationalA/E/RHS, maps/cost/originalbounds and NEW K/G/k/pivot traces. Source586/587 coefficients/proofs inheritedbyhash; no oldinverse/identity/forward/reverse proof replay. No588 compensation/roundedE reused. No newnumericinput/LP/witness algorithm exists.',
 finite='No primalpoint constructed. Future backend10541retained values muststayBITWISE,scalarALREADYzero. Evaluate NEW moment/gap graph EXACTLY, then586footgraph onEXACTUNROUNDED14636 vector, then578auxgraph onEXACTUNROUNDED31016 vector. Round EACH68250eliminatedoriginalcoordinateONCE. No roundedintermediate feed, universalbinary64extension, clip/repair/normalization. Fullnewexact/586/578/572/570/546/516/FULL490 and anyfuture numericinput primal/equality/bound/requiredstationarity/absgap<=1e-9 inside1e-7 remainmandatory.573analyticdual inheritedonly,no newproof/pair.',
 future='Separate numericrepresentation/errorpolicy/independent audit andcompleteboundedwitness construction beforeLP. SAMEcompleteoriginalprimarydirection computes modeledphi/theta/ACTUAL(mu+theta)/2 interiorceiling/prediction,primaryNEVERmapped,phi notassumed0. ZeroFACE graphs invalid at nonzero actualinteriorceiling: freshFULLoriginal branch, newstate/trust independentassembly/proofs BEFORELPinside45sTOTALfromMAINentry beforeimports/globalchecks/prep/backend/point/lift/KKT.40GiB/120sparent,24major48LP120maps5dyadics,trust/guards/tolerances unchanged. Exactunscaledeta>0before490encoder,freshnewXmechanics/J,no15rootdrift; allphysical/filter/runtime/72retention/PPOgates unchanged.',
 resources='240s producer and240s independent audit TOTALfromBEFOREscientificimports/globalhashes;40GiBAS, checksbetweenoperations mayovershoot reportednotextension, peakunmeasured. No numericalLP/smallfixture/primalpoint/physicaltrial. Firstfailure/cap terminal; partialfiles/counts preserved, no completedarithmetic replay/restart. These symboliccaps doNOTextend45sLP.',
 audit='Standalone595 importsNOproducer/optimizer andconstructsNOinverse/refactor. Verify every B*K=I,B*G+C=0,B*k=h. For each newA/E, signedlambda=a_z*K fromsource586 coefficients, ADDlambda ORIGINAL586moment/gapE rows/RHS tosaved reducedrow torestoreENTIREsourceEXACTLY. This independentreverse proof differsfromforward producer. Verify maps/cost/bounds/IDs/coverage. Inherited586587/578579/572573/559571547517proofs not replayed.',
 physical_admission=False,numerical_algorithm_exists=False,rounded_input_exists=False,primal_witness_exists=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s exact moment/gap block preparation cap')
def pair(x):return [x.numerator,x.denominator]
def parse(line):
 i,t,h=json.loads(line);return i,{j:F(n,d) for j,n,d in t},F(*h)
def inverse(B,counts):
 Q=[B[i][:]+[F(int(i==j)) for j in range(3)] for i in range(3)];piv=[]
 for j in range(3):
  p=next((i for i in range(j,3) if Q[i][j]),None)
  if p is None:raise ArithmeticError('Singular fixed3x3 moment/gap block; no alternate')
  piv.append(p);Q[j],Q[p]=Q[p],Q[j];d=Q[j][j]
  for t in range(6):
   if Q[j][t]:Q[j][t]/=d;counts['inverse_divisions']+=1
  for i in range(3):
   if i==j or not Q[i][j]:continue
   m=Q[i][j]
   for t in range(6):
    if Q[j][t]:Q[i][t]-=m*Q[j][t];counts['inverse_products']+=1
 return [r[3:] for r in Q],piv

def build(out,start,progress):
 src=D/'foot_block_preparation_586';a=dict(np.load(src/'arrays.npz'));oldmaps=dict(np.load(src/'maps.npz'))
 with (src/'rational_equalities.jsonl').open() as f:E=[parse(line) for line in f]
 assert len(E)==5462 and a['cost'].shape==(14636,) and a['bounds'].shape==(14636,2)
 assert [x[0] for x in E]==oldmaps['retained_equality_rows'].tolist()
 elim=np.arange(10540,14635,dtype=np.int64);erows=np.arange(1366,5461,dtype=np.int64)
 keep=np.concatenate((np.arange(10540,dtype=np.int64),np.array([14635],dtype=np.int64)))
 eqkeep=np.concatenate((np.arange(1366,dtype=np.int64),np.array([5461],dtype=np.int64)))
 assert np.all(a['cost'][elim]==0) and np.isneginf(a['bounds'][elim,0]).all() and np.isposinf(a['bounds'][elim,1]).all()
 assert all(not t and h==0 for _,t,h in E[:1366]) and E[-1][1]=={14635:F(1)} and E[-1][2]==0
 eset=set(map(int,elim));loc={int(j):i for i,j in enumerate(keep)};graph={}
 counts=dict(inverse_divisions=0,inverse_products=0,graph_coefficient_products=0,graph_RHS_products=0,substitution_coefficient_products=0,substitution_RHS_products=0)
 progress.update(stage='blocks',completed_blocks=0,completed_inequality_rows=0,completed_equality_rows=0,operations=counts)
 gnnz=knnz=maxbits=0
 with (out/'block_graphs.jsonl').open('x') as f:
  for b in range(1365):
   check(start);cols=[10540+3*b+j for j in range(3)];rows=[1366+3*b+j for j in range(3)]
   rr=[E[i][1] for i in rows];h=[E[i][2] for i in rows];assert all((set(r)&eset)<=set(cols) for r in rr)
   rest=sorted(set().union(*(set(r)-set(cols) for r in rr)));B=[[r.get(j,F(0)) for j in cols] for r in rr];C=[[r.get(j,F(0)) for j in rest] for r in rr]
   K,piv=inverse(B,counts);G=[];kv=[]
   for i in range(3):
    gi=[]
    for j in range(len(rest)):
     v=F(0)
     for t in range(3):
      if K[i][t] and C[t][j]:v-=K[i][t]*C[t][j];counts['graph_coefficient_products']+=1
     gi.append(v)
    v=F(0)
    for t in range(3):
     if K[i][t] and h[t]:v+=K[i][t]*h[t];counts['graph_RHS_products']+=1
    G.append(gi);kv.append(v);terms={rest[j]:x for j,x in enumerate(gi) if x};graph[cols[i]]=(terms,v);gnnz+=len(terms)
   knnz+=sum(bool(x) for r in K for x in r)
   for x in [x for m in [K,G] for r in m for x in r]+kv:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
   f.write(json.dumps([b,cols,rows,[E[i][0] for i in rows],rest,[[pair(x) for x in r] for r in K],[[pair(x) for x in r] for r in G],[pair(x) for x in kv],piv],separators=(',',':'))+'\n');progress['completed_blocks']=b+1
 np.savez_compressed(out/'maps.npz',retained_columns=keep,eliminated_columns=elim,pivot_rows=erows,retained_equality_rows=eqkeep,original_columns=oldmaps['original_columns'][keep],source578_equality_rows=oldmaps['retained_equality_rows'][eqkeep])
 np.savez_compressed(out/'arrays.npz',cost=a['cost'][keep],bounds=a['bounds'][keep]);np.save(out/'original_row_ids.npy',np.load(src/'original_row_ids.npy'))
 def transform(terms,rhs):
  v={loc[j]:x for j,x in terms.items() if j in loc}
  for j,x in terms.items():
   if j not in graph:continue
   g,k=graph[j]
   for c,y in g.items():v[loc[c]]=v.get(loc[c],F(0))+x*y;counts['substitution_coefficient_products']+=1
   rhs-=x*k;counts['substitution_RHS_products']+=1
  return {j:x for j,x in sorted(v.items()) if x},rhs
 stats={}
 for role in ['inequality','equality']:
  progress['stage']=role;nnz=constant=total=0
  def source():
   if role=='inequality':
    with (src/'rational_inequalities.jsonl').open() as f:
     for line in f:yield parse(line)
   else:
    for i in eqkeep:yield E[int(i)]
  with (out/('rational_inequalities.jsonl' if role=='inequality' else 'rational_equalities.jsonl')).open('x') as f:
   for count,(i,t,h) in enumerate(source()):
    if count%16==0:check(start)
    v,h=transform(t,h);nnz+=len(v);constant+=not bool(v);total+=1
    for x in list(v.values())+[h]:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
    f.write(json.dumps([i,[[j,*pair(x)] for j,x in v.items()],pair(h)],separators=(',',':'))+'\n');progress['completed_'+role+'_rows']=total
  stats[role]=dict(rows=total,nonzeros=nnz,constant_rows=constant)
 assert stats['inequality']['rows']==265967 and stats['equality']['rows']==1367;check(start);progress['stage']='complete'
 return dict(source_columns=14636,reduced_columns=10541,eliminated_rootxy_f_columns=4095,retained_prior_controls=10540,retained_physical_columns=0,exact_three_by_three_inversions=1365,exact_block_coefficient_solves=1365,rank_certificate_entries=12285,inverse_nonzeros=knnz,graph_coefficient_nonzeros=gnnz,original_scalar_column=78790,source_scalar_column=14635,reduced_scalar_column=10540,all_inequalities_retained=True,conditions_discarded=0,statistics=stats,operations=counts,largest_fraction_bits=maxbits,rounded_coefficients_created=0,primal_graph_values=0,old_proofs_repeated=False)

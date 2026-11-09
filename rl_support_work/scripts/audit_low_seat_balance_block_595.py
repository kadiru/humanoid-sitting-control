"""Independent exact moment/gap rank/graph identities and reverse source-row restoration."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime,itertools
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
PROGRESS={};FLAGS={}
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def limit():
 if time.monotonic()-START>240:raise TimeoutError('240s independent moment/gap block audit cap')
def frac(v):
 assert len(v)==2 and type(v[0]) is int and type(v[1]) is int and v[1]>0
 q=F(v[0],v[1]);assert q.numerator==v[0] and q.denominator==v[1];return q
def source_row(line):
 i,t,h=json.loads(line);return i,{j:F(n,d) for j,n,d in t},F(*h)
def same(a,b):
 assert a.dtype==b.dtype and a.shape==b.shape
 assert np.array_equal(a.view(np.uint64),b.view(np.uint64)) if a.dtype.kind=='f' else np.array_equal(a,b)
def run(out):
 assert read(D/'balance_block_preparation_exit_594.json')['actual_exit']==0
 for p,h in read(D/'balance_block_launch_594.json')['source_input_sha256'].items():assert digest(p)==h,p
 src=D/'foot_block_preparation_586';new=D/'balance_block_preparation_594'
 for p,h in read(new/'SHA256.json').items():assert digest(new/p)==h,p
 assert read(D/'foot_block_audit_587/independent_audit.json')['passed']
 with (src/'rational_equalities.jsonl').open() as f:E=[source_row(line) for line in f]
 a=dict(np.load(src/'arrays.npz'));oldmaps=dict(np.load(src/'maps.npz'));maps=dict(np.load(new/'maps.npz'));arr=dict(np.load(new/'arrays.npz'))
 assert len(E)==5462 and [r[0] for r in E]==oldmaps['retained_equality_rows'].tolist()
 elim=[10540+3*b+j for b in range(1365) for j in range(3)];pivrows=[1366+3*b+j for b in range(1365) for j in range(3)]
 eset=set(elim);pivset=set(pivrows);keep=[i for i in range(14636) if i not in eset];eqkeep=[i for i in range(5462) if i not in pivset]
 assert len(keep)==10541 and len(eqkeep)==1367 and keep[-1]==14635
 for n,x in [('retained_columns',keep),('eliminated_columns',elim),('pivot_rows',pivrows),('retained_equality_rows',eqkeep)]:same(np.array(x,dtype=np.int64),maps[n])
 same(oldmaps['original_columns'][keep],maps['original_columns']);same(oldmaps['retained_equality_rows'][eqkeep],maps['source578_equality_rows']);assert maps['original_columns'][-1]==78790
 assert np.all(a['cost'][elim]==0) and np.isneginf(a['bounds'][elim,0]).all() and np.isposinf(a['bounds'][elim,1]).all()
 assert all(not t and h==0 for _,t,h in E[:1366]) and E[-1][1]=={14635:F(1)} and E[-1][2]==0
 same(a['cost'][keep],arr['cost']);same(a['bounds'][keep],arr['bounds']);assert set(arr)=={'cost','bounds'};same(np.load(src/'original_row_ids.npy'),np.load(new/'original_row_ids.npy'))
 blocks=[];colmap={};flags=np.zeros((1365,3),dtype=np.uint8);FLAGS['block_identities_verified.npy']=flags
 products=dict(inverse_identity=0,graph_identity=0,graph_RHS_identity=0,reverse_multiplier=0,reverse_coefficients=0,reverse_RHS=0)
 PROGRESS.update(stage='blocks',completed_blocks=0,completed_inequality_rows=0,completed_equality_rows=0,products=products);knnz=gnnz=maxbits=0
 with (new/'block_graphs.jsonl').open() as f:
  for ix,line in enumerate(f):
   limit();bid,cols,rows,sourceids,rest,K,G,k,piv=json.loads(line);assert bid==ix and ix<1365
   assert cols==[10540+3*ix+j for j in range(3)] and rows==[1366+3*ix+j for j in range(3)] and sourceids==[E[i][0] for i in rows]
   rr=[E[i][1] for i in rows];h=[E[i][2] for i in rows];assert all((set(r)&eset)<=set(cols) for r in rr);assert rest==sorted(set().union(*(set(r)-set(cols) for r in rr)))
   assert len(K)==len(G)==len(k)==len(piv)==3 and all(len(r)==3 for r in K) and all(len(r)==len(rest) for r in G);assert all(type(p) is int and i<=p<3 for i,p in enumerate(piv))
   K=[[frac(x) for x in r] for r in K];G=[[frac(x) for x in r] for r in G];k=list(map(frac,k));B=[[r.get(c,F(0)) for c in cols] for r in rr]
   for i in range(3):
    for j in range(3):
     q=F(0)
     for t in range(3):
      if B[i][t] and K[t][j]:q+=B[i][t]*K[t][j];products['inverse_identity']+=1
     assert q==int(i==j),(ix,i,j,'BK')
    for j,c in enumerate(rest):
     q=rr[i].get(c,F(0))
     for t in range(3):
      if B[i][t] and G[t][j]:q+=B[i][t]*G[t][j];products['graph_identity']+=1
     assert q==0,(ix,i,j,'BG+C')
    q=F(0)
    for t in range(3):
     if B[i][t] and k[t]:q+=B[i][t]*k[t];products['graph_RHS_identity']+=1
    assert q==h[i],(ix,i,'Bk=h')
   flags[ix]=1;PROGRESS['completed_blocks']=ix+1;knnz+=sum(bool(x) for r in K for x in r);gnnz+=sum(bool(x) for r in G for x in r)
   for x in [x for M in [K,G] for r in M for x in r]+k:maxbits=max(maxbits,x.numerator.bit_length(),x.denominator.bit_length())
   blocks.append((K,rr,h))
   for j,c in enumerate(cols):colmap[c]=(ix,j)
 assert len(blocks)==1365 and flags.all();np.save(out/'block_identities_verified.npy',flags);stats={}
 for role in ['inequality','equality']:
  PROGRESS['stage']=role
  def source():
   if role=='inequality':
    with (src/'rational_inequalities.jsonl').open() as f:
     for line in f:yield source_row(line)
   else:
    for i in eqkeep:yield E[i]
  filename='rational_inequalities.jsonl' if role=='inequality' else 'rational_equalities.jsonl';size=265967 if role=='inequality' else 1367
  rowflags=np.zeros(size,dtype=np.uint8);FLAGS[role+'_reverse_verified.npy']=rowflags;nnz=constant=total=0
  with (new/filename).open() as f:
   for count,(saved,original) in enumerate(itertools.zip_longest(f,source())):
    if count%16==0:limit()
    assert saved is not None and original is not None and count<size
    index,t,h=json.loads(saved);i,terms,rhs=original;assert index==i;restored={};previous=-1
    for j,num,den in t:
     assert type(j) is int and previous<j<len(keep);previous=j;v=frac([num,den]);assert v;restored[keep[j]]=v;maxbits=max(maxbits,v.numerator.bit_length(),v.denominator.bit_length())
    nnz+=len(restored);constant+=not bool(restored);value=frac(h);maxbits=max(maxbits,value.numerator.bit_length(),value.denominator.bit_length())
    lam={}
    for c,v in terms.items():
     if c not in colmap:continue
     bid,j=colmap[c];K=blocks[bid][0]
     for t in range(3):
      if K[j][t]:lam[(bid,t)]=lam.get((bid,t),F(0))+v*K[j][t];products['reverse_multiplier']+=1
    for (bid,t),v in lam.items():
     if not v:continue
     _,rr,hh=blocks[bid]
     for c,x in rr[t].items():restored[c]=restored.get(c,F(0))+v*x;products['reverse_coefficients']+=1
     value+=v*hh[t];products['reverse_RHS']+=1
    restored={c:v for c,v in restored.items() if v};assert restored==terms,(role,i,'coefficients');assert value==rhs,(role,i,'RHS');rowflags[count]=1;total+=1;PROGRESS['completed_'+role+'_rows']=total
  assert total==size and rowflags.all();stats[role]=dict(rows=size,nonzeros=nnz,constant_rows=constant);np.save(out/(role+'_reverse_verified.npy'),rowflags)
 q=read(new/'preparation.json')['primary'];assert q['statistics']==stats and q['inverse_nonzeros']==knnz and q['graph_coefficient_nonzeros']==gnnz and q['largest_fraction_bits']==maxbits
 assert q['exact_three_by_three_inversions']==q['exact_block_coefficient_solves']==1365 and q['eliminated_rootxy_f_columns']==4095 and q['reduced_columns']==10541 and q['conditions_discarded']==0
 np.save(out/'original_retained_columns_verified.npy',maps['original_columns']);limit();PROGRESS['stage']='complete'
 return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,preparation_only=True,primary=q,independent_products=products,exact_copy_error=0,elapsed_s=time.monotonic()-START,producer_import=False,producer_elimination_replayed=False,producer_operation_counts_independently_replayed=False,rank_proof='ALL1365exactB*K=I; uniqueness overREAL follows for squareB. ALLB*G+C=0 andB*k=h exact.',numeric_optimizer_calls=0,numerical_linear_system_solves=0,exact_block_inverses_constructed_by_audit=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,primal_vectors_constructed_or_evaluated=0,analytic_dual_vectors_constructed=0,physical_admission=False,execution_recovery=False,old_proofs_replayed=False,peak_memory_measured=False,scope='Exact coupled3x3 moment/gap dimension reduction from586zeroFACE;alloriginalconditions represented. No roundednumericinput,universalfiniteextension,primalwitness,performance oractualpath claim.')
def main():
 out=D/'balance_block_audit_595';out.mkdir(exist_ok=False)
 try:r=run(out)
 except BaseException as e:
  for n,a in FLAGS.items():np.save(out/n,a)
  failure=dict(passed=False,exception=type(e).__name__,message=str(e),elapsed_s=time.monotonic()-START,progress=PROGRESS,partial_files_are_not_certificates=True,terminal=True,restart_allowed=False,physical_admission=False)
  (out/'terminal_failure.json').write_text(json.dumps(failure,indent=2));print(json.dumps(failure,indent=2),flush=True);raise
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

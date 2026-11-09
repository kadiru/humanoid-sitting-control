"""Exact multirow command implications over immutable projected numeric LPs.

No LP solve, old grouping/projection proof, model call, or candidate construction.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(scope='NEW exact multirow implication reduction of the saved522 projected NUMERIC systems, using a bidirected command forest and retained unary command rows. Preparation only; no solver/trajectory wrapper or candidate.',
 source='All saved522 primary/witness rows, costs, bounds and zero equality systems; inherited523 projection and517 original implication provenance. No532 state or unfinished K vector, old grouping/projection/scaling/factor/solve is input or repeated.',
 graph='Only rows containing exactly two command columns of the SAME joint, opposite strictly nonzero signs, and optionally epigraph/eta qualify. Normalize by the positive command coefficient to x_v-g*x_u+e*z<=b with g>0. All other rows remain explicit. A row with three commands, different joints, same-sign commands or any other feature is never treated as an edge.',
 forest='Use unordered pairs possessing BOTH directed edges. Sort pairs by absolute source-state-index distance, then column IDs; Kruskal constructs a deterministic spanning forest. This ordering is combinatorial, not a coefficient/violation threshold. Keep ALL rows on every forest pair. For each direction use its smallest original row ID. Keep all unary rows. Nonforest edge candidates use the unique oriented forest path only, at least two edges; paths over128 edges stay explicit as an arithmetic resource bound.',
 derivation='Compose positive-gain edges exactly: x_t-G*x_s+E*z<=B. Every path multiplier is strictly positive (a product of later gains divided by positive source coefficient). To imply target x_t-g*x_s+e*z<=b, if G-g is nonzero, add the smallest-ID retained unary source row with its x_s sign matching G-g, normalized by abs(coefficient), multiplied by abs(G-g). This cancels the remaining command coefficient exactly. The residual involves ONLY z. Check B_total+(e-E_total)*z<=b at BOTH unchanged finite z bounds using exact Fractions. This is a nonnegative conic combination plus direct bound implication, not positive-ray grouping or deletion based on a numerical residual.',
 equivalence='Every removed row follows from explicitly retained forest/unary rows and the unchanged z bounds. No removed row is used by any proof, so no circular implication. Subset inclusion gives the converse. This proves exact-real equivalence over the stored binary64 projected numeric matrix, not equivalence to the earlier unrounded522 projection or actual mechanics.',
 exact='All coefficients and RHS are exact binary64 dyadic Fractions; all gain products, coefficient cancellations and endpoint comparisons exact. No coefficient threshold, sign rounding, fitted weights, additional guard, constraint relaxation or returned-vector repair. Stored kept CSR coefficients/RHS and objective/bounds/equalities copied unchanged.',
 coverage='Every original/inward criterion remains explicit, proved here, or covered by inherited517/523 provenance. Both signs/frozen units of consistency remain mandatory. Command variables are NOT eliminated and are NOT assumed disjoint like torque variables. All coupled rate/window/three-command/clock rows not proved remain explicit.',
 future='A future unmodified retained vector must pass the FULL522 projected numeric system at unchanged1e-7, exact21840-auxiliary source endpoint/midpoint backsubstitution, all FULLORIGINAL490 rows/equalities/bounds1e-7, primary-derived actual witness ceiling, exact intervaleta>0 and immutable490 finite encoding. A small reduced residual does not transfer through conic multipliers. All actual unit/geometry/filter/runtime/physical/PPO gates remain.',
 limitations='Saved reference systems only. Witness old illustrative ceiling is not a solved result. Changed state/trust/actual ceiling needs valid new full assembly/provenance and proofs. This is not a new LP algorithm, performance result or feasibility witness. No small arithmetic example/old synthetic test is run.',
 max_path_edges=128,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_candidate=False,physical_admission=False)

def classify(A,ret,extra):
 labels={j:(int(extra[c-10540]//75),int(extra[c-10540]%75-39)) for j,c in enumerate(ret) if 10540<=c<78790 and 39<=extra[c-10540]%75<55}
 edges={};unary={};last=A.shape[1]-1
 for i in range(A.shape[0]):
  p,q=A.indptr[i:i+2];cs=A.indices[p:q];vs=A.data[p:q];assert len(set(map(int,cs)))==len(cs)
  non=[(int(c),float(v)) for c,v in zip(cs,vs) if c!=last and v!=0]
  z=next((float(v) for c,v in zip(cs,vs) if c==last),0.)
  if len(non)==1 and non[0][0] in labels:
   c,v=non[0];unary.setdefault((c,1 if v>0 else -1),i)
  elif len(non)==2 and all(c in labels for c,v in non):
   (c1,a1),(c2,a2)=non
   if ((a1<0<a2) or (a2<0<a1)) and labels[c1][1]==labels[c2][1]:
    u,an,v,ap=(c1,a1,c2,a2) if a1<0 else (c2,a2,c1,a1)
    edges[i]=(u,v,an,ap,z)
 return labels,edges,unary

def construct(A,arr,ret,extra,out):
 labels,edges,unary=classify(A,ret,extra);direct={}
 for i,(u,v,*_) in edges.items():direct.setdefault((u,v),i)
 pairs=sorted({tuple(sorted((u,v))) for u,v in direct if (v,u) in direct},key=lambda p:(abs(labels[p[0]][0]-labels[p[1]][0]),p))
 parent={c:c for c in labels}
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 tree=[];adj={c:[] for c in labels}
 for u,v in pairs:
  a,b=find(u),find(v)
  if a!=b:
   parent[b]=a;tree.append((u,v));adj[u].append(v);adj[v].append(u)
 tree_set=set(tree);par={};depth={};component={}
 for root in sorted(labels):
  if root in par:continue
  par[root]=root;depth[root]=0;component[root]=root;stack=[root]
  while stack:
   u=stack.pop()
   for v in sorted(adj[u],reverse=True):
    if v==par[u]:continue
    assert v not in par;par[v]=u;depth[v]=depth[u]+1;component[v]=root;stack.append(v)
 def path(u,v):
  if component[u]!=component[v]:return None
  left=[];right=[];a,b=u,v
  while a!=b:
   if depth[a]>=depth[b]:left.append((a,par[a]));a=par[a]
   else:right.append((par[b],b));b=par[b]
   if len(left)+len(right)>SPEC['max_path_edges']:return []
  return [direct[x] for x in left+right[::-1]]
 rhs=arr['rhs'];cache={}
 def norm(i):
  if i not in cache:
   u,v,an,ap,z=edges[i];a=F(ap);cache[i]=(-F(an)/a,F(z)/a,F(float(rhs[i]))/a)
  return cache[i]
 records=[];removed=[];paths=[];ptr=[0];up=[];endpoint=[]
 lb,ub=map(lambda x:F(float(x)),arr['bounds'][-1]);assert lb<=ub
 for i,(u,v,*_) in edges.items():
  if tuple(sorted((u,v))) in tree_set:continue
  p=path(u,v);status='disconnected' if p is None else 'path_limit' if not p else 'tested'
  if status!='tested':records.append(dict(row=i,status=status,path_edges=0));continue
  assert 2<=len(p)<=SPEC['max_path_edges']
  G=F(1);E=F(0);B=F(0)
  for j in p:
   g,e,b=norm(j);G=g*G;E=g*E+e;B=g*B+b
  g,e,b=norm(i);delta=G-g;un=-1
  if delta:
   un=unary.get((u,1 if delta>0 else -1),-1)
   if un<0:records.append(dict(row=i,status='no_unary',path_edges=len(p)));continue
   p0,p1=A.indptr[un:un+2];rd={int(c):F(float(x)) for c,x in zip(A.indices[p0:p1],A.data[p0:p1]) if x!=0};a=abs(rd[u]);w=abs(delta)
   E+=w*rd.get(A.shape[1]-1,F(0))/a;B+=w*F(float(rhs[un]))/a
  slack0=b-B-(e-E)*lb;slack1=b-B-(e-E)*ub;passed=slack0>=0 and slack1>=0
  records.append(dict(row=i,status='implied' if passed else 'not_implied',path_edges=len(p),unary=un,endpoint_slack=[str(slack0),str(slack1)]))
  if passed:removed.append(i);paths.extend(p);ptr.append(len(paths));up.append(un);endpoint.append([str(slack0),str(slack1)])
 keep=np.setdiff1d(np.arange(A.shape[0]),np.asarray(removed,dtype=np.int64));K=A[keep].tocsr();sparse.save_npz(out/'matrix.npz',K)
 np.savez_compressed(out/'arrays.npz',rhs=rhs[keep],cost=arr['cost'],bounds=arr['bounds'],equality_rhs=arr['equality_rhs'])
 np.savez_compressed(out/'proof_paths.npz',kept_rows=keep,removed_rows=np.asarray(removed,dtype=np.int64),path_ptr=np.asarray(ptr,dtype=np.int64),path_rows=np.asarray(paths,dtype=np.int64),unary_rows=np.asarray(up,dtype=np.int64),forest_pairs=np.asarray(tree,dtype=np.int64).reshape(-1,2),edge_rows=np.asarray(list(edges),dtype=np.int64))
 (out/'decisions.json').write_text(json.dumps(records,separators=(',',':')))
 (out/'exact_endpoint_slacks.json').write_text(json.dumps(endpoint,separators=(',',':')))
 return dict(source_shape=list(A.shape),source_nnz=A.nnz,shape=list(K.shape),nnz=K.nnz,command_columns=len(labels),directed_edge_rows=len(edges),bidirected_pairs=len(pairs),forest_pairs=len(tree),forest_components=len(set(component.values())),candidates=len(records),implied_rows=len(removed),path_source_row_uses=len(paths),unary_source_row_uses=sum(i>=0 for i in up),decisions={s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},maximum_proved_path=max(np.diff(ptr),default=0).item() if len(ptr)>1 else 0,lp_columns_unchanged=True)

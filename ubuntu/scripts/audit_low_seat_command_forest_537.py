"""Independent exact conic-combination certificate audit; no production import."""
from pathlib import Path
from fractions import Fraction
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def audit(role,out):
 src=D/'torque_projection_preparation_522'/role;new=D/'command_forest_preparation_536'/role
 A=sparse.load_npz(src/'matrix.npz').tocsr();K=sparse.load_npz(new/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'));b=dict(np.load(new/'arrays.npz'));p=dict(np.load(new/'proof_paths.npz'));ret=np.load(src/'projection_ledger.npz')['retained_columns'];extra=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];n=A.shape[1];z=n-1
 labels={j:divmod(int(extra[c-10540]),75) for j,c in enumerate(ret) if 10540<=c<78790 and 39<=int(extra[c-10540])%75<55}
 cache={}
 def row(i):
  i=int(i)
  if i not in cache:
   l,h=A.indptr[i:i+2];cs=A.indices[l:h];assert len(set(map(int,cs)))==len(cs)
   cache[i]={int(c):Fraction(float(v)) for c,v in zip(cs,A.data[l:h]) if v!=0}
  return cache[i]
 def edge(i):
  r=row(i);v=[c for c in r if c!=z];assert len(v)==2 and all(c in labels for c in v) and labels[v[0]][1]==labels[v[1]][1]
  neg=[c for c in v if r[c]<0];pos=[c for c in v if r[c]>0];assert len(neg)==len(pos)==1
  return neg[0],pos[0]
 edge_ids=[];direct={};unary={}
 for i in range(A.shape[0]):
  l,h=A.indptr[i:i+2];v=[(int(c),float(x)) for c,x in zip(A.indices[l:h],A.data[l:h]) if c!=z and x!=0]
  if len(v)==1 and v[0][0] in labels:unary.setdefault((v[0][0],1 if v[0][1]>0 else -1),i)
  if len(v)!=2 or not all(c in labels for c,x in v) or labels[v[0][0]][1]!=labels[v[1][0]][1]:continue
  if not ((v[0][1]<0<v[1][1]) or (v[1][1]<0<v[0][1])):continue
  u=next(c for c,x in v if x<0);v1=next(c for c,x in v if x>0);edge_ids.append(i);direct.setdefault((u,v1),i)
 assert np.array_equal(p['edge_rows'],edge_ids)
 kept=p['kept_rows'];removed=p['removed_rows'];assert np.array_equal(np.sort(np.r_[kept,removed]),np.arange(A.shape[0]));assert len(np.unique(np.r_[kept,removed]))==A.shape[0]
 keep_set=set(map(int,kept));forest={tuple(map(int,t)) for t in p['forest_pairs']};assert len(forest)==len(p['forest_pairs'])
 components={c:{c} for c in labels};owner={c:c for c in labels}
 for u,v in p['forest_pairs']:
  u,v=int(u),int(v);assert labels[u][1]==labels[v][1] and (u,v) in direct and (v,u) in direct
  x,y=owner[u],owner[v];assert x!=y
  if len(components[x])<len(components[y]):x,y=y,x
  for c in components[y]:owner[c]=x
  components[x].update(components.pop(y))
 for i in edge_ids:
  u,v=edge(i)
  if tuple(sorted((u,v))) in forest:assert i in keep_set
 for i in unary.values():assert i in keep_set
 C=A[kept].tocsr();assert C.shape==K.shape
 for key in ['data','indices','indptr']:assert np.array_equal(getattr(C,key),getattr(K,key)),key
 assert np.array_equal(a['rhs'][kept],b['rhs'])
 for key in ['cost','bounds','equality_rhs']:assert np.array_equal(a[key],b[key]),key
 assert sha(src/'equality.npz')==sha(new/'equality.npz')
 endpoints=read(new/'exact_endpoint_slacks.json');ptr=p['path_ptr'];assert len(ptr)==len(removed)+1 and ptr[0]==0 and ptr[-1]==len(p['path_rows']) and len(endpoints)==len(removed)
 verified=[];coefficient_adds=0;unary_uses=0;lb,ub=[Fraction(float(x)) for x in a['bounds'][-1]]
 for k,ri in enumerate(removed):
  ri=int(ri);u,v=edge(ri);target=row(ri);path=list(map(int,p['path_rows'][ptr[k]:ptr[k+1]]));assert 2<=len(path)<=128 and tuple(sorted((u,v))) not in forest
  at=u
  for j in path:
   s,t=edge(j);assert s==at and j in keep_set and tuple(sorted((s,t))) in forest and direct[(s,t)]==j;at=t
  assert at==v
  total={};rhs=Fraction(0);mult=target[v]
  for j in reversed(path):
   s,t=edge(j);r=row(j);w=mult/r[t];assert w>0
   for c,x in r.items():total[c]=total.get(c,Fraction(0))+w*x;coefficient_adds+=1
   rhs+=w*Fraction(float(a['rhs'][j]));mult=w*(-r[s])
  delta={c:target.get(c,Fraction(0))-total.get(c,Fraction(0)) for c in target.keys()|total.keys()};delta={c:x for c,x in delta.items() if x}
  assert all(c in [u,z] for c in delta)
  un=int(p['unary_rows'][k])
  if u in delta:
   assert un>=0 and un in keep_set and unary[(u,1 if delta[u]>0 else -1)]==un
   rr=row(un);assert set(rr)-{z}=={u};w=delta[u]/rr[u];assert w>0;unary_uses+=1
   for c,x in rr.items():total[c]=total.get(c,Fraction(0))+w*x;coefficient_adds+=1
   rhs+=w*Fraction(float(a['rhs'][un]))
  else:assert un==-1
  difference={c:target.get(c,Fraction(0))-total.get(c,Fraction(0)) for c in target.keys()|total.keys()};assert all(c==z or x==0 for c,x in difference.items())
  q=difference.get(z,Fraction(0));B=Fraction(float(a['rhs'][ri]));sl=[B-rhs-q*lb,B-rhs-q*ub];assert min(sl)>=0
  # Producer stores normalized endpoint slack; compare exactly without sharing its recurrence.
  assert [str(x/target[v]) for x in sl]==endpoints[k]
  verified.append(ri)
 np.save(out/(role+'_verified_rows.npy'),np.asarray(verified,dtype=np.int64))
 decisions=read(new/'decisions.json');assert {r['row'] for r in decisions if r['status']=='implied'}==set(map(int,removed))
 return dict(passed=True,source_rows=A.shape[0],kept_rows=len(kept),proved_removed_rows=len(removed),path_row_uses=len(p['path_rows']),unary_row_uses=unary_uses,exact_coefficient_additions=coefficient_adds,coefficient_copy_error=0,rhs_copy_error=0,all_proof_rows_explicitly_retained=True,cost_bounds_equalities_unchanged=True,new_graph_edge_classification=len(edge_ids),old_proofs_repeated=False,lp_calls=0,model_calls=0,physical_admission=False)
def main():
 assert read(D/'command_forest_preparation_exit_536.json')['actual_exit']==0
 launch=read(D/'command_forest_launch_536.json')
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'command_forest_preparation_536/SHA256.json').items():assert sha(D/'command_forest_preparation_536'/p)==h,p
 o=D/'command_forest_audit_537';o.mkdir(exist_ok=False);systems={r:audit(r,o) for r in ['primary','witness']}
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_only=True,systems=systems,proof='Every omitted row independently reconstructed as a positive conic combination of retained original rows plus exact unchanged z-bound implication. No production code imported; no solver, model, old grouping/projection proof or fixture repeated.',lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physical_admission=False)
 (o/'independent_audit.json').write_text(json.dumps(r,indent=2));(o/'SHA256.json').write_text(json.dumps({str(p.relative_to(o)):sha(p) for p in o.iterdir() if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

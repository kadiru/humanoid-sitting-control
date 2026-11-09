"""Independent saved-array472 assembly, finite-map and decision audit.
No mechanics, root, AD, LP solve or old fixture is called. New actual-state
Jacobians are replayed algebraically; the original440 tangent is not reused.
"""
from pathlib import Path
from fractions import Fraction as F
import json,math
import numpy as np
from scipy import sparse
from audit_low_seat_simplex_v2_461 import replay
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
rat=lambda v:F(int(v[0]),int(v[1]))
qf=lambda v:F.from_float(float(v))
def diff(a,b):
 c=(a-b).tocsr();c.eliminate_zeros();return float(abs(c.data).max()) if c.nnz else 0.

def context():
 V=D/'clock_row_preparation_v2_446';Q=D/'clock_row_preparation_446';P=D/'gap_coordinate_preparation_v3_466'
 row=dict(np.load(V/'row_arrays.npz'));cm=dict(np.load(Q/'coordinate_map.npz'));ref=cm['reference'].ravel()[cm['free']]
 d=dict(row=row,cm=cm,M=sparse.load_npz(V/'all_rows_matrix.npz'),E=sparse.load_npz(Q/'equality_matrix.npz'),ea=dict(np.load(Q/'equality_arrays.npz')),B=sparse.load_npz(P/'coordinate_basis.npz'),C0=sparse.load_npz(P/'original_control_box_matrix.npz'),layout=read(P/'layout.json'),dom=read(D/'exact_gap_chart_preparation_468/layout.json'),mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz')),lo=np.maximum(cm['lower'].ravel()[cm['free']],ref-.02),hi=np.minimum(cm['upper'].ravel()[cm['free']],ref+.02))
 rr=[];cc=[];vv=[];by={};nd=[]
 for j,c in enumerate(d['layout']['columns']):
  if c['kind']=='normal_drop':rr.extend([c['edge'],c['pivot']]);cc.extend([j,j]);vv.extend([1.,-1.]);nd.append(j)
  else:by.setdefault((c['state'],c['kind']),[]).append((j,c))
 d.update(N=sparse.csr_matrix((vv,(rr,cc)),shape=(206,d['B'].shape[1])),groups=by,normal_columns=nd)
 return d

def verify_assembly(P,d):
 a=dict(np.load(P/'input.npz'));meta=read(P/'metadata.json');J=a['J'];dep=np.r_[0,1,2,np.arange(4,16)];ind=np.r_[3,np.arange(16,24)];eqs=np.r_[np.arange(14),47]
 rhs=np.concatenate([np.broadcast_to(np.eye(15)[-1],(len(J),15))[:,:,None],-J[:,eqs][:,:,ind]],axis=2)
 T=np.zeros((len(J),24,10));T[:,dep,:]=np.linalg.solve(J[:,eqs][:,:,dep],rhs);T[:,ind,1:]=np.eye(9)
 G=np.concatenate([T,np.einsum('nij,njk->nik',J,T)],axis=1);G[:,24:38,:]=0;G[:,71,:]=0;G[:,71,0]=1
 assert np.array_equal(G[:,38,:],np.tile(np.r_[0.,0.,np.ones(8)],(len(J),1)))
 GB=sparse.block_diag(list(G),format='csr')@d['B'];row=d['row'];A=(d['M']@GB).multiply((1/row['scale'])[:,None]).tocsr();A.eliminate_zeros();C=d['C0']@GB;E=d['E']@GB;E.eliminate_zeros();assert E.nnz==0
 feature=np.c_[a['X'],a['Y']].ravel();r=(d['M']@feature+row['constant'])/row['scale'];e=d['E']@feature+d['ea']['constant'];z=d['C0']@feature
 assert np.array_equal(r,a['r']) and np.array_equal(e,a['e']) and np.array_equal(z,a['z'])
 keep=row['keep'];soft=row['soft']&keep;hard=keep&~soft;var=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&var]=2.**-20;hp=hard&var;guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20);assert np.array_equal(guard,a['guard'])
 radius=.02*meta['trust'];n=d['B'].shape[1]
 if meta['mode']=='predictor':
  dl=np.maximum(d['lo']-z,-radius);du=np.minimum(d['hi']-z,radius);parts=[sparse.hstack([-A[keep],sparse.csr_matrix(-soft[keep,None].astype(float))])];rhs=[r[keep]-guard[keep]];last=[0.,max(1.,2*max(0.,float(-r[soft].min())))]
 else:
  dl=np.maximum.reduce([d['lo']-z,a['origin']-radius-z,np.full(len(z),-.25*radius)]);du=np.minimum.reduce([d['hi']-z,a['origin']+radius-z,np.full(len(z),.25*radius)])
  parts=[sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))]),sparse.hstack([C,sparse.csr_matrix(-np.ones((len(z),1))*radius)]),sparse.hstack([-C,sparse.csr_matrix(-np.ones((len(z),1))*radius)])];rhs=[r[keep]-guard[keep]+soft[keep]*meta['merit_ceiling'],np.zeros(len(z)),np.zeros(len(z))];last=[0.,.25]
 drops=list(map(rat,meta['drops']));dom=d['dom'];nl=np.array([float(rat(l)-x) for l,x in zip(dom['lower'],drops)]);nh=np.array([float(rat(h)-x) for h,x in zip(dom['upper'],drops)])
 cb=sparse.hstack([C,sparse.csr_matrix((len(z),1))]);nb=sparse.hstack([d['N'],sparse.csr_matrix((206,1))]);matrix=sparse.vstack(parts+[cb,-cb,nb,-nb],format='csr');rhs=np.concatenate(rhs+[du,-dl,nh,-nl])
 assert np.array_equal(rhs,a['rhs']) and np.array_equal(dl,a['full_step_lower']) and np.array_equal(du,a['full_step_upper'])
 assert np.array_equal(nl,a['normal_step_lower']) and np.array_equal(nh,a['normal_step_upper'])
 assert np.array_equal(a['bounds'],np.c_[np.r_[np.full(n,-np.inf),last[0]],np.r_[np.full(n,np.inf),last[1]]])
 assert np.array_equal(a['cost'],np.r_[np.zeros(n),1.]) and np.all(a['equality_rhs']==0)
 eq=sparse.load_npz(P/'equality.npz');assert eq.nnz==0 and eq.shape==(1366,n+1)
 errs=[diff(matrix,sparse.load_npz(P/'matrix.npz')),diff(A,sparse.load_npz(P/'A.npz')),diff(C,sparse.load_npz(P/'C.npz')),diff(E,sparse.load_npz(P/'E.npz'))]
 assert max(errs)==0,errs
 return a,meta,T

def finite_drop_replay(origin,t,dom):
 lo=list(map(rat,dom['lower']));hi=list(map(rat,dom['upper']));total=rat(dom['total']);p=dom['pivot'];ids=dom['independent'];v=[F(0)]*206
 for i,x in zip(ids,t):v[i]=qf(x)
 v[p]=-sum(v);assert sum(origin)==total and all(l<=x<=u for l,x,u in zip(lo,origin,hi))
 ratios=[]
 for i,x in enumerate(v):
  if not x:continue
  room=hi[i]-origin[i] if x>0 else origin[i]-lo[i];assert room>0,'outward gap direction';ratios.append(abs(x)/room)
 alpha=1/(1+max(ratios,default=F(0)));q=[x+alpha*z for x,z in zip(origin,v)]
 if not any(v):return origin.copy()
 z=[F(0)]*206
 def floor(x):
  f=float(x);return qf(math.nextafter(f,-math.inf) if qf(f)>x else f)
 def ceil(x):
  f=float(x);return qf(math.nextafter(f,math.inf) if qf(f)<x else f)
 for i in ids:z[i]=floor(q[i])
 S=sum(z)
 for i in ids:
  if S>=total-hi[p]:break
  c=ceil(q[i]);assert 0<=c-z[i]<=F(1,2**59);S+=c-z[i];z[i]=c
 z[p]=total-S
 assert all(l<=x<=u for l,x,u in zip(lo,z,hi)) and sum(z)==total
 return z

def replay_joint(a,meta,t,d):
 target=a['target'].copy();radius=.02*meta['trust'];lo=np.maximum(d['lo'],a['origin']-radius);hi=np.minimum(d['hi'],a['origin']+radius)
 if meta['mode']=='correction':lo=np.maximum(lo,a['z']-.25*radius);hi=np.minimum(hi,a['z']+.25*radius)
 pos={int(v):i for i,v in enumerate(d['cm']['free'])};sumworst=0.
 for (state,kind),entries in d['groups'].items():
  tangent=t[[j for j,c in entries]]
  if kind=='gap':target[state,0]=float(qf(target[state,0])+qf(tangent[0]));continue
  if kind=='orientation':slots=[1];pivot=-1
  else:
   p=entries[0][1]['pivot'];slots=sorted([2+c['index'] for j,c in entries]+[2+p]);pivot=slots.index(2+p)
  ix=[pos[state*10+s] for s in slots];before=target[state,slots].copy();out,alpha,_=replay(before,tangent,lo[ix],hi[ix],pivot)
  if pivot>=0:
   err=abs(sum(map(qf,out))-sum(map(qf,before)));assert err<=len(ix)*F(1,2**54);sumworst=max(sumworst,float(err))
  target[state,slots]=out
 drops=finite_drop_replay(list(map(rat,meta['drops'])),t[d['normal_columns']],d['dom'])
 gap=rat(d['dom']['hover']);gv=[float(gap)]
 for drop in drops:gap-=drop;gv.append(float(gap))
 assert gap==rat(d['dom']['terminal']);target[np.array(d['layout']['normal_states']),0]=gv
 return target,drops,sumworst

def audit_steps(output):
 d=context();O=D/'free_gap_path_472';row=d['row'];hard=row['keep']&~row['soft'];soft=row['keep']&row['soft'];initial=dict(np.load(O/'initial.npz'));result=read(O/'results.json');history=[json.loads(s) for s in (O/'history.jsonl').read_text().splitlines()] if (O/'history.jsonl').exists() else []
 X=initial['X'].copy();Y=initial['Y'].copy();target=initial['target'].copy();drops=read(O/'initial_lift/drop_state.json');accepted=0;trials=0;maps=0;lpcount=0;linearworst=0.;sumworst=0.;details=[];trust=1.
 # New initial complete lift is checked from its stored actual outputs.
 ri=np.load(O/'initial_lift/root_initial.npz');rf=np.load(O/'initial_lift/root_final.npz');assert np.array_equal(rf['X'],X) and np.array_equal(rf['Y'],Y)
 assert np.array_equal(X[:,3],target[:,1]) and np.array_equal(X[:,16:24],target[:,2:10])
 assert drops==d['dom']['initial']
 source=np.load(O/'original440.npz');reference=np.load(D/'gap_coordinate_preparation_v3_466/implicit_map.npz')
 assert np.array_equal(source['X'],reference['referenceX']) and np.array_equal(source['Y'],reference['referenceY'])
 assert np.array_equal(ri['X'],source['X'])
 start_target=reference['target_controls'].copy();gap=rat(d['dom']['hover']);gs=[float(gap)]
 for drop in map(rat,drops):gap-=drop;gs.append(float(gap))
 start_target[np.array(d['layout']['normal_states']),0]=gs
 assert np.array_equal(start_target,target)
 for outer,h in enumerate(history):
  assert h['predictor']==outer and h['trust']==trust
  sr=(d['M']@np.c_[X,Y].ravel()+row['constant'])/row['scale'];merit=max(0.,float(-sr[soft].min()));assert merit==h['start']['soft_merit'];found=False;predicted=None
  for at in h['attempts']:
   k=at['linear_index'];assert k==lpcount;lpcount+=1;P=O/f'linear_{k:02d}';a,meta,T=verify_assembly(P,d);lm=read(P/'solver.json');detail=dict(linear=k,status=lm['status'],candidate=lm['candidate'])
   if lm['candidate']:
    w=np.load(P/'witness.npy');matrix=sparse.load_npz(P/'matrix.npz');v=max(0.,float((matrix@w-a['rhs']).max()),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()));linearworst=max(linearworst,v);assert v==lm['independent_linear_violation'] and (v<=1e-7)==lm['linear_pass']
    if at['correction']==0:predicted=merit-float(w[-1])
   if at.get('map_pass') is False:
    maps+=1;failed=False
    try:replay_joint(a,meta,w[:-1],d)
    except (AssertionError,ValueError):failed=True
    assert failed and not at['accepted'];detail['map_failed']=True
   if (P/'mapped.npz').exists():
    maps+=1;trials+=1;md=np.load(P/'mapped.npz');rt,rd,se=replay_joint(a,meta,w[:-1],d);sumworst=max(sumworst,se)
    assert np.array_equal(rt,md['target']);assert [[v.numerator,v.denominator] for v in rd]==read(P/'mapped.json')['drops']
    start=np.load(P/'root_initial.npz');pred=a['X']+np.einsum('nic,nc->ni',T,rt-a['target']);pred[:,3]=rt[:,1];pred[:,16:24]=rt[:,2:10];pred[d['mesh']['fixed']]=initial['X'][d['mesh']['fixed']];assert np.array_equal(pred,start['X'])
    end=np.load(P/'root_final.npz');tx,ty=end['X'],end['Y'];assert np.array_equal(tx[:,3],rt[:,1]) and np.array_equal(tx[:,16:24],rt[:,2:10]);assert np.array_equal(tx[d['mesh']['fixed']],initial['X'][d['mesh']['fixed']]);assert np.all(tx[:,16:24][d['mesh']['inactive']]==0)
    rooterr=float(abs(np.c_[ty[:,:14],ty[:,47]-rt[:,0]]).max());assert (rooterr<=1e-13)==at['root_pass']
    if (P/'nonlinear_rows.npz').exists():
     f=np.c_[tx,ty].ravel();r=(d['M']@f+row['constant'])/row['scale'];e=d['E']@f+d['ea']['constant'];z=d['C0']@f;saved=np.load(P/'nonlinear_rows.npz');assert np.array_equal(r,saved['rows']) and np.array_equal(e,saved['equalities'])
     valid=bool(np.all(r[hard]>=0) and min((z-d['lo']).min(),(d['hi']-z).min())>=0 and np.max(abs(e[:-1]))<=1e-13 and abs(e[-1])<=1e-8 and np.all(tx[:,16:24]>=0) and rooterr<=1e-13)
     actual=merit-max(0.,float(-r[soft].min()));ratio=actual/predicted;take=bool(valid and actual>0 and ratio>=.1);assert take==at['accepted'];assert actual==at['actual_reduction'] and ratio==at['ratio']
     detail.update(hard_pass=bool(np.all(r[hard]>=0)),normal_maximum=float(abs(e[:-1]).max()),hover=float(abs(e[-1])),accepted=take)
     if take:
      X,Y,target,drops=tx.copy(),ty.copy(),rt.copy(),[[v.numerator,v.denominator] for v in rd];accepted+=1;found=True
      if ratio>.75:trust=min(1.,trust*2)
    else:assert not at['root_pass']
   details.append(detail)
  assert found==h['accepted']
  if not found:trust*=.5
 final=np.load(O/'final.npz');assert np.array_equal(final['X'],X) and np.array_equal(final['Y'],Y) and np.array_equal(final['target'],target);assert drops==read(O/'final_drop_state.json');assert lpcount==result['linear_programs'] and trials==result['nonlinear_trial_paths'] and accepted==result['accepted_steps'] and maps==result['finite_maps'];assert lpcount<=120 and maps<=120 and len(history)<=24
 feature=np.c_[X,Y].ravel();rr=(d['M']@feature+row['constant'])/row['scale'];ee=d['E']@feature+d['ea']['constant']
 assert np.array_equal(rr,final['rows']) and np.array_equal(ee,final['equalities'])
 out=dict(passed=True,linear_programs=lpcount,linear_witness_worst_violation=linearworst,nonlinear_trial_paths=trials,finite_maps=maps,accepted_steps=accepted,maximum_saved_row_replay=0.,maximum_single_load_map_sum_rounding_error=sumworst,all_actual_state_assemblies_rebuilt=True,all_finite_joint_maps_independently_replayed=True,normal_drop_sum_exact=True,full_dependent_f_and_pivot_bounds_present=True,final_selection_exact=True,details=details,optimizer_calls=0,root_calls=0,model_calls=0,physical_admission=False)
 (output/'step_audit.json').write_text(json.dumps(out,indent=2));return out

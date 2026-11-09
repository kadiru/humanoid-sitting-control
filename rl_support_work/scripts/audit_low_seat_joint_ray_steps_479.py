"""Independent NEW478 assembly, shared affine-ray map and decision audit.
No mechanics, root, AD, LP solve or old fixture is called. New actual-state
Jacobians are replayed algebraically; the original440 tangent is not reused.
"""
from pathlib import Path
from fractions import Fraction as F
import json,math
import numpy as np
from scipy import sparse

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

def exact_product(A,t):
    ts=list(map(qf,t));out=[]
    for i in range(A.shape[0]):
        a,b=A.indptr[i:i+2];out.append(sum((qf(v)*ts[j] for v,j in zip(A.data[a:b],A.indices[a:b])),F(0)))
    return out
def replay_affine(a,t,B,C,cm,dom,layout,mode,trust):
    dr=exact_product(B,t);planned=exact_product(C,t);target=a['target'];radius=.02*trust
    ref=cm['reference'].ravel()[cm['free']];lo=np.maximum(cm['lower'].ravel()[cm['free']],ref-.02);hi=np.minimum(cm['upper'].ravel()[cm['free']],ref+.02)
    lo=np.maximum(lo,a['origin']-radius);hi=np.minimum(hi,a['origin']+radius)
    if mode=='correction':lo=np.maximum(lo,a['z']-.25*radius);hi=np.minimum(hi,a['z']+.25*radius)
    low=F(0);high=F(1)
    def restrict(l,u,v):
        nonlocal low,high
        assert l<=u
        if not v:assert l<=0<=u;return
        x,y=sorted([l/v,u/v]);low=max(low,x);high=min(high,y)
    for j,index in enumerate(cm['free']):
        b,s=divmod(int(index),10)
        if s:restrict(qf(lo[j])-qf(target[b,s]),qf(hi[j])-qf(target[b,s]),dr[b*10+s])
        restrict(qf(a['full_step_lower'][j]),qf(a['full_step_upper'][j]),planned[j])
    dl=list(map(rat,dom['lower']));du=list(map(rat,dom['upper']));dv=[F(0)]*206
    for j,c in enumerate(layout['columns']):
        if c['kind']=='normal_drop':dv[c['edge']]+=qf(t[j]);dv[c['pivot']]-=qf(t[j])
    for l,u,x,v in zip(dl,du,a['drops'],dv):restrict(l-x,u-x,v)
    assert 0<=low<=high<=1
    if np.any(t) and high==0:raise ValueError("Only zero shared ray")
    out=target.copy()
    for i,v in enumerate(dr):
        b,s=divmod(i,10);out[b,s]=float(qf(target[b,s])+high*v)
    q=[x+high*v for x,v in zip(a['drops'],dv)];total=rat(dom['total']);pivot=dom['pivot'];ids=dom['independent'];z=[F(0)]*206
    def floor(x):
        f=float(x);return qf(math.nextafter(f,-math.inf) if qf(f)>x else f)
    def ceil(x):
        f=float(x);return qf(math.nextafter(f,math.inf) if qf(f)<x else f)
    assert sum(q)==total
    if not any(dv):z=a['drops'].copy()
    else:
        for i in ids:z[i]=floor(q[i])
        S=sum(z)
        for i in ids:
            if S>=total-du[pivot]:break
            c=ceil(q[i]);assert 0<=c-z[i]<=F(1,2**59);S+=c-z[i];z[i]=c
        z[pivot]=total-S
    assert sum(z)==total and all(l<=v<=u for l,v,u in zip(dl,z,du))
    value=rat(dom['hover']);gg=[float(value)]
    for x in z:value-=x;gg.append(float(value))
    assert value==rat(dom['terminal']);out[np.array(layout['normal_states']),0]=gg
    worst=F(0)
    for b in range(len(out)):
        before=sum(map(qf,target[b,2:10]),F(0));after=sum(map(qf,out[b,2:10]),F(0))
        assert abs(before-after)<=8*F(1,2**54);worst=max(worst,abs(before-after))
    for j,index in enumerate(cm['free']):
        b,s=divmod(int(index),10)
        if s:assert qf(lo[j])<=qf(out[b,s])<=qf(hi[j])
    return out,z,low,high,worst

def replay_joint(a,meta,t,d):
 aa=dict(a);aa['drops']=list(map(rat,meta['drops']))
 # Independent sparse incidence expansion, exact ray intersection and direct
 # floor/ceiling reconstruction; production476 decoder is never called.
 out,z,l,h,se=replay_affine(aa,t,d['B'].tocsr(),d['_actualC'],d['cm'],d['dom'],d['layout'],meta['mode'],meta['trust'])
 return out,z,float(se),h

def audit_steps(output):
 d=context();O=D/'joint_ray_path_478';row=d['row'];hard=row['keep']&~row['soft'];soft=row['keep']&row['soft'];initial=dict(np.load(O/'initial.npz'));result=read(O/'results.json');history=[json.loads(s) for s in (O/'history.jsonl').read_text().splitlines()] if (O/'history.jsonl').exists() else []
 X=initial['X'].copy();Y=initial['Y'].copy();target=initial['target'].copy();drops=read(O/'initial_drop_state.json');accepted=0;trials=0;maps=0;lpcount=0;linearworst=0.;sumworst=0.;details=[];trust=1.
 # Exact inherited INITIAL is checked against its sealed source and audit.
 identity=read(O/'seed_identity.json');assert not identity['initial_root_repeated'] and not identity['rejected_trial_used']
 old=np.load(D/'free_gap_path_472/initial.npz');assert all(np.array_equal(initial[k],old[k]) for k in ['X','Y','J','target','rows','equalities'])
 assert drops==read(D/'free_gap_path_472/initial_lift/drop_state.json')
 assert read(D/'free_gap_path_472/results.json')['accepted_steps']==0
 assert read(D/'free_gap_path_audit_473/independent_audit.json')['passed']
 for outer,h in enumerate(history):
  assert h['predictor']==outer and h['trust']==trust
  sr=(d['M']@np.c_[X,Y].ravel()+row['constant'])/row['scale'];merit=max(0.,float(-sr[soft].min()));assert merit==h['start']['soft_merit'];found=False;predicted=None
  for at in h['attempts']:
   k=at['linear_index'];assert k==lpcount;lpcount+=1;P=O/f'linear_{k:02d}';a,meta,T=verify_assembly(P,d);d['_actualC']=sparse.load_npz(P/'C.npz').tocsr();lm=read(P/'solver.json');detail=dict(linear=k,status=lm['status'],candidate=lm['candidate'])
   if lm['candidate']:
    w=np.load(P/'witness.npy');matrix=sparse.load_npz(P/'matrix.npz');v=max(0.,float((matrix@w-a['rhs']).max()),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()));linearworst=max(linearworst,v);assert v==lm['independent_linear_violation'] and (v<=1e-7)==lm['linear_pass']
    if at['correction']==0:predicted=merit-float(w[-1])
   if at.get('map_pass') is False:
    maps+=1;failed=False
    try:replay_joint(a,meta,w[:-1],d)
    except (AssertionError,ValueError):failed=True
    assert failed and not at['accepted'];detail['map_failed']=True
   if (P/'mapped.npz').exists():
    maps+=1;trials+=1;md=np.load(P/'mapped.npz');rt,rd,se,alpha=replay_joint(a,meta,w[:-1],d);sumworst=max(sumworst,se)
    assert alpha==rat(read(P/'mapped.json')['alpha']);assert np.array_equal(rt,md['target']);assert [[v.numerator,v.denominator] for v in rd]==read(P/'mapped.json')['drops']
    start=np.load(P/'root_initial.npz');pred=a['X']+np.einsum('nic,nc->ni',T,rt-a['target']);pred[:,3]=rt[:,1];pred[:,16:24]=rt[:,2:10];pred[d['mesh']['fixed']]=initial['X'][d['mesh']['fixed']];assert np.array_equal(pred,start['X'])
    end=np.load(P/'root_final.npz');tx,ty=end['X'],end['Y'];assert np.array_equal(tx[:,3],rt[:,1]) and np.array_equal(tx[:,16:24],rt[:,2:10]);assert np.array_equal(tx[d['mesh']['fixed']],initial['X'][d['mesh']['fixed']]);assert np.all(tx[:,16:24][d['mesh']['inactive']]==0)
    rooterr=float(abs(np.c_[ty[:,:14],ty[:,47]-rt[:,0]]).max());assert (rooterr<=1e-13)==at['root_pass']
    if (P/'nonlinear_rows.npz').exists():
     f=np.c_[tx,ty].ravel();r=(d['M']@f+row['constant'])/row['scale'];e=d['E']@f+d['ea']['constant'];z=d['C0']@f;saved=np.load(P/'nonlinear_rows.npz');assert np.array_equal(r,saved['rows']) and np.array_equal(e,saved['equalities'])
     valid=bool(np.all(r[hard]>=0) and min((z-d['lo']).min(),(d['hi']-z).min())>=0 and np.max(abs(e[:-1]))<=1e-13 and abs(e[-1])<=1e-8 and np.all(tx[:,16:24]>=0) and rooterr<=1e-13)
     if at['correction']==0:
      modeled=a['r']+sparse.load_npz(P/'A.npz')@(float(alpha)*w[:-1]);predicted=merit-max(0.,float(-modeled[soft].min()))
      assert predicted==at['shared_ray_predicted_reduction'] and float(alpha)==at['shared_ray_alpha']
     actual=merit-max(0.,float(-r[soft].min()));ratio=actual/predicted if predicted>0 else float('-inf');take=bool(valid and actual>0 and ratio>=.1);assert take==at['accepted'];assert actual==at['actual_reduction'] and ratio==at['ratio']
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
 out=dict(passed=True,assembled_subproblems=lpcount,actual_lp_calls=sum(int(x['status']!=-10) for x in details),linear_programs=lpcount,linear_witness_worst_violation=linearworst,nonlinear_trial_paths=trials,finite_maps=maps,accepted_steps=accepted,maximum_saved_row_replay=0.,maximum_single_load_map_sum_rounding_error=sumworst,all_actual_state_assemblies_rebuilt=True,all_finite_joint_maps_independently_replayed=True,normal_drop_sum_exact=True,full_dependent_f_and_pivot_bounds_present=True,final_selection_exact=True,details=details,optimizer_calls=0,root_calls=0,model_calls=0,physical_admission=False)
 (output/'step_audit.json').write_text(json.dumps(out,indent=2));return out

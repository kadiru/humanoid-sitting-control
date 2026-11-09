"""Independent NEW484 two-stage assembly/certificate/map/decision audit.
Saved actual Jacobians only; no model/root/optimizer or old fixture is run.
"""
from audit_low_seat_joint_ray_steps_479 import context,verify_assembly,exact_product,replay_joint,read,rat,qf,diff,D
from fractions import Fraction as F
import json,math
import numpy as np
from scipy import sparse

def interior(P,a,meta,d):
    Q=P/'interior';saved=dict(np.load(Q/'arrays.npz'));m=read(Q/'intervals.json');C=sparse.load_npz(P/'base/C.npz').tocsr();A=sparse.load_npz(P/'base/A.npz').tocsr();B=d['B'].tocsr();ids=[int(v) for v in d['cm']['free'] if int(v)%10];pos={int(v):i for i,v in enumerate(d['cm']['free'])};H=sparse.vstack([C,B[ids],d['N']],format='csr')
    radius=.02*meta['trust'];lo=np.maximum(d['lo'],a['origin']-radius);hi=np.minimum(d['hi'],a['origin']+radius)
    if meta['mode']=='correction':lo=np.maximum(lo,a['z']-.25*radius);hi=np.minimum(hi,a['z']+.25*radius)
    l=list(map(qf,a['full_step_lower']));u=list(map(qf,a['full_step_upper']));cl=l.copy();cu=u.copy();owners=[['full_linear_control',i] for i in range(len(l))]
    for v in ids:
        b,s=divmod(v,10);i=pos[v];c=qf(a['target'][b,s]);l.append(qf(math.nextafter(lo[i],math.inf))-c);u.append(qf(math.nextafter(hi[i],-math.inf))-c);cl.append(qf(lo[i])-c);cu.append(qf(hi[i])-c);owners.append(['target',b,s])
    dom=d['dom'];drops=list(map(rat,meta['drops']));errors=[F(1,2**59)]*206;errors[dom['pivot']]*=205
    for i,(low,high,x,error) in enumerate(zip(map(rat,dom['lower']),map(rat,dom['upper']),drops,errors)):
        l.append(low-x+error);u.append(high-x-error);cl.append(low-x);cu.append(high-x);owners.append(['drop',i])
    widths=[b-a for a,b in zip(l,u)];assert all(w>0 for w in widths)
    for key,val in [('lower',l),('upper',u),('widths',widths),('closed_lower',cl),('closed_upper',cu),('drop_encoding_errors',errors)]:assert list(map(rat,m[key]))==val,key
    assert owners==m['owners'] and m['target_rows']==len(ids) and diff(H,sparse.load_npz(Q/'interval_matrix.npz'))==0
    keep=d['row']['keep'];soft=d['row']['soft']&keep;n=B.shape[1]
    top=sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))]);closed=sparse.hstack([H,sparse.csr_matrix((H.shape[0],1))]);HN=H.multiply((1/np.array([float(v) for v in widths]))[:,None]).tocsr();ones=sparse.csr_matrix(np.ones((H.shape[0],1)))
    matrix=sparse.vstack([top,closed,-closed,sparse.hstack([HN,ones]),sparse.hstack([-HN,ones])],format='csr')
    rhs=np.r_[a['r'][keep]-a['guard'][keep]+soft[keep]*m['soft_ceiling'],list(map(float,cu)),[-float(x) for x in cl],[float(x/w) for x,w in zip(u,widths)],[-float(x/w) for x,w in zip(l,widths)]]
    assert diff(matrix,sparse.load_npz(Q/'matrix.npz'))==0 and np.array_equal(rhs,saved['rhs'])
    bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),.5]]
    assert np.array_equal(bounds,saved['bounds']) and np.array_equal(saved['cost'],np.r_[np.zeros(n),-1.])
    eq=sparse.load_npz(Q/'equality.npz');assert eq.nnz==0 and eq.shape==(1366,n+1) and np.all(saved['equality_rhs']==0)
    return matrix,rhs,bounds,H,l,u,widths,owners,m

def exact_certificate(H,l,u,widths,owners,t):
    vals=exact_product(H.tocsr(),t);eta=F(1,2);owner=None;bad=[]
    for i,(x,low,high,w) in enumerate(zip(vals,l,u,widths)):
        v=min(x-low,high-x)/w
        if v<eta:eta=v;owner=owners[i]
        if v<=0:bad.append(i)
    return eta,owner,bad

def audit_steps(output):
    d=context();O=D/'interior_path_484';row=d['row'];soft=row['keep']&row['soft'];hard=row['keep']&~row['soft'];initial=dict(np.load(O/'initial.npz'));result=read(O/'results.json');history=[json.loads(s) for s in (O/'history.jsonl').read_text().splitlines()]
    old=np.load(D/'free_gap_path_472/initial.npz');assert all(np.array_equal(initial[k],old[k]) for k in ['X','Y','J','target','rows','equalities']);assert not read(O/'seed_identity.json')['initial_root_repeated']
    X=initial['X'].copy();Y=initial['Y'].copy();target=initial['target'].copy();drops=read(O/'initial_drop_state.json');assert drops==read(D/'free_gap_path_472/initial_lift/drop_state.json')
    trust=1.;count=0;actualcalls=0;accepted=0;maps=0;trials=0;certfail=0;mapfail=0;worst=0.;sumworst=0.;details=[];last=None
    def check_lp(P,lm,matrix,rhs,bounds):
        nonlocal worst,actualcalls
        actualcalls+=int(lm['actual_lp_call'])
        if not lm['candidate']:return None
        w=np.load(P/'witness.npy');v=max(0.,float((matrix@w-rhs).max()),float((bounds[:,0]-w).max()),float((w-bounds[:,1]).max()));worst=max(worst,v)
        assert v==lm['independent_linear_violation'] and (np.isfinite(w).all() and v<=1e-7)==lm['linear_pass'];return w
    for outer,h in enumerate(history):
        assert h['predictor']==outer and h['trust']==trust;sr=(d['M']@np.c_[X,Y].ravel()+row['constant'])/row['scale'];mu=max(0.,float(-sr[soft].min()));assert mu==h['start']['soft_merit'];origin=d['C0']@np.c_[X,Y].ravel();found=False;predicted=None;last_tx=None
        primary=h['primary'];k=primary['linear_index'];assert k==count;count+=1;P=O/f'linear_{k:03d}';a,meta,T=verify_assembly(P/'base',d);assert np.array_equal(a['X'],X) and np.array_equal(a['Y'],Y) and meta['mode']=='predictor';assert not (P/'mapped.npz').exists();lm=read(P/'solver.json')
        w=check_lp(P,lm,sparse.load_npz(P/'base/matrix.npz'),a['rhs'],a['bounds']);eligible=w is not None and lm.get('linear_pass',False)
        if eligible:
            theta=max(0.,float(-(a['r']+sparse.load_npz(P/'base/A.npz')@w[:-1])[soft].min()));assert theta==primary['modeled_soft'];eligible=theta<mu;ceiling=(mu+theta)/2
        assert eligible==primary['eligible_for_interior_stage'];details.append(dict(linear=k,role='primary',status=lm['status'],never_mapped=True))
        if not eligible:assert not h['attempts']
        for at in h['attempts']:
            k=at['linear_index'];assert k==count;count+=1;P=O/f'linear_{k:03d}';a,meta,T=verify_assembly(P/'base',d);d['_actualC']=sparse.load_npz(P/'base/C.npz').tocsr();lm=read(P/'solver.json');detail=dict(linear=k,role='interior',status=lm['status']);stage=read(P/'stage.json');assert stage['soft_ceiling']==ceiling==at['soft_ceiling']
            if at['correction']==0:assert np.array_equal(a['X'],X) and np.array_equal(a['Y'],Y)
            else:assert last_tx is not None and np.array_equal(a['X'],last_tx) and np.array_equal(a['origin'],origin) and meta['merit_ceiling']==ceiling
            if not (P/'interior').exists():
                assert lm['status']==-10 and not lm['actual_lp_call'] and not at['accepted']
                # Independently reconstructing an invalid interval must reject.
                dl=a['full_step_lower'];du=a['full_step_upper'];assert np.any(dl>=du) or 'positive representable interior' in lm['message'];details.append(detail);continue
            matrix,rhs,bounds,H,l,u,widths,owners,im=interior(P,a,meta,d);assert im['soft_ceiling']==ceiling
            w=check_lp(P,lm,matrix,rhs,bounds)
            if w is None or not lm.get('linear_pass',False):assert not at['accepted'];details.append(detail);continue
            eta,owner,bad=exact_certificate(H,l,u,widths,owners,w[:-1]);cert=read(P/'certificate.json');assert eta==rat(cert['certified_relative_clearance']) and owner==cert['limiting_owner'] and bad==cert['nonpositive_intervals'];assert (eta>0)==cert['passed']==at['certificate_pass'];assert not cert['witness_modified'];detail['exact_clearance']=float(eta)
            if eta<=0:
                certfail+=1;assert not (P/'mapped.npz').exists() and not at['accepted'];details.append(detail);continue
            if at['correction']==0:
                modeled=max(0.,float(-(a['r']+sparse.load_npz(P/'base/A.npz')@w[:-1])[soft].min()));predicted=mu-modeled;assert modeled==at['certified_ray_modeled_soft'] and predicted==at['predicted_reduction']
                if predicted<=0:assert not (P/'mapped.npz').exists();details.append(detail);continue
            maps+=1
            rt,rd,se,alpha=replay_joint(a,meta,w[:-1],d);assert alpha==1;sumworst=max(sumworst,se)
            md=np.load(P/'mapped.npz');record=read(P/'mapped.json');assert np.array_equal(md['target'],rt) and np.array_equal(md['tangent'],w[:-1]);assert record['drops']==[[x.numerator,x.denominator] for x in rd]
            radius=.02*meta['trust'];lo=np.maximum(d['lo'],a['origin']-radius);hi=np.minimum(d['hi'],a['origin']+radius)
            if meta['mode']=='correction':lo=np.maximum(lo,a['z']-.25*radius);hi=np.minimum(hi,a['z']+.25*radius)
            strict=all(lo[i]<rt[divmod(int(v),10)]<hi[i] for i,v in enumerate(d['cm']['free']) if int(v)%10)
            dropstrict=all(rat(l)<x<rat(u) for l,x,u in zip(d['dom']['lower'],rd,d['dom']['upper']));assert strict and dropstrict and record['alpha_exact_one'] and record['decoded_targets_strictly_interior'] and record['decoded_drops_strictly_interior']
            if at.get('map_pass') is False:mapfail+=1;raise AssertionError('A verified strict alpha1 map cannot have failed its declared contract')
            trials+=1;start=np.load(P/'root_initial.npz');pred=a['X']+np.einsum('nic,nc->ni',T,rt-a['target']);pred[:,3]=rt[:,1];pred[:,16:24]=rt[:,2:10];pred[d['mesh']['fixed']]=initial['X'][d['mesh']['fixed']];assert np.array_equal(pred,start['X'])
            end=np.load(P/'root_final.npz');tx,ty=end['X'],end['Y'];last_tx=tx;assert np.array_equal(tx[:,3],rt[:,1]) and np.array_equal(tx[:,16:24],rt[:,2:10]) and np.array_equal(tx[d['mesh']['fixed']],initial['X'][d['mesh']['fixed']]) and np.all(tx[:,16:24][d['mesh']['inactive']]==0)
            rooterr=float(abs(np.c_[ty[:,:14],ty[:,47]-rt[:,0]]).max());assert (rooterr<=1e-13)==at['root_pass']
            if not at['root_pass']:assert not at['accepted'];details.append(detail);continue
            feature=np.c_[tx,ty].ravel();r=(d['M']@feature+row['constant'])/row['scale'];e=d['E']@feature+d['ea']['constant'];z=d['C0']@feature;saved=np.load(P/'nonlinear_rows.npz');assert np.array_equal(r,saved['rows']) and np.array_equal(e,saved['equalities'])
            valid=bool(np.all(r[hard]>=0) and min((z-d['lo']).min(),(d['hi']-z).min())>=0 and np.max(abs(e[:-1]))<=1e-13 and abs(e[-1])<=1e-8 and np.all(tx[:,16:24]>=0) and rooterr<=1e-13)
            actual=mu-max(0.,float(-r[soft].min()));ratio=actual/predicted;take=bool(valid and actual>0 and ratio>=.1);assert take==at['accepted'] and ratio==at['ratio'] and actual==at['actual_reduction'];detail.update(hard_pass=bool(np.all(r[hard]>=0)),normal_maximum=float(abs(e[:-1]).max()),hover=float(abs(e[-1])),accepted=take)
            if take:
                X,Y,target,drops=tx.copy(),ty.copy(),rt.copy(),[[v.numerator,v.denominator] for v in rd];accepted+=1;found=True;last=k
                if ratio>.75:trust=min(1.,trust*2)
            else:ceiling=mu-.5*predicted
            details.append(detail)
        assert found==h['accepted']
        if not found:trust*=.5
    final=np.load(O/'final.npz');assert np.array_equal(final['X'],X) and np.array_equal(final['Y'],Y) and np.array_equal(final['target'],target) and read(O/'final_drop_state.json')==drops
    for k,v in [('assembled_subproblems',count),('actual_lp_calls',actualcalls),('nonlinear_trial_paths',trials),('finite_maps',maps),('accepted_steps',accepted),('certificate_failures',certfail),('coordinate_map_failures',mapfail),('last_accepted_linear_index',last)]:assert result[k]==v,(k,result[k],v)
    assert count<=144 and maps<=120 and len(history)<=24
    f=np.c_[X,Y].ravel();assert np.array_equal((d['M']@f+row['constant'])/row['scale'],final['rows']) and np.array_equal(d['E']@f+d['ea']['constant'],final['equalities'])
    out=dict(passed=True,assembled_subproblems=count,actual_lp_calls=actualcalls,linear_programs=count,linear_witness_worst_violation=worst,nonlinear_trial_paths=trials,finite_maps=maps,accepted_steps=accepted,certificate_failures=certfail,coordinate_map_failures=mapfail,maximum_saved_row_replay=0.,maximum_single_load_map_sum_rounding_error=sumworst,all_actual_state_assemblies_rebuilt=True,all_certificates_exact=True,all_finite_joint_maps_independently_replayed=True,normal_drop_sum_exact=True,full_dependent_f_and_pivot_bounds_present=True,final_selection_exact=True,details=details,optimizer_calls=0,root_calls=0,model_calls=0,physical_admission=False)
    (output/'step_audit.json').write_text(json.dumps(out,indent=2));return out

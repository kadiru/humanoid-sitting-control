"""Independent saved-array audit of NEW582 actual-state LPs, maps and decisions.
No production assembly/decoder, model, AD, root or optimizer is called.
"""
from audit_low_seat_lifted_feasibility_491 import read,sha,error,mul,floor,ceil,fq,rat,F,D
from pathlib import Path
import numpy as np
from scipy import sparse
import json,math
O=D/'graph_path_582'
OUT=np.r_[np.arange(14),np.arange(15,51)]
VERIFIED_NEW_PROOFS=set()
SOLVER_AUDITS=[]

def context():
    P=D/'lifted_feasibility_preparation_490';z=dict(np.load(P/'reference.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
    dom=read(D/'exact_gap_chart_preparation_468/layout.json');gl=read(D/'gap_coordinate_preparation_v3_466/layout.json')
    return dict(z=z,mesh=mesh,dom=dom,normal=np.asarray(gl['normal_states']),S=sparse.load_npz(P/'feature_basis.npz').tocsr(),N=sparse.load_npz(P/'normal_drop_chain.npz').tocsr(),
        M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz'),E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz'),
        row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz')),eq=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz')))

def interval(d,f,drops,trust):
    z=d['z'];S=d['S'];lo=z['lower'];hi=z['upper'];ids=z['varying'];controls=z['control_feature_rows'];v=z['feature_scale'].ravel();flat=f.ravel();ng=set((d['normal']*75+71).tolist());U=F(1,2**59)
    low=[];high=[];cl=[];cu=[];owners=[]
    for j,row in enumerate(list(controls)+list(ids)):
        if j<len(lo):l=float(lo[j]);u=float(hi[j]);owner=['original_control',j]
        else:l=float(flat[row]-trust*v[row]);u=float(flat[row]+trust*v[row]);owner=['feature_trust',int(row)]
        c=fq(flat[row]);cl.append(fq(l)-c);cu.append(fq(u)-c);owners.append(owner)
        if j>=len(lo) and int(row) in ng:low.append(fq(l)-c+207*U);high.append(fq(u)-c-207*U)
        else:low.append(fq(math.nextafter(l,math.inf))-c);high.append(fq(math.nextafter(u,-math.inf))-c)
    for i,(l,u,c) in enumerate(zip(map(rat,d['dom']['lower']),map(rat,d['dom']['upper']),drops)):
        e=(205 if i==d['dom']['pivot'] else 1)*U;cl.append(l-c);cu.append(u-c);low.append(l-c+e);high.append(u-c-e);owners.append(['drop',i])
    width=[u-l for l,u in zip(low,high)];assert all(x>0 for x in width)
    return sparse.vstack([S[controls],S[ids],d['N']],format='csr'),low,high,width,cl,cu,owners

def summary(d,f,y):
    row=d['row'];soft=row['keep']&row['soft'];hard=row['keep']&~soft;active=d['z']['active'];err=f[:,24:]-y
    ar=(d['M']@f.ravel()+row['constant'])/row['scale'];rr=(d['M']@np.c_[f[:,:24],y].ravel()+row['constant'])/row['scale'];ae=d['E']@f.ravel()+d['eq']['constant'];ee=d['E']@np.c_[f[:,:24],y].ravel()+d['eq']['constant']
    h=(err[active][:,OUT]/d['z']['sigma'][active][:,OUT]).ravel();z=f.ravel()[d['z']['control_feature_rows']];box=float(min((z-d['z']['lower']).min(),(d['z']['upper']-z).min()))
    fixed=np.array_equal(f[d['mesh']['fixed']],d['z']['features'][d['mesh']['fixed']]);inactive=np.all(f[:,16:24][d['mesh']['inactive']]==0);normal=float(abs(ee[:-1]).max());hover=float(abs(ee[-1]));phase=row['groups']=='mesh_phase'
    parts={'feet_m':float(abs(err[:,[0,1,2,6,7,8]]).max()),'orientation_0_1_vee':float(abs(err[:,[3,4,5,9,10,11]]).max()),'moment_BW_m':float(abs(err[:,12:14]).max()),'command_rad':float(abs(err[:,15:31]).max()),'torque_Nm':float(abs(err[:,31:47]).max()),'gap_m':float(abs(err[:,47]).max()),'COM_m':float(abs(err[:,48:51]).max())}
    closure=all(x<=1e-13 for x in parts.values());valid=bool(np.all(ar[hard]>=0) and box>=0 and fixed and inactive and np.all(f[:,16:24]>=0) and normal<=1e-13 and abs(ae[-1])<=1e-8)
    final=bool(np.all(rr>=0) and box>=0 and fixed and inactive and normal<=1e-13 and hover<=1e-8 and np.max(abs(y[:,:14]))<=1e-13 and closure)
    s=dict(phi=max(0.,float(-ar[soft].min()),float(abs(h).max())),auxiliary_soft_merit=max(0.,float(-ar[soft].min())),consistency_scaled_max=float(abs(h).max()),auxiliary_hard_min=float(ar[hard].min()),auxiliary_hard_pass=bool(np.all(ar[hard]>=0)),auxiliary_phase_min=float(ar[phase].min()),actual_all_rows_min=float(rr.min()),actual_hard_min=float(rr[hard].min()),actual_soft_merit=max(0.,float(-rr[soft].min())),actual_phase_min=float(rr[phase].min()),full_control_box_min=box,actual_normal_max=normal,actual_hover_normalized=hover,auxiliary_hover_normalized=float(abs(ae[-1])),actual_foot_moment_max=float(abs(y[:,:14]).max()),consistency_by_unit=parts,consistency_pass=closure,fixed_exact=bool(fixed),inactive_exact=bool(inactive),nonnegative_loads=bool(np.all(f[:,16:24]>=0)),auxiliary_acceptance_domain_pass=valid,actual_numerical_admission_gate=final,physical_admission=False)
    return s,dict(auxiliary_rows=ar,actual_rows=rr,auxiliary_equalities=ae,actual_equalities=ee,consistency=err),h

def matrix(d,f,y,J,H,l,u,width,cl,cu,ceiling):
    S=d['S'];row=d['row'];keep=row['keep'];soft=row['soft']&keep;hard=keep&~soft;active=d['z']['active']
    ss,vals,h=summary(d,f,y);r=vals['auxiliary_rows'];A=(d['M']@S).multiply((1/row['scale'])[:,None]).tocsr()
    blocks=np.zeros((len(active),50,75));blocks[:,:,:24]=-J[active][:,OUT,:]
    for j,k in enumerate(OUT):blocks[:,j,24+k]=1.
    ids=(active[:,None]*75+np.arange(75)).ravel();C=(sparse.block_diag(list(blocks),format='csr')@S[ids]).multiply((1/d['z']['sigma'][active][:,OUT].ravel())[:,None]).tocsr()
    variable=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&variable]=2.**-20;hp=hard&variable;guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20)
    n=S.shape[1];m=C.shape[0];HN=H.multiply((1/np.array([float(w) for w in width]))[:,None]).tocsr();closed=sparse.hstack([HN,sparse.csr_matrix((H.shape[0],1))]);crhs=np.r_[[float(v/w) for v,w in zip(cu,width)],[-float(v/w) for v,w in zip(cl,width)]]
    if ceiling is None:
        top=sparse.hstack([-A[keep],sparse.csr_matrix(-soft[keep,None].astype(float))]);last=sparse.csr_matrix(-np.ones((m,1)));rhs=np.r_[r[keep]-guard[keep],-h,h,crhs];cost=np.r_[np.zeros(n),1.];boundmax=max(1.,2*ss['phi'])
    else:
        top=sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))]);last=sparse.csr_matrix((m,1));rhs=np.r_[r[keep]-guard[keep]+soft[keep]*ceiling,-h+ceiling,h+ceiling,crhs];cost=np.r_[np.zeros(n),-1.];boundmax=.5
    closure=sparse.vstack([sparse.hstack([C,last]),sparse.hstack([-C,last])]);parts=[top,closure,closed,-closed]
    if ceiling is not None:
        one=sparse.csr_matrix(np.ones((H.shape[0],1)));parts += [sparse.hstack([HN,one]),sparse.hstack([-HN,one])];rhs=np.r_[rhs,[float(v/w) for v,w in zip(u,width)],[-float(v/w) for v,w in zip(l,width)]]
    bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),boundmax]]
    return sparse.vstack(parts,format='csr'),rhs,cost,bounds,A,C,r,h,guard

def replay(d,f,drops,t):
    delta=mul(d['S'],t);q=[fq(x)+v for x,v in zip(f.ravel(),delta)];out=np.asarray(list(map(float,q))).reshape(f.shape);dd=mul(d['N'],t);raw=[c+x for c,x in zip(drops,dd)];dom=d['dom'];total=rat(dom['total']);pivot=dom['pivot'];ind=dom['independent'];lo=list(map(rat,dom['lower']));hi=list(map(rat,dom['upper']));assert sum(raw)==total and all(l<=x<=u for l,x,u in zip(lo,raw,hi))
    if any(dd):
        enc=[F(0)]*206
        for i in ind:enc[i]=fq(floor(raw[i]))
        mass=sum(enc);raised=[]
        for i in ind:
            if mass>=total-hi[pivot]:break
            v=fq(ceil(raw[i]));mass+=v-enc[i];enc[i]=v;raised.append(i)
        enc[pivot]=total-mass
    else:enc=drops.copy();raised=[]
    assert sum(enc)==total and all(l<=x<=u for l,x,u in zip(lo,enc,hi));g=[rat(dom['hover'])]
    for x in enc:g.append(g[-1]-x)
    assert g[-1]==rat(dom['terminal']);out[d['normal'],71]=list(map(float,g));out[d['z']['active'],24:38]=0.;worst=F(0)
    for b in d['z']['active']:
        before=sum(map(fq,f[b,16:24]),F(0));after=sum(map(fq,out[b,16:24]),F(0));assert sum(q[b*75+16:b*75+24],F(0))==before;worst=max(worst,abs(after-before));out[b,38]=float(after-1)
    assert worst<=8*F(1,2**54)
    return out,enc,raised,worst

def audit_native(folder,Q,m,rhs,cost,bounds,w,lm):
    T=Q/'lp';assert lm['actual_lp_call']==(T/'scipy_started.json').exists() and lm['method']=='compensated-graph-zero-face-IPM'
    if (T/'result.json').exists():
        cr=read(T/'result.json')
        for k in ['status','message','iterations','elapsed_s']:assert lm[k]==cr[k]
        from audit_low_seat_graph_path_solver_583 import audit_solver
        SOLVER_AUDITS.append(dict(folder=str(T),**audit_solver(T)))
    if w is None:
        assert lm['terminal_failure'] and not (Q/'witness.npy').exists();return None
    assert lm['status']==0 and lm['worker_actual_exit']==0 and not lm['worker_wall_timeout'] and np.array_equal(w,np.load(T/'witness.npy'))
    checks=read(T/'original_checks.json');assert checks['passed'];assert lm['source_linear_violation']==checks['full']['primal'];return checks['full']['primal']

def audit_steps(output):
    d=context();initial=dict(np.load(O/'initial.npz'));f=initial['features'];y=initial['actualY'];J=initial['J'];assert np.array_equal(f,d['z']['features']) and np.array_equal(y,d['z']['actualY']) and np.array_equal(J,d['z']['J'])
    drops=list(map(rat,read(O/'initial_drop_state.json')));assert drops==list(map(rat,read(D/'free_gap_path_472/initial_lift/drop_state.json')))
    history=[json.loads(s) for s in (O/'history.jsonl').read_text().splitlines()];result=read(O/'results.json');trust=1.;lps=0;assemblies=0;maps=0;trials=0;accepted=0;certfail=0;mapfail=0;last=None;worst=0.;roundworst=0.;valuecalls=0;valuerecords=0;jcalls=0;jrecords=0;details=[]
    for k,hist in enumerate(history):
        folder=O/f'major_{k:02d}';saved=dict(np.load(folder/'state.npz'));assert hist['major']==k and hist['trust']==trust and all(np.array_equal(saved[key],val) for key,val in [('features',f),('actualY',y),('J',J)])
        ss,_,_=summary(d,f,y);assert ss==hist['start'];mu=ss['phi']
        if k==0:
            assert read(D/'torque_path_audit_525/independent_audit.json')['passed']
            prior=D/'torque_path_524/major_00'
            for name in ['state.npz','A.npz','C.npz','H.npz','intervals.json']:assert sha(folder/name)==sha(prior/name)
            im=read(folder/'intervals.json');H=sparse.load_npz(folder/'H.npz');l,u,width,cl,cu=[list(map(rat,im[key])) for key in ['lower','upper','widths','closed_lower','closed_upper']];owners=im['owners']
            assert im['trust']==trust and im['drops']==[[x.numerator,x.denominator] for x in drops]
        else:
            H,l,u,width,cl,cu,owners=interval(d,f,drops,trust);im=read(folder/'intervals.json')
            for key,val in [('lower',l),('upper',u),('widths',width),('closed_lower',cl),('closed_upper',cu)]:assert list(map(rat,im[key]))==val
            assert im['owners']==owners and im['drops']==[[x.numerator,x.denominator] for x in drops] and error(H,sparse.load_npz(folder/'H.npz'))==0
        ceiling=None;taken=False;eligible=False
        for role in ['primary','interior']:
            if role=='interior' and not eligible:assert not (folder/role).exists();break
            Q=folder/role
            if k==0 and role=='primary':
                prior=D/'torque_path_524/major_00/primary'
                for name in ['matrix.npz','arrays.npz','equality.npz','stage.json']:assert sha(Q/name)==sha(prior/name)
                a=dict(np.load(Q/'arrays.npz'));m=sparse.load_npz(Q/'matrix.npz');rhs=a['rhs'];cost=a['cost'];bounds=a['bounds'];A=sparse.load_npz(folder/'A.npz');C=sparse.load_npz(folder/'C.npz');r=saved['r'];hh=saved['h'];guard=saved['guard']
            else:
                m,rhs,cost,bounds,A,C,r,hh,guard=matrix(d,f,y,J,H,l,u,width,cl,cu,ceiling);a=dict(np.load(Q/'arrays.npz'))
            lm=read(Q/'solver.json');assemblies+=1;lps+=int(lm['actual_lp_call'])
            assert error(m,sparse.load_npz(Q/'matrix.npz'))==0 and np.array_equal(rhs,a['rhs']) and np.array_equal(cost,a['cost']) and np.array_equal(bounds,a['bounds'])
            eq=sparse.load_npz(Q/'equality.npz');assert eq.shape==(1366,78791) and eq.nnz==0 and np.all(a['equality_rhs']==0)
            assert read(Q/'stage.json')==dict(ceiling=ceiling,primary=role=='primary')
            assert error(A,sparse.load_npz(folder/'A.npz'))==0 and error(C,sparse.load_npz(folder/'C.npz'))==0
            assert np.array_equal(r,saved['r']) and np.array_equal(hh,saved['h']) and np.array_equal(guard,saved['guard'])
            w=None
            if lm['candidate']:
                w=np.load(Q/'witness.npy');v=max(0.,float((m@w-rhs).max()),float((bounds[:,0]-w).max()),float((w-bounds[:,1]).max()));worst=max(worst,v);assert v==lm['linear_violation']
                source_violation=audit_native(folder,Q,m,rhs,cost,bounds,w,lm)
                assert (np.all(np.isfinite(w)) and v<=1e-7 and source_violation<=1e-7)==lm['linear_pass']
            else:audit_native(folder,Q,m,rhs,cost,bounds,None,lm)
            good=w is not None and lm.get('linear_pass',False) and not lm['terminal_failure']
            if role=='primary':
                if good:
                    soft=d['row']['soft']&d['row']['keep'];theta=max(0.,float(-(r+A@w[:-1])[soft].min()),float(abs(hh+C@w[:-1]).max()));assert theta==hist['primary_modeled_phi'];eligible=theta<mu;ceiling=(mu+theta)/2
                assert bool(eligible)==hist['primary_eligible'];assert not (Q/'mapped.npz').exists()
            elif good:
                assert hist['witness_ceiling']==ceiling
                for at in hist['trials']:
                    j=at['dyadic'];T=folder/f'trial_{j}';t=w[:-1]*2.**-j;assert np.array_equal(t,np.load(T/'direction.npy'));vals=mul(H,t);eta=F(1,2);bad=[];owner=None
                    for i,(x,low,high,widthi) in enumerate(zip(vals,l,u,width)):
                        q=min(x-low,high-x)/widthi
                        if q<eta:eta=q;owner=owners[i]
                        if q<=0:bad.append(i)
                    cert=read(T/'certificate.json');assert rat(cert['eta'])==eta and cert['nonpositive']==bad and cert['limiting_owner']==owner and (eta>0)==cert['passed']==at['certificate_pass']
                    if eta<=0:certfail+=1;assert not at['accepted'] and not (T/'mapped.npz').exists();continue
                    maps+=1
                    if not at['map_pass']:
                        mapfail+=1
                        # A map failure is retained as failure; this branch is not certified by hiding it.
                        raise AssertionError('Unexpected finite-contract failure requires explicit new saved-input audit before closure')
                    nf,nd,raised,se=replay(d,f,drops,t);roundworst=max(roundworst,float(se));md=read(T/'mapped.json');assert np.array_equal(nf,np.load(T/'mapped.npz')['features']) and md['drop_state']==[[x.numerator,x.denominator] for x in nd] and md['raised']==raised and rat(md['load_sum_rounding'])==se
                    z=nf.ravel()[d['z']['control_feature_rows']];assert np.all(z>=d['z']['lower']) and np.all(z<=d['z']['upper'])
                    for i,row in enumerate(d['z']['varying']):v=fq(nf.ravel()[row])-fq(f.ravel()[row]);assert cl[len(z)+i]<=v<=cu[len(z)+i]
                    assert np.array_equal(nf[d['mesh']['fixed']],f[d['mesh']['fixed']]) and np.all(nf[:,16:24][d['mesh']['inactive']]==0)
                    actual=dict(np.load(T/'actual.npz'));ny=actual['Y'];changed=np.flatnonzero(np.any(f[:,:24]!=nf[:,:24],axis=1));assert np.array_equal(changed,actual['changed']);mask=np.ones(len(f),bool);mask[changed]=False;assert np.array_equal(ny[mask],y[mask]);trials+=1;valuecalls+=bool(len(changed));valuerecords+=len(changed)
                    ns,vals2,_=summary(d,nf,ny);assert ns==at['summary']
                    for name,v in vals2.items():assert np.array_equal(v,actual[name])
                    delta=nf-f;modelY=y+np.einsum('nyx,nx->ny',J,delta[:,:24]);predphi=summary(d,nf,modelY)[0]['phi'];pred=mu-predphi;ared=mu-ns['phi'];ratio=ared/pred if pred>0 else None;take=bool(ns['auxiliary_acceptance_domain_pass'] and pred>0 and ared>0 and ratio>=.1)
                    assert predphi==at['predicted_phi'] and pred==at['predicted_reduction'] and ared==at['actual_reduction'] and ratio==at['ratio'] and take==at['accepted']
                    details.append(dict(major=k,dyadic=j,eta=float(eta),accepted=take,phi=ns['phi'],actual_gate=ns['actual_numerical_admission_gate'],actual_hard_min=ns['actual_hard_min']))
                    if take:
                        aj=np.load(T/'accepted_J.npz');assert np.array_equal(aj['changed'],changed) and np.array_equal(aj['J'][mask],J[mask]);J=aj['J'];f,y,drops=nf,ny,nd;accepted+=1;last=[k,j];taken=True;jcalls+=bool(len(changed));jrecords+=len(changed)
                        if ratio>.75:trust=min(1.,trust*2)
                        break
            else:assert not hist['trials']
        assert taken==hist['accepted']
        if not taken:trust*=.5
    final=dict(np.load(O/'final.npz'));assert np.array_equal(f,final['features']) and np.array_equal(y,final['Y']) and np.array_equal(J,final['J']) and read(O/'final_drop_state.json')==[[x.numerator,x.denominator] for x in drops]
    ss,vals,_=summary(d,f,y);assert ss==result['final']
    for key,v in vals.items():assert np.array_equal(final[key],v)
    for key,val in [('major_iterations',len(history)),('assembled_subproblems',assemblies),('actual_lp_calls',lps),('finite_maps',maps),('actual_trial_evaluations',trials),('accepted_solver_iterates',accepted),('certificate_failures',certfail),('map_failures',mapfail),('last_accepted',last)]:assert result[key]==val
    assert len(history)<=24 and assemblies<=48 and lps<=48 and maps<=120
    assert result['new_local_block_solves']==result['new_inverse_certificates']==0 and not result['initial_inverse_inherited'] and result['no_mixed_coordinate_transform']
    if result['stop'].startswith('terminal LP failure'):
        last_record=history[-1];assert last_record.get('interior',last_record['primary'])['terminal_failure']
    elif result['stop']=='declared trust floor':assert trust<1/32
    elif result['stop']=='no positive primary modeled improvement; no retry':assert not history[-1]['primary_eligible'] and not history[-1]['primary']['terminal_failure']
    scopes=[read(q) for q in O.glob('major_*/*/lp/input_proof.json')]
    assert result['inherited_native_systems']==sum(x['inherited'] for x in scopes)
    assert result['new_native_systems']==sum(not x['inherited'] for x in scopes)
    assert result['new_quotient_proof_rows']==sum(x.get('new_quotient_proof_rows',0) for x in scopes)
    assert result['new_projection_pairs']==0
    assert result['wrapper_counts']==dict(value_requests=valuecalls,value_state_records=valuerecords,jacobian_requests=jcalls,jacobian_state_records=jrecords,root_updates=0,initial_lifts=0)
    out=dict(passed=True,solver_audits=SOLVER_AUDITS,changed_state_assemblies_replayed=True,initial_assembly_inherited525=True,initial_assembly_repeated=False,exact_certificates_maps_replayed=True,certificate_map_scope='Empty-set truth if finite_maps=0; no successful witness/map implied',all_actual_and_auxiliary_rows_retained=True,final_selection_exact=True,
        native_system_scopes=[{k:v for k,v in x.items() if k not in ['source_row_ids','input_sha256','full_input_sha256']} for x in scopes],initial_native_proofs_inherited_not_repeated=True,all_full_original_rows_certified_for_each_complete_witness=True,no_mixed_transform=True,actual_lp_calls=lps,assembled_subproblems=assemblies,major_iterations=len(history),finite_maps=maps,actual_trial_evaluations=trials,accepted_solver_iterates=accepted,certificate_failures=certfail,map_failures=mapfail,
        linear_violation_maximum=worst,load_sum_rounding_maximum=roundworst,row_replay_error=0.,details=details,model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
    (output/'step_audit.json').write_text(json.dumps(out,indent=2));return out

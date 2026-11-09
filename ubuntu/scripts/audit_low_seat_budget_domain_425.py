"""Independent budget tree bookkeeping and final structural/domain replay.
No model construction, root solver, nonlinear optimizer or Torch code is called.
Full geometry/finite work is separately replayed for the exported node only.
All node/callback primal arrays are used to verify the declared branching rules.
"""
from pathlib import Path
import numpy as np,json,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def simplex(z):
    e=np.exp(np.r_[z,0.]-max(np.r_[z,0.]));return e/math.fsum(map(float,e))
def audit_domain(fixture,data):
    folder=D/'budget_path_424';initial=dict(np.load(folder/'structural_initial.npz'));fin=dict(np.load(folder/'final.npz'));z=np.load(folder/'parameters.npy');solver=np.load(folder/'solver_final.npy');prep=dict(np.load(D/'monotone_preparation_410/fixture.npz'));keep=initial['keep'];scale=initial['scale']
    hard=np.ones(27708,bool);hard[12212:19508]=False;hard[19759]=False;hard[20284:]=False
    assert np.array_equal(hard,fin['hard']) and np.array_equal(hard,initial['hard'])
    assert np.array_equal(z,np.r_[solver[:898],650.-solver[894:898].sum()]) and np.array_equal(initial['z'],prep['z'])
    b=dict(np.load(folder/'initial.npz'));lo=np.c_[b['lower'][:,2:4],b['lower'][:,21:23],b['lower'][:,16:21]].ravel()[prep['shape']];hi=np.c_[b['upper'][:,2:4],b['upper'][:,21:23],b['upper'][:,16:21]].ravel()[prep['shape']]
    radius=np.r_[np.ones(785),np.full(109,.1)];assert np.array_equal(initial['lower'][:894],initial['z'][:894]-radius) and np.array_equal(initial['upper'][:894],initial['z'][:894]+radius)
    assert np.array_equal(initial['lower'][894:898],[88,189,241,101]) and np.array_equal(initial['upper'][894:898],[98,199,251,106])
    assert np.array_equal(lo,initial['lo']) and np.array_equal(hi,initial['hi'])
    def rows(c):
        xx=c['solver'];zz=c['z'];assert np.array_equal(zz,np.r_[xx[:898],650.-xx[894:898].sum()])
        return np.r_[c['ineq'][keep]/scale[keep]+(~hard[keep])*xx[-1],c['v'][:786]-lo,hi-c['v'][:786],zz[-1]-11.,21.-zz[-1]]
    queue=[dict(parent=None,kind='root_relaxation',lower=initial['lower'],upper=initial['upper'],initial=initial['initial'])];records=[];total_callbacks=0;strict_ids=[]
    for index,rec in enumerate(data['algorithm']['nodes']):
        assert queue;expected=queue.pop(0);nf=folder/('node_%02d'%index);ni=dict(np.load(nf/'initial.npz'));c=dict(np.load(nf/'final.npz'));xx=c['solver'];zz=c['z'];aa=rows(c);L=ni['lower'];U=ni['upper']
        assert rec['parent']==expected['parent'] and rec['kind']==expected['kind'] and rec['node']==index
        for key,ekey in [('solver','initial'),('lower','lower'),('upper','upper')]:assert np.array_equal(ni[key],expected[ekey])
        assert np.array_equal(ni['free'],np.flatnonzero(L<U)) and np.array_equal(ni['fixed'],np.flatnonzero(L==U))
        assert np.array_equal(aa,c['augmented']) and float(aa.min())==rec['minimum_augmented']
        soft=float(max(0.,np.max(-c['ineq'][keep&~hard]/scale[keep&~hard])));cm=float(min((c['v'][:786]-lo).min(),(hi-c['v'][:786]).min()));boxmin=float(min((xx-L).min(),(U-xx).min()));bg=zz[-5:];fraction=abs(bg[:4]-np.rint(bg[:4]));exact=np.array_equal(bg,np.rint(bg))
        assert soft==rec['soft_merit'] and cm==rec['coordinate_margin'] and boxmin==rec['box_margin'] and float(c['ineq'][keep&hard].min())==rec['minimum_hard'] and xx[-1]==rec['slack']
        numerical=bool(aa.min()>=-1e-8 and soft<=1e-8 and xx[-1]<=1e-8 and abs(c['eq']).max()<=1e-8 and boxmin>=0)
        strict=bool(exact and np.all(c['ineq'][keep]>=0) and cm>=0 and np.all(bg>=[88,189,241,101,11]) and np.all(bg<=[98,199,251,106,21]) and abs(c['eq']).max()<=1e-8 and boxmin>=0)
        assert numerical==rec['numerical_branching_gate_passed'] and strict==rec['strict_model_point']
        decision='unresolved_local_relaxation';branch=None
        if strict:
            strict_ids.append(index);decision='exact_integer_strict_model_point';assert index==len(data['algorithm']['nodes'])-1
        elif numerical and rec['kind']!='exact_budget_fixing':
            if fraction.max()<=1e-7:
                integers=np.rint(bg[:4]);fifth=650.-integers.sum()
                if np.all(integers>=L[894:898]) and np.all(integers<=U[894:898]) and 11<=fifth<=21:
                    nl=L.copy();nu=U.copy();nl[894:898]=integers;nu[894:898]=integers;ns=xx.copy();ns[894:898]=integers;queue.append(dict(parent=index,kind='exact_budget_fixing',lower=nl,upper=nu,initial=ns));decision='queued_exact_fixing_recheck'
                else:decision='near_integer_fixing_outside_domain'
            else:
                g=int(np.argmax(fraction));at=894+g;floor=float(np.floor(xx[at]));ceil=floor+1;branch=dict(budget=g,value=float(xx[at]),floor=floor,ceil=ceil);decision='branched_fractional_relaxation'
                for side in ['lower','upper']:
                    nl=L.copy();nu=U.copy()
                    if side=='lower':nu[at]=floor
                    else:nl[at]=ceil
                    if nl[at]<=nu[at]:
                        ns=xx.copy();ns[at]=floor if side=='lower' else ceil;queue.append(dict(parent=index,kind='branch_'+side,lower=nl,upper=nu,initial=ns))
        assert rec['decision']==decision and rec['branch']==branch
        for j,rr in enumerate(rec['history']):
            cb=dict(np.load(nf/('callback_%02d.npz'%j)));ar=rows(cb);ss=float(max(0.,np.max(-cb['ineq'][keep&~hard]/scale[keep&~hard])))
            assert rr['callback']==j and rr['not_selected'] and rr['slack']==float(cb['solver'][-1]) and rr['soft_merit']==ss and rr['minimum_augmented']==float(ar.min()) and rr['minimum_hard']==float(cb['ineq'][keep&hard].min()) and rr['budgets']==cb['z'][-5:].tolist()
        assert len(rec['history'])==rec['callback_count']<=30 and rec['optimizer']['iterations']<=30
        total_callbacks+=rec['callback_count'];records.append(dict(node=index,decision=decision,numerical_gate=numerical,strict_model_point=strict,callbacks=rec['callback_count']))
    assert 1<=len(records)<=5 and len(records)==data['algorithm']['optimizer_calls'] and total_callbacks==data['algorithm']['callback_count']
    exported=strict_ids[0] if strict_ids else 0;assert exported==data['algorithm']['exported_node']
    chosen=dict(np.load(folder/('node_%02d'%exported)/'final.npz'));assert np.array_equal(chosen['solver'],solver) and np.array_equal(chosen['ineq'],fin['ineq']) and np.array_equal(chosen['eq'],fin['eq'])
    finalstate=dict(np.load(folder/'structural.npz'));v=finalstate['v'];aug=rows(dict(solver=solver,z=z,ineq=fin['ineq'],v=v));assert np.array_equal(aug,np.load(folder/'augmented_final.npy'))
    coordinate_margin=float(min((v[:786]-lo).min(),(hi-v[:786]).min()));assert coordinate_margin==data['algorithm']['minimum_coordinate_bound']
    s=dict(np.load(folder/'structural.npz'));z=s['z'];FLOOR=2.**(-19);ss=[slice(int(a),int(b)) for a,b in prep['coordinate_slices']];loads=np.zeros((114,8));maxmoment=0.
    for i in range(114):
        aa=np.flatnonzero(prep['active'][i]);P=prep['PP'][i,aa,:2];N=prep['null'][i,aa];N=N[:,np.any(N!=0,axis=0)]
        if i<103:
            F=math.fsum(map(float,prep['wbase'][i,aa]));loads[i,aa]=FLOOR+(F-len(aa)*FLOOR)*simplex(z[ss[i]])
        else:
            A=np.vstack([np.ones(len(aa)),((P-P.mean(0))/.1).T]);loads[i,aa]=FLOOR+np.exp(N@z[ss[i]]+A.T@s['moment_lam'][i-103]);maxmoment=max(maxmoment,float(abs(A@(loads[i,aa]-prep['wbase'][i,aa])).max()))
    loaderr=float(abs(loads-s['Y'][:,46:54]).max());assert loaderr<1e-10 and maxmoment<1e-10
    assert np.all(loads[prep['active']]>FLOOR) and np.all(loads[~prep['active']]==0)
    ori=np.cumsum(FLOOR+(1-104*FLOOR)*simplex(z[ss[114]]))[:103];fH=s['X'][98,2];pre=np.cumsum(FLOOR+(fH-99*FLOOR)*simplex(z[ss[115]]))[:98];post=fH+np.cumsum(FLOOR+(1-fH-5*FLOOR)*simplex(z[ss[116]]))[:4]
    phaseerr=float(max(abs(ori-s['X'][:103,3]).max(),abs(pre-s['X'][:98,2]).max(),abs(post-s['X'][99:103,2]).max()));assert phaseerr<1e-12
    increments=np.diff(np.r_[fH,post,1.]);assert np.all(increments>=FLOOR)
    gaperr=float(abs(.012-s['Y'][98,54]));assert gaperr<1e-10
    return dict(nodes=records,all_node_and_callback_bookkeeping_replayed=True,callback_count=total_callbacks,callbacks_not_selected=True,export_rule_verified=True,exported_node=exported,optimizer_calls=len(records),maximum_nodes=5,maximum_iterations_per_node=30,event_budgets=z[-5:].tolist(),mandatory_events_exactly_integer=bool(np.array_equal(z[-5:],np.rint(z[-5:]))),all_3648_static_command_rows_hard=True,additional_coordinate_bound_rows=1572,dependent_budget_bound_rows=2,final_augmented_minimum=float(aug.min()),final_coordinate_margin=coordinate_margin,strict_hard_passed=bool(np.all(fin['ineq'][keep&hard]>=0) and coordinate_margin>=0 and abs(fin['eq']).max()<=1e-8),positive_load_reconstruction=loaderr,endpoint_moment_error=maxmoment,cumulative_phase_error=phaseerr,posthover_increment_minimum=float(increments.min()),hover_gap_error=gaperr,actual_float32_schedule=False,physical_admission=False)

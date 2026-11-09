"""Independent saved predictor, fraction and strict acceptance replay412.
No construction/root solver, optimizer or Torch code is called.
"""
from pathlib import Path
import numpy as np,json,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def simplex(z):
    e=np.exp(np.r_[z,0.]-max(np.r_[z,0.]));return e/math.fsum(map(float,e))
def audit_steps(fixture,data):
    folder=D/'monotone_path_412';s=dict(np.load(folder/'structural_initial.npz'));n=len(s['z']);assert n==899;current=s['z'].copy();keep=s['keep'];scale=s['scale'];hard=np.ones(27708,bool)
    for k in range(114):hard[k*80:k*80+32]=False
    hard[12212:19508]=False;hard[19759]=False;hard[20284:]=False;assert np.array_equal(hard,s['hard']);soft=keep&~hard;hardmask=keep&hard
    assert np.all(s['ineq'][hardmask]>=0) and abs(s['eq']).max()<=1e-8
    base=dict(np.load(folder/'initial.npz'));lo=np.c_[base['lower'][:,2:4],base['lower'][:,21:23],base['lower'][:,16:21]].ravel();hi=np.c_[base['upper'][:,2:4],base['upper'][:,21:23],base['upper'][:,16:21]].ravel();prep=dict(np.load(D/'monotone_preparation_410/fixture.npz'));shape=prep['shape']
    oldrad=np.r_[np.full(786,.02),np.full(109,.1),np.full(5,5.)];newrad=np.r_[np.ones(785),np.full(109,.1),np.full(5,5.)];records=[];trust=1.;trials_count=0;accepted_count=0
    for rec in data['algorithm']['history']:
        stepdir=folder/('step_%02d'%rec['iteration']);f=dict(np.load(stepdir/'linear_problem.npz'));assert np.array_equal(f['z'],current) and float(f['radius'])==trust
        m=int(keep.sum());A=np.zeros((m+901,n+1));lower=np.full(m+901,-np.inf);upper=np.full(m+901,np.inf)
        A[:m,:n]=f['I'][keep]/scale[keep,None];A[:m,n]=(~hard[keep]).astype(float);lower[:m]=-f['ineq'][keep]/scale[keep]*np.where(hard[keep],.99,1.)
        A[m:m+900,:n]=f['G'];lower[m:m+900]=-oldrad*trust;upper[m:m+900]=oldrad*trust
        lower[m:m+786]=np.maximum(lower[m:m+786],.99*(lo[shape]-f['v'][:786]));upper[m:m+786]=np.minimum(upper[m:m+786],.99*(hi[shape]-f['v'][:786]));A[m+900,894:899]=1.;lower[-1]=upper[-1]=0.
        assert np.array_equal(A,f['A']) and np.array_equal(lower,f['lower']) and np.array_equal(upper,f['upper'])
        blo=-newrad*trust;bhi=newrad*trust
        for g,(a,b) in enumerate([(0,16),(16,64),(64,99),(99,104),(104,114)]):blo[894+g]=max(blo[894+g],b-a+1-current[894+g])
        blo[897]=max(blo[897],101-current[897]);merit=max(0.,float((-f['ineq'][soft]/scale[soft]).max()));blo=np.r_[blo,0.];bhi=np.r_[bhi,merit+1e-12]
        assert np.array_equal(blo,f['bounds_lower']) and np.array_equal(bhi,f['bounds_upper']);assert np.array_equal(f['cost'],np.r_[np.zeros(n),1.]);assert np.array_equal(f['integrality'],np.r_[np.zeros(894),np.ones(5),0])
        if not rec['candidate_available']:records.append(dict(iteration=rec['iteration'],matrix_exact=True,candidate_available=False));break
        u=np.load(stepdir/'solution.npy');av=A@u;lin=float(min((av-lower)[np.isfinite(lower)].min(),(upper-av)[np.isfinite(upper)].min()));bound=float(min((u-blo).min(),(bhi-u).min()));integer=float(abs(u[894:899]-np.rint(u[894:899])).max());passed=bool(lin>=-1e-7 and bound>=-1e-7 and integer<1e-7)
        assert abs(lin-rec['linear_minimum'])<1e-12 and abs(bound-rec['bound_minimum'])<1e-12 and integer==rec['integer_decode_error'] and passed==rec['linear_numerical_passed']
        out=dict(iteration=rec['iteration'],matrix_exact=True,linear_minimum=lin,bound_minimum=bound,integer_error=integer,passed=passed)
        if not passed:records.append(out);break
        changed=bool(np.any(np.rint(u[894:899])!=0));assert changed==rec['integer_budget_changed'];assert 1<=len(rec['trials'])<=(1 if changed else 8)
        decision=False;outcomes=[]
        for j,t in enumerate(rec['trials']):
            alpha=2.**(-j);assert t['index']==j and alpha==t['alpha'];assert not changed or alpha==1.
            delta=alpha*u[:n];delta[894:899]=np.rint(u[894:899]);wanted=current+delta;trial=np.load(stepdir/f'trial_{j:02d}.npy');assert np.array_equal(trial,wanted)
            predicted=max(0.,float((-(f['ineq']+f['I']@delta)[soft]/scale[soft]).max()));assert abs(predicted-t['predicted_merit'])<1e-12
            if not t['evaluated']:assert 'error' in t and not t['accepted'];outcomes.append(dict(index=j,alpha=alpha,domain_failure=True));break
            q=dict(np.load(stepdir/f'trial_{j:02d}_constraints.npz'));after=max(0.,float((-q['ineq'][soft]/scale[soft]).max()));margin=float(min((q['v'][:786]-lo[shape]).min(),(hi[shape]-q['v'][:786]).min()));hardok=bool(np.all(q['ineq'][hardmask]>=0) and abs(q['eq']).max()<=1e-8 and margin>=0);pred=merit-predicted;reduction=merit-after;ratio=reduction/pred if pred>0 else None;accept=bool(hardok and pred>0 and reduction>0 and ratio>=.1)
            assert margin==t['actual_coordinate_bound_margin'] and accept==t['accepted'] and hardok==t['hard_nonlinear_passed'] and abs(after-t['trial_merit'])<1e-12
            trials_count+=1;outcomes.append(dict(index=j,alpha=alpha,hard_passed=hardok,accepted=accept,merit=after,ratio=ratio))
            if accept:
                assert j==len(rec['trials'])-1;current=trial;accepted_count+=1;decision=True
                if alpha<1:trust*=.5
                elif ratio>.75:trust=min(1.,2*trust)
        assert decision==rec['accepted']
        if not decision:trust*=.5
        out.update(accepted=decision,trials=outcomes);records.append(out)
    assert np.array_equal(current,np.load(folder/'parameters.npy')) and len(records)<=16 and trials_count<=128
    assert accepted_count==data['algorithm']['accepted_steps'] and trials_count==data['algorithm']['actual_nonlinear_trials']
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
    return dict(records=records,unchanged_independent_linear_bound=1e-7,maximum_subproblems=16,maximum_nonlinear_trials=128,actual_nonlinear_trials=trials_count,source_matrices_and_bounds_exact=True,fraction_to_boundary_predictor_replayed=True,strict_hard_nonlinear_backtracking_replayed=True,positive_load_reconstruction=loaderr,endpoint_moment_error=maxmoment,cumulative_phase_error=phaseerr,posthover_increment_minimum=float(increments.min()),hover_gap_error=gaperr,exact_linear_gap_profile_removed=True,physical_admission=False)

"""Independent saved structural-map and constrained-acceptance replay406.
No construction root solver, optimizer, or Torch code is called.
"""
from pathlib import Path
import numpy as np,json,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def simplex(z):
    vals=list(z)+[0.];top=max(vals);e=[math.exp(float(x)-top) for x in vals];return np.array(e)/math.fsum(e)
def audit_steps(fixture,data):
    folder=D/'structural_path_406';s=dict(np.load(folder/'structural_initial.npz'));n=len(s['z']);assert n==895;current=s['z'].copy();keep=s['keep'];scale=s['scale'];hard=np.ones(27708,bool)
    for k in range(114):hard[k*80:k*80+32]=False
    hard[12212:19508]=False;hard[19759]=False;hard[20284:]=False;assert np.array_equal(hard,s['hard']);soft=keep&~hard;hardmask=keep&hard
    assert np.all(s['ineq'][hardmask]>=0) and abs(s['eq']).max()<=1e-8
    base=dict(np.load(folder/'initial.npz'));lo=np.c_[base['lower'][:,2:4],base['lower'][:,21:23],base['lower'][:,16:21]].ravel();hi=np.c_[base['upper'][:,2:4],base['upper'][:,21:23],base['upper'][:,16:21]].ravel();prep=dict(np.load(D/'structural_preparation_404/fixture.npz'));shape=prep['shape']
    oldrad=np.r_[np.full(786,.02),np.full(109,.1),np.full(5,5.)];newrad=np.r_[np.ones(781),np.full(109,.1),np.full(5,5.)];records=[];trust=1.
    for rec in data['algorithm']['history']:
        stepdir=folder/('step_%02d'%rec['iteration']);f=dict(np.load(stepdir/'linear_problem.npz'));assert np.array_equal(f['z'],current) and float(f['radius'])==trust
        m=int(keep.sum());A=np.zeros((m+901,896));lower=np.full(m+901,-np.inf);upper=np.full(m+901,np.inf)
        A[:m,:n]=f['I'][keep]/scale[keep,None];A[:m,n]=(~hard[keep]).astype(float);lower[:m]=-f['ineq'][keep]/scale[keep]
        A[m:m+900,:n]=f['G'];lower[m:m+900]=-oldrad*trust;upper[m:m+900]=oldrad*trust
        lower[m:m+786]=np.maximum(lower[m:m+786],lo[shape]-f['v'][:786]);upper[m:m+786]=np.minimum(upper[m:m+786],hi[shape]-f['v'][:786]);A[m+900,890:895]=1.;lower[-1]=upper[-1]=0.
        assert np.array_equal(A,f['A']) and np.array_equal(lower,f['lower']) and np.array_equal(upper,f['upper'])
        blo=-newrad*trust;bhi=newrad*trust
        for g,(a,b) in enumerate([(0,16),(16,64),(64,99),(99,104),(104,114)]):blo[890+g]=max(blo[890+g],b-a+1-current[890+g])
        blo[893]=max(blo[893],101-current[893]);merit=max(0.,float((-f['ineq'][soft]/scale[soft]).max()));blo=np.r_[blo,0.];bhi=np.r_[bhi,merit+1e-12]
        assert np.array_equal(blo,f['bounds_lower']) and np.array_equal(bhi,f['bounds_upper']);assert np.array_equal(f['cost'],np.r_[np.zeros(895),1.]);assert np.array_equal(f['integrality'],np.r_[np.zeros(890),np.ones(5),0])
        if not rec['candidate_available']:records.append(dict(iteration=rec['iteration'],matrix_exact=True,candidate_available=False));break
        u=np.load(stepdir/'solution.npy');av=A@u;lin=float(min((av-lower)[np.isfinite(lower)].min(),(upper-av)[np.isfinite(upper)].min()));bound=float(min((u-blo).min(),(bhi-u).min()));integer=float(abs(u[890:895]-np.rint(u[890:895])).max());passed=bool(lin>=-1e-7 and bound>=-1e-7 and integer<1e-7)
        assert abs(lin-rec['linear_minimum'])<1e-12 and abs(bound-rec['bound_minimum'])<1e-12 and integer==rec['integer_decode_error'] and passed==rec['linear_numerical_passed']
        out=dict(iteration=rec['iteration'],matrix_exact=True,linear_minimum=lin,bound_minimum=bound,integer_error=integer,passed=passed)
        if not passed:records.append(out);break
        trial=np.load(stepdir/'trial.npy');wanted=current+u[:895];wanted[890:895]=current[890:895]+np.rint(u[890:895]);assert np.array_equal(trial,wanted)
        if not (stepdir/'trial_constraints.npz').exists():assert 'domain failure' in rec['termination'];records.append(out);break
        q=dict(np.load(stepdir/'trial_constraints.npz'));after=max(0.,float((-q['ineq'][soft]/scale[soft]).max()));coordinate_margin=float(min((q['v'][:786]-lo[shape]).min(),(hi[shape]-q['v'][:786]).min()));hardok=bool(np.all(q['ineq'][hardmask]>=0) and abs(q['eq']).max()<=1e-8 and coordinate_margin>=0);pred=merit-u[-1];reduction=merit-after;ratio=reduction/pred if pred>0 else None;accept=bool(hardok and pred>0 and reduction>0 and ratio>=.1)
        assert coordinate_margin==rec['actual_coordinate_bound_margin']
        assert accept==rec['accepted'] and hardok==rec['hard_nonlinear_passed'];assert abs(after-rec['trial_merit'])<1e-12
        if accept:
            current=trial
            if ratio>.75:trust=min(1.,2*trust)
        else:trust*=.5
        out.update(accepted=accept,hard_nonlinear_passed=hardok,actual_merit=after,acceptance_ratio=ratio);records.append(out)
    assert np.array_equal(current,np.load(folder/'parameters.npy')) and len(records)<=16
    assert sum(v.get('accepted',False) for v in records)==data['algorithm']['accepted_steps'];assert sum('actual_merit' in v for v in records)==data['algorithm']['actual_nonlinear_trials']
    # Rebuild final positive weights, phase coordinates and gap construction.
    s=dict(np.load(folder/'structural.npz'));z=s['z'];FLOOR=2.**(-19);ss=[slice(int(a),int(b)) for a,b in prep['coordinate_slices']];loads=np.zeros((114,8));maxmoment=0.
    for i in range(114):
        aa=np.flatnonzero(prep['active'][i]);P=prep['PP'][i,aa,:2];N=prep['null'][i,aa];N=N[:,np.any(N!=0,axis=0)]
        if i<103:
            F=math.fsum(map(float,prep['wbase'][i,aa]));loads[i,aa]=FLOOR+(F-len(aa)*FLOOR)*simplex(z[ss[i]])
        else:
            A=np.vstack([np.ones(len(aa)),((P-P.mean(0))/.1).T]);loads[i,aa]=FLOOR+np.exp(N@z[ss[i]]+A.T@s['moment_lam'][i-103]);maxmoment=max(maxmoment,float(abs(A@(loads[i,aa]-prep['wbase'][i,aa])).max()))
    loaderr=float(abs(loads-s['Y'][:,46:54]).max());assert loaderr<1e-10 and maxmoment<1e-10
    assert np.all(loads[prep['active']]>FLOOR) and np.all(loads[~prep['active']]==0)
    ori=np.cumsum(FLOOR+(1-104*FLOOR)*simplex(z[ss[114]]))[:103];pre=s['X'][98,2];descent=np.cumsum(FLOOR+(pre-99*FLOOR)*simplex(z[ss[115]]))[:98];phaseerr=float(max(abs(ori-s['X'][:103,3]).max(),abs(descent-s['X'][:98,2]).max()));assert phaseerr<1e-12
    old=json.loads((D/'clearance_lift_path_330/results.json').read_text())['dense']['path'];dt=s['dt'];elapsed=np.array([math.fsum(map(float,dt[99:k])) for k in range(99,104)]);gap=.012+(old[104]['pelvis_gap_m']-.012)*elapsed/math.fsum(map(float,dt[99:104]));gaperr=float(abs(gap-s['Y'][98:103,54]).max());assert gaperr<1e-10
    return dict(records=records,unchanged_independent_linear_bound=1e-7,maximum_subproblems=16,source_matrices_and_bounds_exact=True,strict_hard_nonlinear_acceptance_replayed=True,positive_load_reconstruction=loaderr,endpoint_moment_error=maxmoment,cumulative_phase_error=phaseerr,prescribed_gap_error=gaperr,physical_admission=False)

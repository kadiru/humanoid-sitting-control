"""Independent all-hard integer-exact closest-path domain replay; no optimizer/root."""
from pathlib import Path
import numpy as np,json,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def simplex(z):
    e=np.exp(np.r_[z,0.]-max(np.r_[z,0.]));return e/math.fsum(map(float,e))
def audit_domain(fixture,data):
    folder=D/'lattice_path_430';initial=dict(np.load(folder/'structural_initial.npz'));fin=dict(np.load(folder/'final.npz'));prep=dict(np.load(D/'monotone_preparation_410/fixture.npz'));f=dict(np.load(D/'lattice_preparation_428/fixture.npz'));state=dict(np.load(folder/'structural.npz'));z=np.load(folder/'parameters.npy');x=np.load(folder/'solver_final.npy')
    assert np.array_equal(z,np.r_[x,f['budgets']]) and np.array_equal(initial['reference'],f['reference'])
    assert np.array_equal(initial['lower'],f['lower']) and np.array_equal(initial['upper'],f['upper'])
    assert np.array_equal(z[-5:],np.rint(z[-5:])) and sum(z[-5:])==650
    hard=np.ones(27708,bool);assert np.array_equal(hard,fin['hard'])
    keep=f['keep'];scale=f['scale'];lo=f['lo'];hi=f['hi'];eps=2.**-23
    def rows(iq,v):return np.r_[iq[keep]/scale[keep]-eps,v[:786]-lo-eps,hi-v[:786]-eps]
    ar=rows(fin['ineq'],state['v']);assert np.array_equal(ar,np.load(folder/'hard_final.npy'));assert ar.min()==data['algorithm']['minimum_buffered_hard_row']
    boxmargin=float(min((x-f['lower']).min(),(f['upper']-x).min()));cm=float(min((state['v'][:786]-lo).min(),(hi-state['v'][:786]).min()))
    assert boxmargin==data['algorithm']['minimum_box_bound'] and cm==data['algorithm']['minimum_original_coordinate_bound']
    strict=bool(ar.min()>=0 and abs(fin['eq']).max()<=1e-8 and boxmargin>=0);numerical=bool(ar.min()>=-1e-8 and abs(fin['eq']).max()<=1e-8 and boxmargin>=0)
    assert strict==data['algorithm']['buffered_hard_strict_passed'] and numerical==data['algorithm']['buffered_hard_numerical_passed']
    obj=.5*float(((x-f['reference'])/f['radius'])@((x-f['reference'])/f['radius']));assert obj==data['optimizer']['nearest_path_objective']
    history=data['algorithm']['history'];assert len(history)==data['algorithm']['callback_count']<=30 and data['algorithm']['optimizer_calls']==1 and data['algorithm']['feasibility_slack_variables']==0
    for k,rr in enumerate(history):
        cb=dict(np.load(folder/('callback_%02d.npz'%k)));xx=cb['solver'];a=rows(cb['ineq'],cb['v']);assert np.array_equal(cb['z'],np.r_[xx,f['budgets']]) and np.array_equal(a,cb['hard_rows'])
        value=.5*float(((xx-f['reference'])/f['radius'])@((xx-f['reference'])/f['radius']));assert value==rr['objective'] and float(a.min())==rr['minimum_all_hard_row'] and rr['not_selected']
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
    return dict(optimizer_calls=1,all_constraints_hard=True,feasibility_slack_variables=0,callback_count=len(history),callbacks_not_selected=True,budgets=z[-5:].tolist(),mandatory_events_exactly_integer=True,all_28887_buffered_rows_replayed=True,buffered_minimum=float(ar.min()),buffered_strict_passed=strict,buffered_numerical_passed=numerical,objective_replay_exact=True,coordinate_margin=cm,box_margin=boxmargin,positive_load_reconstruction=loaderr,endpoint_moment_error=maxmoment,cumulative_phase_error=phaseerr,posthover_increment_minimum=float(increments.min()),hover_gap_error=gaperr,actual_float32_schedule=False,physical_admission=False)

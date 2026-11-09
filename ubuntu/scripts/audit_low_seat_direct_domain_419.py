"""Independent final direct-NLP domain/epigraph and saved callback bookkeeping.
No construction/root solver, optimizer, or Torch code is called. Callback records
are intermediate diagnostics, never candidate selection or physical evidence.
"""
from pathlib import Path
import numpy as np,json,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def simplex(z):
    e=np.exp(np.r_[z,0.]-max(np.r_[z,0.]));return e/math.fsum(map(float,e))
def audit_domain(fixture,data):
    folder=D/'direct_path_418';initial=dict(np.load(folder/'structural_initial.npz'));fin=dict(np.load(folder/'final.npz'));z=np.load(folder/'parameters.npy');solver=np.load(folder/'solver_final.npy')
    prep=dict(np.load(D/'monotone_preparation_410/fixture.npz'));keep=initial['keep'];scale=initial['scale']
    hard=np.ones(27708,bool);hard[12212:19508]=False;hard[19759]=False;hard[20284:]=False
    assert np.array_equal(hard,fin['hard']) and np.array_equal(hard,initial['hard'])
    assert np.array_equal(z,np.r_[solver[:894],initial['z'][-5:]]) and np.array_equal(z[-5:],[93,194,246,101,16])
    assert np.array_equal(initial['z'],prep['z']) and data['revision']['maximum_iterations']==30 and data['algorithm']['optimizer_calls']==1
    b=dict(np.load(folder/'initial.npz'));lo=np.c_[b['lower'][:,2:4],b['lower'][:,21:23],b['lower'][:,16:21]].ravel()[prep['shape']];hi=np.c_[b['upper'][:,2:4],b['upper'][:,21:23],b['upper'][:,16:21]].ravel()[prep['shape']]
    radius=np.r_[np.ones(785),np.full(109,.1)];assert np.array_equal(initial['lower'],initial['z'][:894]-radius) and np.array_equal(initial['upper'],initial['z'][:894]+radius)
    assert np.array_equal(lo,initial['lo']) and np.array_equal(hi,initial['hi'])
    finalstate=dict(np.load(folder/'structural.npz'));v=finalstate['v'];aug=np.r_[fin['ineq'][keep]/scale[keep]+(~hard[keep])*solver[-1],v[:786]-lo,hi-v[:786]]
    assert np.array_equal(aug,np.load(folder/'augmented_final.npy')) and solver[-1]==float(fin['slack'])
    assert float(aug.min())==data['algorithm']['minimum_augmented_row']
    coordinate_margin=float(min((v[:786]-lo).min(),(hi-v[:786]).min()));assert coordinate_margin==data['algorithm']['minimum_coordinate_bound']
    box_margin=float(min((solver[:894]-initial['lower']).min(),(initial['upper']-solver[:894]).min(),solver[-1],initial['initial'][-1]-solver[-1]))
    records=[]
    for index,rec in enumerate(data['algorithm']['history']):
        c=dict(np.load(folder/('callback_%02d.npz'%index)));assert np.array_equal(c['z'],np.r_[c['x'][:894],initial['z'][-5:]])
        cs=float(c['x'][-1]);soft=max(0.,float((-c['ineq'][keep&~hard]/scale[keep&~hard]).max()));hm=float(c['ineq'][keep&hard].min());cm=float(min((c['v'][:786]-lo).min(),(hi-c['v'][:786]).min()))
        ca=np.r_[c['ineq'][keep]/scale[keep]+(~hard[keep])*cs,c['v'][:786]-lo,hi-c['v'][:786]]
        assert cs==rec['slack'] and soft==rec['soft_merit'] and hm==rec['minimum_hard_row'] and cm==rec['minimum_coordinate_bound'] and float(ca.min())==rec['minimum_augmented_row']
        assert rec['callback']==index and rec['intermediate_not_candidate']
        records.append(dict(callback=index,hard_passed=bool(hm>=0 and cm>=0 and abs(c['eq']).max()<=1e-8),slack=cs,not_selected=True))
    assert len(records)==data['algorithm']['callback_count'] and len(records)<=30
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
    return dict(callback_records=records,callback_arrays_replayed=True,callbacks_not_selected=True,final_solver_result_used=True,optimizer_calls=1,maximum_iterations=30,fixed_event_budgets=z[-5:].tolist(),all_3648_static_command_rows_hard=True,additional_coordinate_bound_rows=1572,final_augmented_minimum=float(aug.min()),final_coordinate_margin=coordinate_margin,final_box_margin=box_margin,strict_hard_passed=bool(np.all(fin['ineq'][keep&hard]>=0) and coordinate_margin>=0 and abs(fin['eq']).max()<=1e-8),positive_load_reconstruction=loaderr,endpoint_moment_error=maxmoment,cumulative_phase_error=phaseerr,posthover_increment_minimum=float(increments.min()),hover_gap_error=gaperr,actual_float32_schedule=False,physical_admission=False)

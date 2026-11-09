"""ONE bounded direct nonlinear complete-path solve, original406 seed.

Static command limits are hard nonlinear rows, as are support/phase/gap/normal
speed/planning/original coordinate bounds. Only rates and target headroom/window
have a nonnegative epigraph. All894 continuous path coordinates vary together.
Five inherited event budgets stay fixed [93,194,246,101,16]; no allocation search.
SLSQP sees actual nonlinear constraints/Jacobians directly, with at most30
iterations, no MILP predictor, trust-floor loop, restart, correction or polishing.
Internal iterates are NOT admitted. Only the unselected final solver output is
exported for independent original-criteria checks, regardless of success flag.
No actual filter, runtime, physics or policy updates.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,time
from scipy.optimize import minimize,Bounds
from low_seat_monotone_phase_410 import make_phase,FLOOR
from low_seat_direct_constraints_416 import partition,coordinate_bounds,assemble,box
from low_seat_event_clock_v2_386 import clock_samples,clock_rows
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'direct_path_418'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False)
    prep=json.loads((D/'direct_preparation_416/preparation.json').read_text());audit=json.loads((D/'direct_preparation_audit_417/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
    hashes={**prep['source_input_sha256'],**audit['source_sha256']}
    for q in [Path(__file__),Path('scripts/audit_low_seat_direct_path_419.py'),Path('scripts/audit_low_seat_direct_domain_419.py'),Path('scripts/export_low_seat_support_373.py')]:hashes[str(q.resolve())]=sha(q)
    for path,h in hashes.items():assert sha(path)==h,path
    f=dict(np.load(D/'direct_preparation_416/fixture.npz'));base=dict(np.load(D/'window_path_400/initial.npz'))
    d=make_phase();w=d['d'];p=d['p'];seed=d['initial'].copy();assert np.array_equal(seed,f['seed'])
    hard=partition();keep=w['keep'];scale=w['scale'];lo,hi=coordinate_bounds(base,d['shape']);budgets=seed[894:].copy()
    lower,upper=box(seed);initial=np.r_[seed[:894],float(f['slack'])];cost=np.r_[np.zeros(894),1.]
    eq,iq=d['evaluate'](seed);assert np.array_equal(assemble(iq,d['last']['v'],initial[-1],scale,keep,lo,hi),f['values'])
    revision=dict(scope=__doc__,formulation='direct nonlinear command-hard epigraph',source_input_sha256=hashes,seed='original406 reconstructed by immutable410; not412',maximum_optimizer_calls=1,maximum_iterations=30,optimizer='SLSQP',ftol=1e-10,continuous_free_path_variables=894,nonnegative_epigraph_variables=1,fixed_event_budgets=budgets.tolist(),event_allocation_search=False,maximum_log_coordinate_displacement=1.,maximum_time_logratio_displacement=.1,coordinate_box_scope='single declared research domain; no shrink/expand or retuning',old_coordinate_bounds_explicit=True,static_command_rows_hard=True,hard_rows=int((hard&keep).sum()),soft_rows=int((~hard&keep).sum()),original_rows=20284,continuous_window_rows=7424,total_rows=27708,augmented_rows=len(f['values']),additional_coordinate_bound_rows=1572,omitted_constant_contact_rows=392,omitted_total_time_identity=1,hover_constructed_and_final_checked=True,inward_reserves_unchanged=True,new_structural_floor=FLOOR,internal_iterates_may_be_infeasible=True,callback_selection=False,solver_success_is_not_admission=True,actual_float32_filter_required_later=True,full_geometry_mandatory_final_screen=True,physical_admission=False)
    (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'initial.npz',**base)
    np.savez_compressed(O/'structural_initial.npz',z=seed,eq=eq,ineq=iq,X=d['last']['X'],Y=d['last']['Y'],hard=hard,keep=keep,scale=scale,lo=lo,hi=hi,lower=lower,upper=upper,initial=initial)
    history=[];started=time.monotonic()
    def expand(x):return np.r_[x[:894],budgets]
    def constraints(x):
        e,i=d['evaluate'](expand(x));return assemble(i,d['last']['v'],x[-1],scale,keep,lo,hi)
    def jacobian(x):
        e,i=d['evaluate'](expand(x));E,I=d['evaluate'](expand(x),True)
        return assemble(i,d['last']['v'],x[-1],scale,keep,lo,hi,I,d['last']['G'])[1]
    def callback(x):
        z=expand(x);e,i=d['evaluate'](z);s=d['last'];rows=assemble(i,s['v'],x[-1],scale,keep,lo,hi)
        index=len(history);np.savez_compressed(O/('callback_%02d.npz'%index),x=x,z=z,eq=e,ineq=i,v=s['v'],X=s['X'],Y=s['Y'])
        rec=dict(callback=index,elapsed_s=time.monotonic()-started,slack=float(x[-1]),soft_merit=float(max(0.,np.max(-i[keep&~hard]/scale[keep&~hard]))),minimum_hard_row=float(i[keep&hard].min()),minimum_coordinate_bound=float(min((s['v'][:786]-lo).min(),(hi-s['v'][:786]).min())),minimum_augmented_row=float(rows.min()),normalized_hover=float(abs(e).max()),minimum_static_command_row=float(i[:9120].reshape(114,80)[:,:32].min()),intermediate_not_candidate=True)
        history.append(rec);(O/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(rec),flush=True)
    try:
        fit=minimize(lambda x:float(x[-1]),initial,jac=lambda x:cost,method='SLSQP',bounds=Bounds(np.r_[lower,0.],np.r_[upper,float(f['slack'])]),constraints=[dict(type='ineq',fun=constraints,jac=jacobian)],callback=callback,options=dict(maxiter=30,ftol=1e-10,disp=True))
    except Exception as exc:
        (O/'termination.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error_type=type(exc).__name__,error=str(exc),optimizer_calls=1,restarted=False,callback_count=len(history),final_candidate_available=False,physical_admission=False),indent=2));raise
    x=np.asarray(fit.x);z=expand(x);eq,iq=d['evaluate'](z);saved=dict(d['last']);softmerit=float(max(0.,np.max(-iq[keep&~hard]/scale[keep&~hard])));samples,loc=clock_samples(saved['C'],saved['dt'],budgets.astype(int));_,craw,cfull,_,targets=clock_rows(samples,p['command_lo'],p['command_hi'],p['rates'],w['ulp'])
    augmented=assemble(iq,saved['v'],x[-1],scale,keep,lo,hi)
    opt=dict(success=bool(fit.success),message=str(fit.message),status=int(fit.status),iterations=int(fit.nit),evaluations=int(fit.nfev),jacobian_evaluations=int(fit.njev),minimum_augmented_inequality=float(augmented.min()))
    ep=dict(p);ep['evaluate']=lambda _: (eq,iq)
    result=export_result(ep,saved['x'],opt,revision,hashes,O,float(x[-1]),scale,keep,hard)
    result.update(candidate_available=True,algorithm=dict(optimizer_calls=1,stop=str(fit.message),final_soft_merit=softmerit,final_solver_slack=float(x[-1]),fixed_budgets=True,hard_static_command_rows=True,callback_count=len(history),callbacks_not_selected=True,actual_constraint_evaluations=int(fit.nfev),history=history,minimum_augmented_row=float(augmented.min()),minimum_coordinate_bound=float(min((saved['v'][:786]-lo).min(),(hi-saved['v'][:786]).min()))),event_clock=dict(event_budgets=budgets.astype(int).tolist(),total_ticks=675,integer_time_s=13.5,minimum_original_clock_rate_row=float(craw[:,:32].min()),minimum_original_clock_target_row=float(craw[:,32:].min()),all_tightened_clock_rate_rows_passed=bool(np.all(cfull[:,:32]>=0)),all_tightened_clock_target_rows_passed=bool(np.all(cfull[:,32:]>=0)),actual_float32_schedule=False,physical_admission=False),window=dict(breakpoints=232,minimum_original_target_row=float(saved['env']['raw'].min()),minimum_tightened_target_row=float(saved['env']['tightened'].min()),all_strict_window_rows_passed=bool(np.all(saved['env']['tightened']>=0))))
    core=w['last'];np.savez_compressed(O/'event_clock.npz',parameters=saved['v'][:895],budgets=budgets.astype(int),raw_ineq=core['raw'],tighten=core['tighten'],clock=samples,clock_raw=craw,clock_full=cfull,locations=loc,targets=targets,ulp=w['ulp'])
    np.savez_compressed(O/'window.npz',**saved['env']);np.savez_compressed(O/'structural.npz',**{k:v for k,v in saved.items() if isinstance(v,np.ndarray)});np.save(O/'parameters.npy',z);np.save(O/'solver_final.npy',x);np.save(O/'augmented_final.npy',augmented)
    (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file()},indent=2));print(json.dumps({k:result[k] for k in ['optimizer','algorithm','event_clock','window']},indent=2),flush=True)
if __name__=='__main__':main()

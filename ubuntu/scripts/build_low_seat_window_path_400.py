"""ONE bounded mixed-integer nonlinear-feasibility run for a COMPLETE115-knot path.
Unlike394's one tangent step, each proposed whole-path/event-budget step is accepted
only against the actual nonlinear maximum normalized violation. The continuous20ms
window envelope controls target extrema between clock samples while event budgets
remain integer. Saved388 is an infeasible seed, not a resumed old optimization.
Maximum12 linear subproblems,60s/1000nodes each; no extension or post-run polishing.
The epigraph is an algorithmic merit value; positive slack NEVER admits a path.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,time
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_array
from low_seat_window_envelope_398 import make_window
from low_seat_event_clock_v2_386 import GROUPS,clock_samples,clock_rows
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'window_path_400'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def merit(eq,iq,keep,scale):return float(max(0.,np.max(abs(eq)),np.max(-iq[keep]/scale[keep])))
def main():
    O.mkdir(exist_ok=False)
    prep=json.loads((D/'window_preparation_398/preparation.json').read_text());audit=json.loads((D/'window_preparation_audit_399/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
    hashes={**prep['source_input_sha256'],**audit['source_sha256']}
    hashes.update({str(p.resolve()):sha(p) for p in [Path(__file__),Path('scripts/audit_low_seat_window_path_401.py'),Path('scripts/export_low_seat_support_373.py')]})
    for p,h in hashes.items():assert sha(p)==h,p
    d=make_window();p=d['p'];v=d['initial'].copy();keep=d['keep'];scale=d['scale'];rad=np.r_[np.full(786,.02),np.full(109,.1),np.full(5,5.)]
    radius=1.;history=[];started=time.monotonic();stop='12-subproblem cap';accepted=0
    # These tolerances concern the OFFLINE HiGHS numerical optimizer, NOT PhysX.
    # Installed SciPy1.17 forwards them; installed HighsOptions exposes all three.
    # They are stricter than the unchanged independent1e-7 primal acceptance bound.
    options=dict(time_limit=60,node_limit=1000,mip_rel_gap=0.,mip_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
    revision=dict(scope=__doc__,source_input_sha256=hashes,maximum_subproblems=12,maximum_trial_evaluations=12,maximum_subproblem_wall_s=60,maximum_subproblem_nodes=1000,initial_trust_scale=1.,coordinate_trust_units=rad.tolist(),acceptance_ratio_minimum=.1,shrink_on_rejection=.5,expand_if_ratio_above=.75,expansion_factor=2.,maximum_trust_scale=1.,minimum_trust_scale=1/128,merit='maximum normalized violation of every retained nonlinear row and both signs of hover equality; no changed physical thresholds',linear_solver_options=options,unchanged_independent_linear_acceptance_bound=1e-7,nonlinear_algorithm_stop_merit=1e-10,all_final_original_and_strict_screens_still_required=True,original_rows=20284,continuous_window_rows=7424,total_rows=27708,omitted_constant_contact_rows=392,omitted_total_time_identity=1,clock_rows_all_final_checked=True,actual_float32_filter_required_later=True,full_geometry_mandatory_final_screen=True,physical_admission=False)
    (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'initial.npz',**dict(np.load(D/'event_path_388/initial.npz')))
    for iteration in range(12):
        eq,iq=d['evaluate'](v);current=merit(eq,iq,keep,scale)
        if current<=1e-10:stop='nonlinear merit target reached; strict final gates remain';break
        E,I=d['evaluate'](v,True);m=int(keep.sum());A=np.zeros((m+3,901));lower=np.full(m+3,-np.inf);upper=np.full(m+3,np.inf)
        A[:m,:900]=I[keep]/scale[keep,None];A[:m,900]=1.;lower[:m]=-iq[keep]/scale[keep]
        A[m,:900]=E[0];A[m,900]=1.;lower[m]=-eq[0]
        A[m+1,:900]=-E[0];A[m+1,900]=1.;lower[m+1]=eq[0]
        A[m+2,895:900]=1.;lower[m+2]=upper[m+2]=0.
        blo=-rad*radius;bhi=rad*radius
        for k,(a,b) in enumerate(d['d']['bounds']):
            if a is not None:blo[k]=max(blo[k],a-v[k])
            if b is not None:bhi[k]=min(bhi[k],b-v[k])
        for g,(a,b) in enumerate(GROUPS):blo[895+g]=max(blo[895+g],b-a+1-v[895+g])
        blo[898]=max(blo[898],101-v[898]);blo=np.r_[blo,0.];bhi=np.r_[bhi,current+1e-12]
        assert np.all(blo<=bhi)
        integrality=np.r_[np.zeros(895,int),np.ones(5,int),0];cost=np.r_[np.zeros(900),1.]
        folder=O/f'step_{iteration:02d}';folder.mkdir()
        np.savez_compressed(folder/'linear_problem.npz',v=v,E=E,I=I,eq=eq,ineq=iq,A=A,lower=lower,upper=upper,bounds_lower=blo,bounds_upper=bhi,integrality=integrality,cost=cost,radius=radius)
        solverstart=time.monotonic();fit=milp(cost,integrality=integrality,bounds=Bounds(blo,bhi),constraints=LinearConstraint(csc_array(A),lower,upper),options=options.copy())
        record=dict(iteration=iteration,elapsed_s=time.monotonic()-started,trust_scale=radius,current_merit=current,solver_status=int(fit.status),solver_success=bool(fit.success),solver_message=str(fit.message),solver_wall_s=time.monotonic()-solverstart,mip_nodes=getattr(fit,'mip_node_count',None),mip_gap=getattr(fit,'mip_gap',None),candidate_available=fit.x is not None,accepted=False)
        if fit.x is None:
            stop='linear subproblem returned no candidate';record['termination']=stop;history.append(record);(folder/'result.json').write_text(json.dumps(record,indent=2));break
        step=np.asarray(fit.x);np.save(folder/'solution.npy',step);Av=A@step
        linear_min=float(min((Av-lower)[np.isfinite(lower)].min(),(upper-Av)[np.isfinite(upper)].min()));bound_min=float(min((step-blo).min(),(bhi-step).min()));integer_error=float(abs(step[895:900]-np.rint(step[895:900])).max())
        record.update(linear_minimum=linear_min,bound_minimum=bound_min,integer_decode_error=integer_error,predicted_merit=float(step[-1]),linear_numerical_passed=bool(linear_min>=-1e-7 and bound_min>=-1e-7 and integer_error<1e-7))
        if not record['linear_numerical_passed']:
            stop='independent linear primal threshold failed';record['termination']=stop;history.append(record);(folder/'result.json').write_text(json.dumps(record,indent=2));break
        trial=v+step[:900];trial[895:900]=v[895:900]+np.rint(step[895:900]);assert trial[895:].sum()==650 and trial[898]>=101
        np.save(folder/'trial.npy',trial)
        # Domain failure terminates this one run; no recovery or hidden polishing.
        try:te,ti=d['evaluate'](trial)
        except RuntimeError as exc:
            stop='bounded nonlinear domain failure';record.update(termination=stop,error=str(exc));history.append(record);(folder/'result.json').write_text(json.dumps(record,indent=2));break
        actual=merit(te,ti,keep,scale);predicted=current-float(step[-1]);reduction=current-actual
        ratio=float(reduction/predicted) if predicted>0 else None
        accept=bool(predicted>0 and reduction>0 and ratio>=.1)
        np.savez_compressed(folder/'trial_constraints.npz',eq=te,ineq=ti,X=d['last']['X'],Y=d['last']['Y'])
        record.update(trial_merit=actual,predicted_reduction=predicted,actual_reduction=reduction,acceptance_ratio=ratio,accepted=accept,trial_budgets=trial[895:].tolist(),maximum_nonlinear_minus_linear_scaled=float(np.max(abs(ti-iq-I@step[:900])/scale)))
        if accept:
            v=trial;accepted+=1
            if ratio>.75:radius=min(1.,2*radius)
        else:radius*=.5
        history.append(record);(folder/'result.json').write_text(json.dumps(record,indent=2));(O/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(record),flush=True)
        if radius<1/128:stop='trust radius floor reached';break
    eq,iq=d['evaluate'](v);saved=dict(d['last']);finalmerit=merit(eq,iq,keep,scale);budgets=np.rint(v[895:]).astype(int)
    samples,loc=clock_samples(saved['C'],saved['dt'],budgets);_,craw,cfull,_,targets=clock_rows(samples,p['command_lo'],p['command_hi'],p['rates'],d['ulp'])
    hard=np.r_[d['d']['hard'][:20284],np.zeros(7424,bool)]
    # The old hard/soft categories below are diagnostic labels ONLY; this run's
    # feasibility merit includes ALL retained rows and BOTH hover signs together.
    opt=dict(success=bool(finalmerit<=1e-10),message=stop,iterations=len(history),evaluations=len(history)+1,accepted_steps=accepted,minimum_augmented_inequality=float((iq[keep]/scale[keep]+finalmerit).min()),hard_soft_labels_diagnostic_only=True)
    ep=dict(p);ep['evaluate']=lambda x:(eq,iq)
    result=export_result(ep,saved['x'],opt,revision,hashes,O,finalmerit,scale,keep,hard)
    result.update(candidate_available=True,algorithm=dict(stop=stop,subproblems=len(history),accepted_steps=accepted,maximum_subproblems=12,final_merit=finalmerit,history=history),event_clock=dict(event_budgets=budgets.tolist(),total_ticks=675,integer_time_s=13.5,minimum_original_clock_rate_row=float(craw[:,:32].min()),minimum_original_clock_target_row=float(craw[:,32:].min()),all_tightened_clock_rate_rows_passed=bool(np.all(cfull[:,:32]>=0)),all_tightened_clock_target_rows_passed=bool(np.all(cfull[:,32:]>=0)),actual_float32_schedule=False,physical_admission=False),window=dict(breakpoints=232,minimum_original_target_row=float(saved['env']['raw'].min()),minimum_tightened_target_row=float(saved['env']['tightened'].min()),all_strict_window_rows_passed=bool(np.all(saved['env']['tightened']>=0))))
    np.savez_compressed(O/'event_clock.npz',parameters=v[:895],budgets=budgets,raw_ineq=saved['raw'],tighten=saved['tighten'],clock=samples,clock_raw=craw,clock_full=cfull,locations=loc,targets=targets,ulp=d['ulp'])
    np.savez_compressed(O/'window.npz',**saved['env']);np.save(O/'parameters.npy',v);(O/'history.json').write_text(json.dumps(history,indent=2));(O/'results.json').write_text(json.dumps(result,indent=2))
    (O/'SHA256.json').write_text(json.dumps({str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file()},indent=2));print(json.dumps({k:result[k] for k in ['optimizer','algorithm','event_clock','window']},indent=2),flush=True)
if __name__=='__main__':main()

"""ONE bounded complete-path structural/hard-feasibility algorithm.

All positive-load, cumulative-phase and prescribed-normal-gap coordinates vary
jointly with all five integer event budgets and interior durations. Original hard
rows receive NO feasibility slack and every accepted nonlinear trial must satisfy
them without a signed-row tolerance. Only command/rate/headroom/window rows use the
merit epigraph. The constructed hover equation is checked at the unchanged1e-8
numerical bound. No finite-difference velocity, timer release or physics occurs.

Maximum16 MILP subproblems,60s/1000nodes each. No extension/polishing. Coordinate
trust bounds are imposed on the old physical construction coordinates via the
audited map Jacobian, plus an explicit one-natural-log-unit local coordinate box.
No manual weight/trust/allocation sweep; full final strict gates remain mandatory.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,time
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import csc_array
from low_seat_structural_coordinates_404 import make_structural,FLOOR
from low_seat_event_clock_v2_386 import GROUPS,clock_samples,clock_rows
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'structural_path_406'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def merit(iq,mask,scale):return float(max(0.,np.max(-iq[mask]/scale[mask])))
def main():
    O.mkdir(exist_ok=False);prep=json.loads((D/'structural_preparation_404/preparation.json').read_text());audit=json.loads((D/'structural_preparation_audit_v2_405/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
    hashes={**prep['source_input_sha256'],**audit['source_sha256']}
    hashes.update({str(q.resolve()):sha(q) for q in [Path(__file__),Path('scripts/audit_low_seat_structural_path_407.py'),Path('scripts/audit_low_seat_structural_steps_407.py'),Path('scripts/export_low_seat_support_373.py')]})
    for path,h in hashes.items():assert sha(path)==h,path
    d=make_structural();w=d['d'];p=d['p'];z=d['initial'].copy();n=len(z);assert n==895
    scale=w['scale'];keep=w['keep'];hard=np.r_[w['d']['hard'][:20284],np.zeros(7424,bool)];hardmask=hard&keep;softmask=~hard&keep
    budget_start=d['budget_slice'].start;oldrad=np.r_[np.full(786,.02),np.full(109,.1),np.full(5,5.)]
    def coordinate_margin(v):
        return float(min([v[j]-a for j,(a,b) in enumerate(w['d']['bounds']) if a is not None]+[b-v[j] for j,(a,b) in enumerate(w['d']['bounds']) if b is not None]))
    newrad=np.r_[np.ones(d['timing_slice'].start),np.full(109,.1),np.full(5,5.)]
    options=dict(time_limit=60,node_limit=1000,mip_rel_gap=0.,mip_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9)
    revision=dict(scope=__doc__,source_input_sha256=hashes,maximum_subproblems=16,maximum_trial_evaluations=16,maximum_subproblem_wall_s=60,maximum_subproblem_nodes=1000,initial_trust_scale=1.,old_coordinate_trust_units=oldrad.tolist(),new_coordinate_trust_units=newrad.tolist(),acceptance_ratio_minimum=.1,shrink_on_rejection=.5,expand_if_ratio_above=.75,expansion_factor=2.,maximum_trust_scale=1.,minimum_trust_scale=1/128,merit='maximum normalized violation of only command/rate/headroom/window rows; no slack on original hard rows',hard_actual_acceptance='every retained signed hard row >=0; constructed hover abs(eq)<=unchanged1e-8; physical admission still separate',linear_solver_options=options,unchanged_independent_linear_acceptance_bound=1e-7,nonlinear_algorithm_stop_merit=1e-10,all_final_original_and_strict_screens_still_required=True,new_structural_floor=FLOOR,variables=n,original_rows=20284,continuous_window_rows=7424,total_rows=27708,omitted_constant_contact_rows=392,omitted_total_time_identity=1,hover_removed_from_linear_optimizer_only_because_of_implicit_construction=True,posthover_phase_order_still_explicit_hard=True,actual_float32_filter_required_later=True,full_geometry_mandatory_final_screen=True,physical_admission=False)
    (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'initial.npz',**dict(np.load(D/'window_path_400/initial.npz')))
    eq,iq=d['evaluate'](z);seed=dict(np.load(D/'structural_preparation_404/fixture.npz'));assert max(abs(d['last']['X']-seed['X']).max(),abs(iq-seed['iq']).max())<1e-12
    assert np.all(iq[hardmask]>=0) and abs(eq).max()<=1e-8 and coordinate_margin(d['last']['v'])>=0
    np.savez_compressed(O/'structural_initial.npz',z=z,eq=eq,ineq=iq,X=d['last']['X'],Y=d['last']['Y'],hard=hard,keep=keep,scale=scale)
    trust=1.;history=[];accepted=0;started=time.monotonic();stop='16-subproblem cap'
    for iteration in range(16):
        eq,iq=d['evaluate'](z);current=merit(iq,softmask,scale)
        assert np.all(iq[hardmask]>=0) and abs(eq).max()<=1e-8 and coordinate_margin(d['last']['v'])>=0
        if current<=1e-10:stop='soft merit target reached; all strict final gates remain';break
        E,I=d['evaluate'](z,True);G=d['last']['G'];v=d['last']['v'];m=int(keep.sum())
        A=np.zeros((m+901,n+1));low=np.full(m+901,-np.inf);high=np.full(m+901,np.inf)
        A[:m,:n]=I[keep]/scale[keep,None];A[:m,n]=(~hard[keep]).astype(float);low[:m]=-iq[keep]/scale[keep]
        A[m:m+900,:n]=G;low[m:m+900]=-oldrad*trust;high[m:m+900]=oldrad*trust
        for j,(a,b) in enumerate(w['d']['bounds']):
            if a is not None:low[m+j]=max(low[m+j],a-v[j])
            if b is not None:high[m+j]=min(high[m+j],b-v[j])
        A[m+900,budget_start:n]=1.;low[-1]=high[-1]=0.
        blo=-newrad*trust;bhi=newrad*trust
        for g,(a,b) in enumerate(GROUPS):blo[budget_start+g]=max(blo[budget_start+g],b-a+1-z[budget_start+g])
        blo[budget_start+3]=max(blo[budget_start+3],101-z[budget_start+3]);blo=np.r_[blo,0.];bhi=np.r_[bhi,current+1e-12]
        assert np.all(blo<=bhi)
        integrality=np.r_[np.zeros(budget_start,int),np.ones(5,int),0];cost=np.r_[np.zeros(n),1.]
        folder=O/f'step_{iteration:02d}';folder.mkdir();np.savez_compressed(folder/'linear_problem.npz',z=z,v=v,G=G,E=E,I=I,eq=eq,ineq=iq,A=A,lower=low,upper=high,bounds_lower=blo,bounds_upper=bhi,integrality=integrality,cost=cost,radius=trust)
        ts=time.monotonic();fit=milp(cost,integrality=integrality,bounds=Bounds(blo,bhi),constraints=LinearConstraint(csc_array(A),low,high),options=options.copy())
        rec=dict(iteration=iteration,elapsed_s=time.monotonic()-started,trust_scale=trust,current_merit=current,solver_status=int(fit.status),solver_success=bool(fit.success),solver_message=str(fit.message),solver_wall_s=time.monotonic()-ts,mip_nodes=getattr(fit,'mip_node_count',None),mip_gap=getattr(fit,'mip_gap',None),candidate_available=fit.x is not None,accepted=False)
        if fit.x is None:
            stop='linear subproblem returned no candidate; no infeasibility certificate';rec['termination']=stop;history.append(rec);(folder/'result.json').write_text(json.dumps(rec,indent=2));break
        step=np.asarray(fit.x);np.save(folder/'solution.npy',step);av=A@step;lin=float(min((av-low)[np.isfinite(low)].min(),(high-av)[np.isfinite(high)].min()));bound=float(min((step-blo).min(),(bhi-step).min()));integer=float(abs(step[budget_start:n]-np.rint(step[budget_start:n])).max());passed=bool(lin>=-1e-7 and bound>=-1e-7 and integer<1e-7)
        rec.update(linear_minimum=lin,bound_minimum=bound,integer_decode_error=integer,linear_numerical_passed=passed,predicted_merit=float(step[-1]))
        if not passed:
            stop='independent linear primal threshold failed';rec['termination']=stop;history.append(rec);(folder/'result.json').write_text(json.dumps(rec,indent=2));break
        trial=z+step[:n];trial[budget_start:n]=z[budget_start:n]+np.rint(step[budget_start:n]);assert trial[budget_start:n].sum()==650 and trial[budget_start+3]>=101;np.save(folder/'trial.npy',trial)
        try:te,ti=d['evaluate'](trial)
        except RuntimeError as exc:
            stop='bounded structural/root domain failure';rec.update(termination=stop,error=str(exc));history.append(rec);(folder/'result.json').write_text(json.dumps(rec,indent=2));break
        actual=merit(ti,softmask,scale);pred=current-float(step[-1]);reduction=current-actual;ratio=float(reduction/pred) if pred>0 else None
        actual_coordinate_margin=coordinate_margin(d['last']['v'])
        actualhard=bool(np.all(ti[hardmask]>=0) and abs(te).max()<=1e-8 and actual_coordinate_margin>=0)
        accept=bool(actualhard and pred>0 and reduction>0 and ratio>=.1)
        np.savez_compressed(folder/'trial_constraints.npz',eq=te,ineq=ti,X=d['last']['X'],Y=d['last']['Y'],loads=d['last']['loads'],targets=d['last']['targets'],v=d['last']['v'])
        rec.update(trial_merit=actual,predicted_reduction=pred,actual_reduction=reduction,acceptance_ratio=ratio,hard_nonlinear_passed=actualhard,minimum_actual_hard_row=float(ti[hardmask].min()),actual_coordinate_bound_margin=actual_coordinate_margin,actual_hover_residual=float(abs(te).max()),accepted=accept,trial_budgets=trial[budget_start:n].tolist())
        if accept:
            z=trial;accepted+=1
            if ratio>.75:trust=min(1.,2*trust)
        else:trust*=.5
        history.append(rec);(folder/'result.json').write_text(json.dumps(rec,indent=2));(O/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(rec),flush=True)
        if trust<1/128:stop='trust radius floor reached';break
    eq,iq=d['evaluate'](z);saved=dict(d['last']);finalmerit=merit(iq,softmask,scale);budgets=np.rint(z[budget_start:n]).astype(int);samples,loc=clock_samples(saved['C'],saved['dt'],budgets);_,craw,cfull,_,targets=clock_rows(samples,p['command_lo'],p['command_hi'],p['rates'],w['ulp'])
    actual_trials=sum('trial_merit' in r for r in history)
    opt=dict(success=bool(finalmerit<=1e-10 and np.all(iq[hardmask]>=0) and abs(eq).max()<=1e-8),message=stop,iterations=len(history),evaluations=actual_trials,accepted_steps=accepted,minimum_augmented_inequality=float((iq[keep]/scale[keep]+np.where(hard[keep],0.,finalmerit)).min()),hard_soft_labels_diagnostic_only=False)
    ep=dict(p);ep['evaluate']=lambda x:(eq,iq);result=export_result(ep,saved['x'],opt,revision,hashes,O,finalmerit,scale,keep,hard)
    result.update(candidate_available=True,algorithm=dict(stop=stop,subproblems=len(history),actual_nonlinear_trials=actual_trials,accepted_steps=accepted,maximum_subproblems=16,final_merit=finalmerit,history=history),event_clock=dict(event_budgets=budgets.tolist(),total_ticks=675,integer_time_s=13.5,minimum_original_clock_rate_row=float(craw[:,:32].min()),minimum_original_clock_target_row=float(craw[:,32:].min()),all_tightened_clock_rate_rows_passed=bool(np.all(cfull[:,:32]>=0)),all_tightened_clock_target_rows_passed=bool(np.all(cfull[:,32:]>=0)),actual_float32_schedule=False,physical_admission=False),window=dict(breakpoints=232,minimum_original_target_row=float(saved['env']['raw'].min()),minimum_tightened_target_row=float(saved['env']['tightened'].min()),all_strict_window_rows_passed=bool(np.all(saved['env']['tightened']>=0))))
    core=w['last'];np.savez_compressed(O/'event_clock.npz',parameters=saved['v'][:895],budgets=budgets,raw_ineq=core['raw'],tighten=core['tighten'],clock=samples,clock_raw=craw,clock_full=cfull,locations=loc,targets=targets,ulp=w['ulp'])
    np.savez_compressed(O/'window.npz',**saved['env']);np.savez_compressed(O/'structural.npz',**{k:v for k,v in saved.items() if isinstance(v,np.ndarray)});np.save(O/'parameters.npy',z);(O/'history.json').write_text(json.dumps(history,indent=2));(O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file()},indent=2));print(json.dumps({k:result[k] for k in ['optimizer','algorithm','event_clock','window']},indent=2),flush=True)
if __name__=='__main__':main()

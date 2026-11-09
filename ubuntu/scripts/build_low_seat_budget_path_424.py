"""ONE bounded direct nonlinear integer-event feasibility tree, original406 seed.

All894 continuous path coordinates retain the EXACT416 domain. Four event
budgets are integer variables in the previously used +/-5 tick box; the fifth
is650-sum(first4), with normal>=101 and load>=11. Static command limits and all
previous hard constraints stay hard. Only rate/headroom/window rows have slack.
At most FIVE SLSQP nodes,30 iterations each, breadth-first lower branch first.
Branch only an augmented-feasible, zero-slack numerical relaxation (1e-8), on
its most fractional independent budget. A near-integer relaxation (1e-7) gets
ONE exact-budget fixing node, never an admitted rounded result. An unresolved
local NLP is reported UNRESOLVED, never an infeasibility or global bound proof.
No callbacks selected. If no exact-integer strict point is found, the ROOT FINAL
is exported as an explicitly fractional/nonadmitted research diagnostic. No418
result is a seed, no old optimizer is continued, no post-tree polishing occurs.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,time
from scipy.optimize import minimize,Bounds
from low_seat_monotone_phase_410 import make_phase,FLOOR
from low_seat_budget_domain_422 import *
from export_low_seat_support_373 import export_result
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'budget_path_424'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);prep=json.loads((D/'budget_preparation_422/preparation.json').read_text());audit=json.loads((D/'budget_preparation_audit_423/independent_audit.json').read_text());launch=json.loads((D/'budget_path_launch_424.json').read_text());assert prep['passed'] and audit['passed']
    hashes=launch['source_input_sha256']
    for p,h in hashes.items():assert sha(p)==h,p
    f=dict(np.load(D/'budget_preparation_422/fixture.npz'));base=dict(np.load(D/'window_path_400/initial.npz'));d=make_phase();w=d['d'];p=d['p'];seed=d['initial'].copy();assert np.array_equal(seed,f['seed'])
    hard=partition();keep=w['keep'];scale=w['scale'];lo=f['lo'];hi=f['hi'];lower,upper=domain(seed);x0=np.r_[seed[:898],float(f['slack'])];lower=np.r_[lower,0.];upper=np.r_[upper,float(f['slack'])]
    eq,iq=d['evaluate'](seed);assert np.array_equal(assemble(iq,d['last']['v'],seed,x0[-1],scale,keep,lo,hi),f['values'])
    revision=dict(scope=__doc__,seed='original406 through unchanged410; not418',maximum_nonlinear_nodes=5,maximum_iterations_per_node=30,maximum_total_iterations=150,optimizer='SLSQP',ftol=1e-10,continuous_path_coordinates=894,independent_integer_budgets=4,dependent_budget='650-sum(first4)',budget_lower=BUDGET_LO.tolist(),budget_upper=BUDGET_HI.tolist(),numerical_branching_gate=1e-8,near_integer_fixing_trigger=1e-7,strict_final_criteria_unchanged=True,continuous_domain_exactly416=True,hard_rows=int((hard&keep).sum()),soft_rows=int((~hard&keep).sum()),optimizer_rows=len(f['values']),original_support_window_rows=27708,extra_coordinate_bounds=1572,extra_dependent_budget_bounds=2,full_geometry_mandatory_final_screen=True,source_input_sha256=hashes,local_NLP_failure_not_infeasibility=True,callback_selection=False,actual_float32_schedule=False,physical_admission=False)
    (O/'revision.json').write_text(json.dumps(revision,indent=2));np.savez_compressed(O/'initial.npz',**base);np.savez_compressed(O/'structural_initial.npz',z=seed,eq=eq,ineq=iq,X=d['last']['X'],Y=d['last']['Y'],hard=hard,keep=keep,scale=scale,lo=lo,hi=hi,lower=lower,upper=upper,initial=x0)
    started=time.monotonic();queue=[dict(parent=None,kind='root_relaxation',lower=lower,upper=upper,initial=x0)];records=[];root_final=None;selected=None;stop='node budget exhausted';request_counts=dict(constraint_requests=0,jacobian_requests=0,primal_cache_misses=0)
    def evaluate(x,derivative=False):
        z=expand(x[:898]);request_counts['jacobian_requests' if derivative else 'constraint_requests']+=1
        if 'z' not in d['last'] or not np.array_equal(z,d['last']['z']):request_counts['primal_cache_misses']+=1
        return d['evaluate'](z,derivative)
    while queue and len(records)<5:
        node=queue.pop(0);index=len(records);folder=O/('node_%02d'%index);folder.mkdir();L=node['lower'];U=node['upper'];start=node['initial'].copy();assert np.all(start>=L) and np.all(start<=U)
        free=np.flatnonzero(L<U);fixed=np.flatnonzero(L==U);assert np.all(start[fixed]==L[fixed]);history=[]
        def full(y):
            x=start.copy();x[free]=y;return x
        def constraints(y):
            x=full(y);ee,ii=evaluate(x);return assemble(ii,d['last']['v'],expand(x[:898]),x[-1],scale,keep,lo,hi)
        def jacobian(y):
            x=full(y);ee,ii=evaluate(x);E,I=evaluate(x,True);return assemble(ii,d['last']['v'],expand(x[:898]),x[-1],scale,keep,lo,hi,I,d['last']['G'])[1][:,free]
        def callback(y):
            x=full(y);ee,ii=evaluate(x);s=d['last'];aug=assemble(ii,s['v'],expand(x[:898]),x[-1],scale,keep,lo,hi);ci=len(history)
            np.savez_compressed(folder/('callback_%02d.npz'%ci),solver=x,z=expand(x[:898]),eq=ee,ineq=ii,v=s['v'])
            rec=dict(callback=ci,slack=float(x[-1]),soft_merit=float(max(0.,np.max(-ii[keep&~hard]/scale[keep&~hard]))),minimum_hard=float(ii[keep&hard].min()),minimum_augmented=float(aug.min()),budgets=expand(x[:898])[-5:].tolist(),not_selected=True)
            history.append(rec);(folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(node=index,elapsed_s=time.monotonic()-started,**rec)),flush=True)
        np.savez_compressed(folder/'initial.npz',solver=start,lower=L,upper=U,free=free,fixed=fixed)
        try:
            fit=minimize(lambda y:float(full(y)[-1]),start[free],jac=lambda y:(free==898).astype(float),method='SLSQP',bounds=Bounds(L[free],U[free]),constraints=[dict(type='ineq',fun=constraints,jac=jacobian)],callback=callback,options=dict(maxiter=30,ftol=1e-10,disp=True))
        except Exception as exc:
            (O/'termination.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),node=index,error_type=type(exc).__name__,error=str(exc),completed_nodes=len(records),restarted=False,physical_admission=False),indent=2));raise
        x=full(fit.x);ee,ii=evaluate(x);s=dict(d['last']);z=expand(x[:898]);aug=assemble(ii,s['v'],z,x[-1],scale,keep,lo,hi);soft=float(max(0.,np.max(-ii[keep&~hard]/scale[keep&~hard])));boxmin=float(min((x-L).min(),(U-x).min()));cm=float(min((s['v'][:786]-lo).min(),(hi-s['v'][:786]).min()));bg=z[-5:];frac=np.abs(bg[:4]-np.rint(bg[:4]));exact=bool(np.array_equal(bg,np.rint(bg)))
        numerical=bool(aug.min()>=-1e-8 and soft<=1e-8 and x[-1]<=1e-8 and abs(ee).max()<=1e-8 and boxmin>=0)
        strict=bool(exact and np.all(ii[keep]>=0) and cm>=0 and np.all(bg>=BUDGET_LO) and np.all(bg<=BUDGET_HI) and abs(ee).max()<=1e-8 and boxmin>=0)
        decision='unresolved_local_relaxation';branch=None
        if strict:decision='exact_integer_strict_model_point';selected=index;stop=decision
        elif numerical and node['kind']!='exact_budget_fixing':
            if frac.max()<=1e-7:
                integers=np.rint(bg[:4]);fifth=650.-integers.sum()
                if np.all(integers>=L[894:898]) and np.all(integers<=U[894:898]) and 11<=fifth<=21:
                    nl=L.copy();nu=U.copy();nl[894:898]=integers;nu[894:898]=integers;ns=x.copy();ns[894:898]=integers;queue.append(dict(parent=index,kind='exact_budget_fixing',lower=nl,upper=nu,initial=ns));decision='queued_exact_fixing_recheck'
                else:decision='near_integer_fixing_outside_domain'
            else:
                g=int(np.argmax(frac));at=894+g;floor=float(np.floor(x[at]));ceil=floor+1;branch=dict(budget=g,value=float(x[at]),floor=floor,ceil=ceil);decision='branched_fractional_relaxation'
                for side in ['lower','upper']:
                    nl=L.copy();nu=U.copy()
                    if side=='lower':nu[at]=floor
                    else:nl[at]=ceil
                    if nl[at]<=nu[at]:
                        ns=x.copy();ns[at]=floor if side=='lower' else ceil;queue.append(dict(parent=index,kind='branch_'+side,lower=nl,upper=nu,initial=ns))
        opt=dict(success=bool(fit.success),status=int(fit.status),message=str(fit.message),iterations=int(fit.nit),objective_nfev=int(fit.nfev),jacobian_njev=int(fit.njev))
        rec=dict(node=index,parent=node['parent'],kind=node['kind'],optimizer=opt,decision=decision,branch=branch,slack=float(x[-1]),soft_merit=soft,minimum_augmented=float(aug.min()),minimum_hard=float(ii[keep&hard].min()),coordinate_margin=cm,box_margin=boxmin,budgets=bg.tolist(),integer_error=float(abs(bg-np.rint(bg)).max()),numerical_branching_gate_passed=numerical,strict_model_point=strict,callback_count=len(history),history=history)
        np.savez_compressed(folder/'final.npz',solver=x,z=z,eq=ee,ineq=ii,v=s['v'],X=s['X'],Y=s['Y'],augmented=aug);(folder/'result.json').write_text(json.dumps(rec,indent=2));records.append(rec);(O/'nodes.json').write_text(json.dumps(records,indent=2));print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
        if index==0:root_final=x.copy()
        if selected is not None:break
    if selected is None:
        selected=0;x=root_final;stop='node cap reached with unresolved branches' if queue else 'no further justified branches; unresolved local NLP is not infeasibility'
    else:x=np.load(O/('node_%02d'%selected)/'final.npz')['solver'].copy()
    eq,iq=evaluate(x);saved=dict(d['last']);z=expand(x[:898]);budgets=z[-5:];exact=bool(np.array_equal(budgets,np.rint(budgets)));samples,times=full_clock(saved['C'],saved['dt']);difference=np.diff(samples,axis=0);targets=10*samples[1:]-9*samples[:-1];craw=np.c_[.02*p['rates']-difference,.02*p['rates']+difference,p['command_hi']-targets,targets-p['command_lo']];cfull=craw-np.r_[4*w['ulp'],4*w['ulp'],32*w['ulp'],32*w['ulp']]
    opt=dict(records[selected]['optimizer']);opt.update(evaluations=opt['objective_nfev'],jacobian_evaluations=opt['jacobian_njev']);ep=dict(p);ep['evaluate']=lambda _:(eq,iq)
    result=export_result(ep,saved['x'],opt,revision,hashes,O,float(x[-1]),scale,keep,hard)
    soft=float(max(0.,np.max(-iq[keep&~hard]/scale[keep&~hard])));aug=assemble(iq,saved['v'],z,x[-1],scale,keep,lo,hi)
    result.update(candidate_available=True,algorithm=dict(optimizer_calls=len(records),stop=stop,final_soft_merit=soft,exported_node=selected,export_rule='first strict exact-integer node, otherwise root final diagnostic',integer_model_point_found=bool(any(r['strict_model_point'] for r in records)),nodes=records,callback_count=sum(r['callback_count'] for r in records),callbacks_not_selected=True,request_counts=request_counts,request_count_scope='Wrapper requests and primal cache misses only; not total internal FK/root operations',minimum_augmented_row=float(aug.min()),minimum_coordinate_bound=float(min((saved['v'][:786]-lo).min(),(hi-saved['v'][:786]).min()))),event_clock=dict(event_budgets=budgets.tolist(),mandatory_events_exactly_integer=exact,event_integer_error=float(abs(budgets-np.rint(budgets)).max()),total_ticks=675,integer_time_s=13.5 if exact else None,shared_clock_diagnostic_only=not exact,minimum_original_clock_rate_row=float(craw[:,:32].min()),minimum_original_clock_target_row=float(craw[:,32:].min()),all_tightened_clock_rate_rows_passed=bool(np.all(cfull[:,:32]>=0)),all_tightened_clock_target_rows_passed=bool(np.all(cfull[:,32:]>=0)),actual_float32_schedule=False,physical_admission=False),window=dict(breakpoints=232,minimum_original_target_row=float(saved['env']['raw'].min()),minimum_tightened_target_row=float(saved['env']['tightened'].min()),all_strict_window_rows_passed=bool(np.all(saved['env']['tightened']>=0))))
    core=w['last'];np.savez_compressed(O/'event_clock.npz',parameters=saved['v'][:895],budgets=budgets,raw_ineq=core['raw'],tighten=core['tighten'],clock=samples,clock_raw=craw,clock_full=cfull,times=times,targets=targets,ulp=w['ulp']);np.savez_compressed(O/'window.npz',**saved['env']);np.savez_compressed(O/'structural.npz',**{k:v for k,v in saved.items() if isinstance(v,np.ndarray)});np.save(O/'parameters.npy',z);np.save(O/'solver_final.npy',x);np.save(O/'augmented_final.npy',aug)
    (O/'results.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(q.relative_to(O)):sha(q) for q in O.rglob('*') if q.is_file()},indent=2));print(json.dumps({k:result[k] for k in ['optimizer','event_clock','window']},indent=2),flush=True)
if __name__=='__main__':main()

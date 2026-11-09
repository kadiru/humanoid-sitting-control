"""ONE bounded two-stage whole-path feasibility construction with exact
interior witness certificates before any mapping/root. No old solver continued.
Primary epigraph directions are never executed. All actual hard/phase/normal/
hover/boxes remain mandatory, and final physics admission is always separate.
"""
import datetime
from low_seat_interior_execution_484 import *
O=D/'interior_path_484'

def main():
    launch=read(D/'interior_path_launch_484.json');assert launch['compile_actual_exit']==0 and launch['dependencies_actual_exit_verified']
    for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
    O.mkdir(exist_ok=False);started=time.monotonic();d=load_saved();ret=InteriorRetraction(d);source=D/'free_gap_path_472'
    for p,h in read(source/'SHA256.json').items():assert sha(source/p)==h,p
    saved=dict(np.load(source/'initial.npz'));X,Y,J,target=[saved[k].copy() for k in ['X','Y','J','target']]
    drops=list(map(unpack,read(source/'initial_lift/drop_state.json')))
    assert read(source/'results.json')['accepted_steps']==0
    prior=np.load(source/'final.npz');assert np.array_equal(prior['X'],X) and np.array_equal(prior['Y'],Y)
    np.savez_compressed(O/'original440.npz',X=d['initialX'],Y=d['initialY'],J=d['initialJ'])
    (O/'seed_identity.json').write_text(json.dumps(dict(source=str(source/'initial.npz'),sha256=sha(source/'initial.npz'),audit=str(D/'free_gap_path_audit_473/SHA256.json'),audit_sha256=sha(D/'free_gap_path_audit_473/SHA256.json'),initial_root_repeated=False,rejected_trial_used=False,initial_infeasible=True),indent=2))
    (O/'initial_drop_state.json').write_text(json.dumps([pair(v) for v in drops]));_,r,e=values(d,X,Y)
    np.savez_compressed(O/'initial.npz',X=X,Y=Y,J=J,target=target,rows=r,equalities=e)
    seed=summary(d,X,Y,r,e);(O/'specification.json').write_text(json.dumps(dict(specification=SPEC,scope=__doc__,initial=seed,seed_is_not_assumed_feasible=True,primary_directions_never_decoded=True,positive_auxiliary_objective_not_admission=True),indent=2))
    history=[];lps=0;actual_lps=0;accepted=0;trust=1.;last=None;cert_failed=0;map_failed=0;stop='declared predictor cap';soft=d['a']['keep']&d['a']['soft']
    for outer in range(SPEC['max_predictors']):
        s=summary(d,X,Y,r,e);mu=s['soft_merit'];origin=d['C']@np.c_[X,Y].ravel();taken=False;record=dict(predictor=outer,trust=trust,start=s,attempts=[])
        a=assemble(d,X,Y,J,target,drops,trust);a['_d']=d
        sub=O/f'linear_{lps:03d}';res,lin,_=solve_stage(a,sub,'primary');lps+=1;actual_lps+=int(lin['actual_lp_call']);primary=dict(linear_index=lps-1,role='primary',linear=lin,accepted=False,never_mapped=True);record['primary']=primary
        good=res is not None and res.x is not None and lin.get('linear_pass',False)
        if good:
            theta=max(0.,float(-(a['r']+a['A']@res.x[:-1])[soft].min()));primary.update(modeled_soft=theta,predicted_reduction=mu-theta)
            good=theta<mu
        primary['eligible_for_interior_stage']=bool(good);(sub/'trial.json').write_text(json.dumps(primary,indent=2))
        if good:
            ceiling=(mu+theta)/2;predicted=None
            for correction in range(SPEC['max_corrections_per_predictor']+1):
                assert lps<SPEC['max_linear_programs'];sub=O/f'linear_{lps:03d}';res,lin,w=solve_stage(a,sub,'interior',ceiling);lps+=1;actual_lps+=int(lin['actual_lp_call'])
                at=dict(linear_index=lps-1,role='interior',correction=correction,linear=lin,soft_ceiling=ceiling,accepted=False);record['attempts'].append(at)
                if res is None or res.x is None or not lin.get('linear_pass',False):
                    at['rejection']='no independently feasible interior LP witness';(sub/'trial.json').write_text(json.dumps(at,indent=2));break
                tangent=res.x[:-1];certificate=certify(w['system'],tangent);certificate['lp_reported_eta']=float(res.x[-1]);(sub/'certificate.json').write_text(json.dumps(certificate,indent=2));at['certificate_pass']=certificate['passed']
                if not certificate['passed']:
                    cert_failed+=1;at['rejection']='nonpositive exact interior clearance; witness unchanged';(sub/'trial.json').write_text(json.dumps(at,indent=2));break
                if correction==0:
                    modeled=max(0.,float(-(a['r']+a['A']@tangent)[soft].min()));predicted=mu-modeled;at.update(certified_ray_modeled_soft=modeled,predicted_reduction=predicted)
                    if predicted<=0:
                        at['rejection']='certified direction has no modeled soft reduction';(sub/'trial.json').write_text(json.dumps(at,indent=2));break
                try:trial=ret.trial(a,tangent,sub)
                except ValueError as exc:
                    map_failed+=1;at.update(map_pass=False,rejection=str(exc));(sub/'trial.json').write_text(json.dumps(at,indent=2));break
                at.update(map_pass=True,root_pass=trial['passed'])
                if not trial['passed']:
                    at['rejection']=trial['reason'];(sub/'trial.json').write_text(json.dumps(at,indent=2));break
                tx,ty=trial['X'],trial['Y'];_,tr,te=values(d,tx,ty);state,valid,gaperr=accepted_state(d,tx,ty,tr,te,trial['target']);actual=mu-state['soft_merit'];ratio=actual/predicted;take=bool(valid and actual>0 and ratio>=SPEC['acceptance_ratio'])
                at.update(summary=state,total_normal_maximum=float(abs(te[:-1]).max()),hover_normalized_error=float(abs(te[-1])),actual_gap_root_residual=gaperr,actual_reduction=actual,predicted_reduction=predicted,ratio=ratio,accepted=take)
                np.savez_compressed(sub/'nonlinear_rows.npz',rows=tr,equalities=te);(sub/'trial.json').write_text(json.dumps(at,indent=2))
                if take:
                    X,Y,r,e,target,drops=tx,ty,tr,te,trial['target'],trial['drops'];J=ret.jac(X);accepted+=1;taken=True;last=lps-1
                    if ratio>.75:trust=min(1.,trust*2)
                    break
                if correction<SPEC['max_corrections_per_predictor']:
                    ceiling=mu-.5*predicted;tj=ret.jac(tx);a=assemble(d,tx,ty,tj,trial['target'],trial['drops'],trust,'correction',ceiling,origin=origin);a['_d']=d
        record.update(accepted=taken,elapsed_s=time.monotonic()-started);history.append(record)
        with (O/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        final=summary(d,X,Y,r,e);print(json.dumps(dict(predictor=outer,assembled_subproblems=lps,actual_lp_calls=actual_lps,accepted=taken,total_accepted=accepted,current=final,elapsed_s=time.monotonic()-started)),flush=True)
        if final['all_inequalities_pass'] and accepted_state(d,X,Y,r,e,target)[1]:stop='all modeled buffered rows passed';break
        if not taken:trust*=.5
        if trust<SPEC['minimum_trust_fraction']:stop='declared trust floor';break
    np.savez_compressed(O/'final.npz',X=X,Y=Y,J=J,target=target,rows=r,equalities=e,label_to_state=d['g']['label_to_state'],canonical=d['g']['canonical'])
    (O/'final_drop_state.json').write_text(json.dumps([pair(v) for v in drops]))
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,specification=SPEC,stop=stop,predictors=len(history),assembled_subproblems=lps,actual_lp_calls=actual_lps,linear_programs=lps,primary_lp_assemblies=len(history),interior_lp_assemblies=lps-len(history),nonlinear_trial_paths=ret.counts['trial_paths'],finite_maps=ret.counts['finite_maps'],certificate_failures=cert_failed,coordinate_map_failures=map_failed,accepted_steps=accepted,last_accepted_linear_index=last,initial=seed,final=summary(d,X,Y,r,e),wrapper_counts=ret.counts,seed_is_sealed472_initial=True,initial_root_repeated=False,primary_directions_never_mapped=True,sampled_full_geometry_pending=True,independent_audit_pending=True,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,elapsed_s=time.monotonic()-started)
    (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

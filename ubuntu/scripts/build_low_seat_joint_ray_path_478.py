"""ONE new bounded shared-affine-ray whole-path formulation.
Exact common alpha jointly respects all target/drop and complete linearized
f/p/load bounds, preserving the planned coupled direction before finite encoding.
Rate/window prediction is evaluated on that SAME scaled direction. Nonlinear
root/phase/full-box/hard/equality gates are unchanged and still decide acceptance.
The sealed472 INITIAL is an explicit infeasible research seed; no old optimizer,
rejected trial, root fixture, complete initial lift or finite-work loop is rerun.
"""
import datetime
from low_seat_joint_ray_execution_478 import *
O=D/'joint_ray_path_478'

def main():
 launch=read(D/'joint_ray_path_launch_478.json')
 assert launch['compile_actual_exit']==0 and launch['dependencies_actual_exit_verified']
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);start=time.monotonic();d=load_saved();ret=SharedRayRetraction(d)
 source=D/'free_gap_path_472';saved=dict(np.load(source/'initial.npz'))
 X,Y,J,target=saved['X'].copy(),saved['Y'].copy(),saved['J'].copy(),saved['target'].copy()
 drops=list(map(unpack,read(source/'initial_lift/drop_state.json')))
 for p,h in read(source/'SHA256.json').items():assert sha(source/p)==h,p
 assert read(source/'results.json')['accepted_steps']==0
 previous=np.load(source/'final.npz');assert np.array_equal(previous['X'],X) and np.array_equal(previous['Y'],Y)
 # Sealed initial state is inherited EXACTLY, not reconstructed or selected
 # from a rejected trial. Its actual J is valid only for this identical seed.
 np.savez_compressed(O/'original440.npz',X=d['initialX'],Y=d['initialY'],J=d['initialJ'])
 (O/'seed_identity.json').write_text(json.dumps(dict(source=str(source/'initial.npz'),sha256=sha(source/'initial.npz'),audit=str(D/'free_gap_path_audit_473/SHA256.json'),audit_sha256=sha(D/'free_gap_path_audit_473/SHA256.json'),initial_root_repeated=False,rejected_trial_used=False,initial_infeasible=True),indent=2))
 (O/'initial_drop_state.json').write_text(json.dumps([pair(v) for v in drops]))
 _,r,e=values(d,X,Y);lifted=dict(passed=True,reason='Inherited exact sealed472 INITIAL; no new root')
 seed=summary(d,X,Y,r,e);history=[];lps=0;accepted=0;trust=1.;last_accepted=None;map_failures=0
 np.savez_compressed(O/'initial.npz',X=X,Y=Y,J=J,target=target,rows=r,equalities=e)
 (O/'specification.json').write_text(json.dumps(dict(specification=SPEC,scope=__doc__,initial=seed,initial_lift_passed=lifted['passed'],initial_lift_reason=lifted['reason'],seed_is_not_assumed_feasible=True,old450_or462_trials_used=False,seed472_initial_inherited=True,finite_map_jacobian_claim=False,quantized_map_uses_continuous_zero_tangent_model=True,accepted_hard_minimum=0.,accepted_total_normal_bound=1e-13,accepted_hover_bound=1e-8,full_geometry_and_float32_are_separate_mandatory_gates=True),indent=2))
 stop='initial15-equation lift failed' if not lifted['passed'] else 'declared predictor cap'
 if lifted['passed']:
  for outer in range(SPEC['max_predictors']):
   current=summary(d,X,Y,r,e);merit=current['soft_merit'];origin=d['C']@np.c_[X,Y].ravel()
   a=assemble(d,X,Y,J,target,drops,trust);record=dict(predictor=outer,trust=trust,start=current,attempts=[]);taken=False;ceiling=None;predicted=None
   for correction in range(SPEC['max_corrections_per_predictor']+1):
    assert lps<SPEC['max_linear_programs'];sub=O/f'linear_{lps:02d}'
    res,lin=solve_new_lp(a,sub);lps+=1;attempt=dict(linear_index=lps-1,correction=correction,linear=lin);record['attempts'].append(attempt)
    if res is None or res.x is None or not lin.get('linear_pass',False):
     attempt.update(accepted=False,rejection='no independently feasible linear witness');(sub/'trial.json').write_text(json.dumps(attempt));break
    tangent=res.x[:-1]
    if correction==0:
     predicted=merit-float(res.x[-1]);ceiling=merit-.5*predicted
     if predicted<=0:
      attempt.update(accepted=False,rejection='no predicted rate/window improvement');(sub/'trial.json').write_text(json.dumps(attempt));break
    try:trial=ret.trial(a,tangent,sub)
    except ValueError as exc:
     map_failures+=1;attempt.update(accepted=False,map_pass=False,rejection=str(exc));(sub/'trial.json').write_text(json.dumps(attempt,indent=2));break
    attempt.update(map_pass=True,root_pass=trial['passed'])
    if not trial['passed']:
     attempt.update(accepted=False,rejection=trial['reason']);(sub/'trial.json').write_text(json.dumps(attempt,indent=2));break
    tx,ty=trial['X'],trial['Y'];_,tr,te=values(d,tx,ty)
    if correction==0:
     raw_prediction=predicted
     alpha=float(unpack(trial['map_record']['alpha']))
     modeled=a['r']+a['A']@(alpha*tangent)
     predicted=merit-max(0.,float(-modeled[d['a']['keep']&d['a']['soft']].min()))
     ceiling=merit-.5*predicted
     attempt.update(raw_lp_prediction=raw_prediction,shared_ray_alpha=alpha,shared_ray_predicted_reduction=predicted)
    s,valid,gaperr=accepted_state(d,tx,ty,tr,te,trial['target']);actual=merit-s['soft_merit'];ratio=actual/predicted if predicted>0 else float('-inf')
    take=bool(valid and actual>0 and ratio>=SPEC['acceptance_ratio'])
    attempt.update(summary=s,total_normal_maximum=float(abs(te[:-1]).max()),hover_normalized_error=float(abs(te[-1])),actual_gap_root_residual=gaperr,actual_reduction=actual,predicted_reduction=predicted,ratio=ratio,accepted=take)
    np.savez_compressed(sub/'nonlinear_rows.npz',rows=tr,equalities=te)
    (sub/'trial.json').write_text(json.dumps(attempt,indent=2))
    if take:
     X,Y,r,e,target,drops=tx,ty,tr,te,trial['target'],trial['drops'];J=ret.jac(X);accepted+=1;taken=True;last_accepted=lps-1
     if ratio>.75:trust=min(1.,trust*2)
     break
    if predicted<=0:break
    if correction<SPEC['max_corrections_per_predictor']:
     tj=ret.jac(tx);a=assemble(d,tx,ty,tj,trial['target'],trial['drops'],trust,'correction',ceiling,origin=origin)
   record.update(accepted=taken,elapsed_s=time.monotonic()-start);history.append(record)
   with (O/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
   final=summary(d,X,Y,r,e)
   print(json.dumps(dict(predictor=outer,linear_programs=lps,accepted=taken,total_accepted=accepted,current=final,elapsed_s=time.monotonic()-start)),flush=True)
   if final['all_inequalities_pass'] and accepted_state(d,X,Y,r,e,target)[1]:stop='all modeled buffered rows passed';break
   if not taken:trust*=.5
   if trust<SPEC['minimum_trust_fraction']:stop='declared trust floor';break
 np.savez_compressed(O/'final.npz',X=X,Y=Y,J=J,target=target,rows=r,equalities=e,label_to_state=d['g']['label_to_state'],canonical=d['g']['canonical'])
 (O/'final_drop_state.json').write_text(json.dumps([pair(v) for v in drops]))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,specification=SPEC,stop=stop,predictors=len(history),assembled_subproblems=lps,actual_lp_calls=sum(int(read(p)['status']!=-10) for p in O.glob('linear_*/solver.json')),linear_programs=lps,nonlinear_trial_paths=ret.counts['trial_paths'],finite_maps=ret.counts['finite_maps'],coordinate_map_failures=map_failures,accepted_steps=accepted,last_accepted_linear_index=last_accepted,initial=seed,initial_lift_pass=lifted['passed'],final=summary(d,X,Y,r,e),wrapper_counts=ret.counts,source440_unchanged=True,old450_or462_point_used=False,reduced_coordinates=d['B'].shape[1],new_final_differs_from_source=bool(not np.array_equal(X,d['initialX'])),seed_is_sealed472_initial=True,initial_root_repeated=False,sampled_full_geometry_pending=True,independent_audit_pending=True,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,elapsed_s=time.monotonic()-start)
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

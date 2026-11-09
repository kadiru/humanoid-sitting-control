"""ONE bounded whole-clock sparse predictor/corrector feasibility run.
Six predictors at most; up to two nonlinear defect corrections per predictor.
The complete new444/446v2 domain varies11906 phase/load coordinates jointly.
The infeasible440 seed is never admitted; only actual hard-feasible, monotone,
positive-load steps with nonlinear rate/window improvement may be accepted.
Correction LPs minimize displacement inside the SAME predictor domain, while
requiring half its predicted improvement. They cannot soften hard rows. No
post-cap polishing, alternate allocation, hidden root fallback or old run restart.
"""
import json,time,datetime
import numpy as np
from low_seat_collocation_solver_448 import *
O=D/'collocation_path_450'
def main():
 launch=read(D/'collocation_path_launch_450.json');assert launch['preparation448_actual_exit']==0 and launch['audit449_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 assert read(D/'collocation_solver_audit_449/independent_audit.json')['passed'];O.mkdir(exist_ok=False);start=time.monotonic();d=load_saved();X=d['initialX'].copy();Y=d['initialY'].copy();J=d['initialJ'].copy();ret=Retraction(d);_,r,e=values(d,X,Y);seed=summary(d,X,Y,r,e);history=[];lps=0;accepted=0;trust=1.;stop='predictor cap';last_accepted=None
 np.savez_compressed(O/'initial.npz',X=X,Y=Y,rows=r,equalities=e);(O/'specification.json').write_text(json.dumps(dict(specification=SPEC,scope=__doc__,initial=seed,correction_target='half of predicted soft-merit improvement',accepted_hard_minimum=0.,accepted_total_normal_bound=1e-13,accepted_hover_bound=1e-8,all_original_and_added_reserves_retained=True),indent=2))
 for outer in range(SPEC['max_predictors']):
  current=summary(d,X,Y,r,e);merit=current['soft_merit'];origin=X[:,d['c']['control']].ravel()[d['free']].copy();baseX=X.copy();baseY=Y.copy();baseJ=J.copy();a=assemble(d,X,Y,J,trust);step_record=dict(predictor=outer,trust=trust,start=current,attempts=[]);accepted_this=False;target=None;predicted=None
  for correction in range(SPEC['max_corrections_per_predictor']+1):
   sub=O/f'linear_{lps:02d}';res,lin=solve_lp(a,sub);lps+=1;attempt=dict(linear_index=lps-1,correction=correction,linear=lin);step_record['attempts'].append(attempt)
   if res is None or res.x is None or not lin.get('linear_pass',False):attempt['accepted']=False;attempt['rejection']='no independently feasible linear witness';break
   delta=res.x[:-1]
   if correction==0:
    predicted=merit-float(res.x[-1]);target=merit-.5*predicted
    if predicted<=0:attempt['accepted']=False;attempt['rejection']='no predicted soft improvement';break
   trial=ret.run(a['X'],a['T'],delta);np.savez_compressed(sub/'nonlinear_trial.npz',X=trial['X'],Y=trial['Y'],delta=delta);attempt['root_pass']=trial['passed'];attempt['root_history']=trial['root_history']
   if not trial['passed']:attempt['accepted']=False;attempt['rejection']=trial['reason'];break
   tx,ty=trial['X'],trial['Y'];_,tr,te=values(d,tx,ty);s=summary(d,tx,ty,tr,te);actual=merit-s['soft_merit'];ratio=actual/predicted;eq_ok=bool(np.max(abs(te[:-1]))<=1e-13 and abs(te[-1])<=1e-8)
   valid=bool(s['hard_pass'] and s['monotone_phase_pass'] and s['coordinate_bound_minimum']>=0 and s['fixed_states_exact'] and s['inactive_exact_zero'] and s['minimum_corner_weight']>=0 and eq_ok)
   take=bool(valid and actual>0 and ratio>=SPEC['acceptance_ratio']);attempt.update(summary=s,total_normal_maximum=float(abs(te[:-1]).max()),hover_normalized_error=float(abs(te[-1])),actual_reduction=actual,predicted_reduction=predicted,ratio=ratio,accepted=take)
   np.savez_compressed(sub/'nonlinear_rows.npz',rows=tr,equalities=te);(sub/'trial.json').write_text(json.dumps(attempt,indent=2))
   if take:
    X,Y,r,e=tx,ty,tr,te;J=ret.jac(X);accepted+=1;accepted_this=True;last_accepted=lps-1
    if ratio>.75:trust=min(1.,trust*2)
    break
   if correction<SPEC['max_corrections_per_predictor']:
    tj=ret.jac(tx);a=assemble(d,tx,ty,tj,trust,'correction',target,origin=origin)
  step_record.update(accepted=accepted_this,elapsed_s=time.monotonic()-start);history.append(step_record)
  with (O/'history.jsonl').open('a') as f:f.write(json.dumps(step_record)+'\n')
  print(json.dumps(dict(predictor=outer,linear_programs=lps,accepted=accepted_this,total_accepted=accepted,current=summary(d,X,Y,r,e),elapsed_s=time.monotonic()-start)),flush=True)
  final=summary(d,X,Y,r,e)
  if final['all_inequalities_pass'] and final['hard_pass'] and final['coordinate_bound_minimum']>=0 and abs(e[-1])<=1e-8 and np.max(abs(e[:-1]))<=1e-13:stop='all modeled buffered rows passed';break
  if not accepted_this:trust*=.5
  if trust<SPEC['minimum_trust_fraction']:stop='declared trust floor';break
 np.savez_compressed(O/'final.npz',X=X,Y=Y,J=J,rows=r,equalities=e,label_to_state=d['g']['label_to_state'],canonical=d['g']['canonical'])
 final=summary(d,X,Y,r,e);out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,specification=SPEC,stop=stop,predictors=len(history),linear_programs=lps,nonlinear_trial_paths=ret.counts['trial_paths'],accepted_steps=accepted,last_accepted_linear_index=last_accepted,initial=seed,final=final,wrapper_counts=ret.counts,source440_unchanged=True,new_final_differs_from_source=bool(not np.array_equal(X,d['initialX'])),sampled_full_geometry_pending=True,independent_audit_pending=True,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,elapsed_s=time.monotonic()-start)
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

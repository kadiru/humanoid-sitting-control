"""ONE new bounded-coordinate whole-path feasibility construction.
The immutable440 source initializes the entire joint clock domain. No450 point
is continued. New exponential/exact-binary-rational control retraction preserves
the unchanged absolute control box structurally. Full nonlinear normal/hover,
geometry-derived hard rows and original/reserved/buffered rate/target rows remain.
24 predictors/four corrections/120 LPs are the predeclared total budget. No
post-budget extension, hidden fallback, force/target clipping or callback choice.
"""
import json,time,datetime
import numpy as np
from low_seat_bounded_retraction_454 import *
O=D/'bounded_path_456'
def main():
 launch=read(D/'bounded_path_launch_456.json');assert launch['preparation454_actual_exit']==0 and launch['audit455_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 assert read(D/'bounded_retraction_audit_455/independent_audit.json')['passed']
 O.mkdir(exist_ok=False);start=time.monotonic();d=load_saved();X=d['initialX'].copy();Y=d['initialY'].copy();J=d['initialJ'].copy();ret=BoundedRetraction(d);_,r,e=values(d,X,Y);seed=summary(d,X,Y,r,e);history=[];lps=0;accepted=0;trust=1.;stop='declared predictor cap';last_accepted=None;map_failures=0
 np.savez_compressed(O/'initial.npz',X=X,Y=Y,rows=r,equalities=e)
 (O/'specification.json').write_text(json.dumps(dict(specification=SPEC,scope=__doc__,initial=seed,seed440_is_infeasible=True,seed450_used=False,normal_balance_is_explicit_nonlinear_equality=True,accepted_hard_minimum=0.,accepted_total_normal_bound=1e-13,accepted_hover_bound=1e-8),indent=2))
 for outer in range(SPEC['max_predictors']):
  current=summary(d,X,Y,r,e);merit=current['soft_merit'];origin=X[:,d['c']['control']].ravel()[d['free']].copy();a=assemble(d,X,Y,J,trust);record=dict(predictor=outer,trust=trust,start=current,attempts=[]);taken=False;target=None;predicted=None
  for correction in range(SPEC['max_corrections_per_predictor']+1):
   sub=O/f'linear_{lps:02d}';res,lin=solve_lp(a,sub);lps+=1;attempt=dict(linear_index=lps-1,correction=correction,linear=lin);record['attempts'].append(attempt)
   if res is None or res.x is None or not lin.get('linear_pass',False):attempt.update(accepted=False,rejection='no independently feasible linear witness');break
   tangent=res.x[:-1]
   if correction==0:
    predicted=merit-float(res.x[-1]);target=merit-.5*predicted
    if predicted<=0:attempt.update(accepted=False,rejection='no predicted rate/window improvement');break
   lo,hi=map_interval(d,a,trust,origin)
   np.savez_compressed(sub/'map_input.npz',origin=a['z'],tangent=tangent,lower=lo,upper=hi,predictor_origin=origin)
   try:trial=ret.run_bounded(a,tangent,lo,hi)
   except ValueError as exc:
    map_failures+=1;attempt.update(accepted=False,map_pass=False,rejection=str(exc));(sub/'trial.json').write_text(json.dumps(attempt,indent=2));break
   attempt['map_pass']=True
   np.savez_compressed(sub/'nonlinear_trial.npz',X=trial['X'],Y=trial['Y'],tangent=tangent,mapped_controls=trial['mapped_controls'],alpha=trial['alpha'],endpoint=trial['endpoint'])
   attempt.update(root_pass=trial['passed'],root_history=trial['root_history'])
   if not trial['passed']:attempt.update(accepted=False,rejection=trial['reason']);(sub/'trial.json').write_text(json.dumps(attempt,indent=2));break
   tx,ty=trial['X'],trial['Y'];_,tr,te=values(d,tx,ty);s=summary(d,tx,ty,tr,te);actual=merit-s['soft_merit'];ratio=actual/predicted;eq_ok=bool(np.max(abs(te[:-1]))<=1e-13 and abs(te[-1])<=1e-8)
   valid=bool(s['hard_pass'] and s['monotone_phase_pass'] and s['coordinate_bound_minimum']>=0 and s['fixed_states_exact'] and s['inactive_exact_zero'] and s['minimum_corner_weight']>=0 and eq_ok)
   take=bool(valid and actual>0 and ratio>=SPEC['acceptance_ratio'])
   attempt.update(summary=s,total_normal_maximum=float(abs(te[:-1]).max()),hover_normalized_error=float(abs(te[-1])),actual_reduction=actual,predicted_reduction=predicted,ratio=ratio,accepted=take)
   np.savez_compressed(sub/'nonlinear_rows.npz',rows=tr,equalities=te);(sub/'trial.json').write_text(json.dumps(attempt,indent=2))
   if take:
    X,Y,r,e=tx,ty,tr,te;J=ret.jac(X);accepted+=1;taken=True;last_accepted=lps-1
    if ratio>.75:trust=min(1.,trust*2)
    break
   if correction<SPEC['max_corrections_per_predictor']:
    tj=ret.jac(tx);a=assemble(d,tx,ty,tj,trust,'correction',target,origin=origin)
  record.update(accepted=taken,elapsed_s=time.monotonic()-start);history.append(record)
  with (O/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  final=summary(d,X,Y,r,e)
  print(json.dumps(dict(predictor=outer,linear_programs=lps,accepted=taken,total_accepted=accepted,current=final,elapsed_s=time.monotonic()-start)),flush=True)
  if final['all_inequalities_pass'] and final['coordinate_bound_minimum']>=0 and abs(e[-1])<=1e-8 and np.max(abs(e[:-1]))<=1e-13:stop='all modeled buffered rows passed';break
  if not taken:trust*=.5
  if trust<SPEC['minimum_trust_fraction']:stop='declared trust floor';break
 np.savez_compressed(O/'final.npz',X=X,Y=Y,J=J,rows=r,equalities=e,label_to_state=d['g']['label_to_state'],canonical=d['g']['canonical'])
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,specification=SPEC,stop=stop,predictors=len(history),linear_programs=lps,nonlinear_trial_paths=ret.counts['trial_paths'],coordinate_map_failures=map_failures,accepted_steps=accepted,last_accepted_linear_index=last_accepted,initial=seed,final=summary(d,X,Y,r,e),wrapper_counts=ret.counts,source440_unchanged=True,source450_used_as_seed=False,new_final_differs_from_source=bool(not np.array_equal(X,d['initialX'])),sampled_full_geometry_pending=True,independent_audit_pending=True,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,elapsed_s=time.monotonic()-start)
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

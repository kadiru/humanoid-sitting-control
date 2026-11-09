"""Saved-only new478 accounting and exact shared-ray conflict localization.
No model/root/J/LP/trajectory/audit loop is repeated. Failed-ray constraint
intervals are indexed from saved matrices and witnesses, without decoding a
new target, selecting a trial, or changing any original source/result.
"""
import json,datetime
from fractions import Fraction as F
import numpy as np
from scipy import sparse
from low_seat_joint_ray_476 import pieces,restrictions,pair
from low_seat_free_gap_solver_470 import D,read,load_saved,unpack
O=D/'joint_ray_path_478'
def main():
 assert not (D/'joint_ray_saved_scope_481.json').exists()
 d=load_saved();row=d['a'];cm=d['c'];mesh=d['g'];records=read(D/'contact_lift_440/records.json')
 fs,sl=np.unravel_index(d['free'],cm['reference'].shape);trials=[];conflicts=[];prechecks=[];summary=[];calls=0;subs=0
 def owner_details(owner):
  if owner is None:return {'kind':'unit_step_cap'}
  out={'identity':list(owner)}
  if owner[0]=='full_linear_control':
   j=owner[1];b=int(fs[j]);out.update(state=b,slot=int(sl[j]),time_s=records[int(mesh['canonical'][b])]['time_s'])
  return out
 for P in sorted(O.glob('linear_*')):
  k=int(P.name.split('_')[-1]);subs+=1;solver=read(P/'solver.json');t=read(P/'trial.json');a=dict(np.load(P/'input.npz'));meta=read(P/'metadata.json');calls+=int(solver['status']!=-10)
  short=dict(index=k,mode=meta['mode'],solver_status=solver['status'],map_pass=t.get('map_pass'),root_pass=t.get('root_pass'),accepted=t['accepted'],alpha=None,soft=None,hard=None,phase=None,full_box=None,reason=t.get('rejection'))
  if solver['status']==-10:
   widths=a['full_step_upper']-a['full_step_lower'];bad=np.flatnonzero(widths<0);assert len(bad);j=int(bad[np.argmin(widths[bad])]);prechecks.append(dict(index=k,count=len(bad),worst_width=float(widths[j]),owner=owner_details(('full_linear_control',j))))
  if t.get('map_pass') is False:
   a.update(C=sparse.load_npz(P/'C.npz'),drops=list(map(unpack,meta['drops'])),mode=meta['mode'],trust=meta['trust']);w=np.load(P/'witness.npy')[:-1]
   gs,dv=pieces(d,a,w);rr=restrictions(d,a,w,gs,dv);low=F(0);high=F(1);li=hi=None;zero=[]
   for l,u,v,own in rr:
    if not v:
     if not l<=0<=u:zero.append(dict(owner=owner_details(own),lower=pair(l),upper=pair(u)))
     continue
    x,y=(l/v,u/v) if v>0 else (u/v,l/v)
    if x>low:low=x;li=own
    if y<high:high=y;hi=own
   conflicts.append(dict(index=k,alpha_lower=pair(low),alpha_upper=pair(high),alpha_interval_width=pair(high-low),alpha_lower_float=float(low),alpha_upper_float=float(high),signed_interval_width_float=float(high-low),lower_owner=owner_details(li),upper_owner=owner_details(hi),zero_direction_conflicts=zero,scope='Exact arithmetic ray interval in scalar step fraction; not physical units or a global infeasibility certificate. No mapped state/root was executed.'))
  if (P/'nonlinear_rows.npz').exists():
   n=dict(np.load(P/'nonlinear_rows.npz'));r=n['rows'];X=np.load(P/'root_final.npz')['X'];w=np.load(P/'witness.npy')[:-1];mapped=read(P/'mapped.json');alpha=float(unpack(mapped['alpha']))
   z=X[:,cm['control']].ravel()[d['free']];planned=a['z']+sparse.load_npz(P/'C.npz')@(alpha*w);neg=[]
   for g in dict.fromkeys(row['groups'].tolist()):
    ix=np.flatnonzero((row['groups']==g)&row['keep']);bad=ix[r[ix]<0]
    if len(bad):neg.append(dict(group=str(g),negative=len(bad),minimum_normalized=float(r[ix].min()),worst_row=int(ix[np.argmin(r[ix])]),minimum_unscaled=float(np.min(r[ix]*row['scale'][ix]))))
   trial=dict(index=k,shared_alpha=alpha,normal=float(abs(n['equalities'][:-1]).max()),hover=float(abs(n['equalities'][-1])),gap_root=t['actual_gap_root_residual'],global_box=t['summary']['coordinate_bound_minimum'],hard=t['summary']['hard_pass'],phase=t['summary']['monotone_phase_pass'],soft=t['summary']['soft_merit'],actual_soft_reduction=t['actual_reduction'],actual_minus_shared_linear_f_maximum=float(abs(z[sl==0]-planned[sl==0]).max()),negative_groups=neg)
   trials.append(trial);short.update(alpha=alpha,soft=trial['soft'],hard=trial['hard'],phase=trial['phase'],full_box=trial['global_box'])
  summary.append(short)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,assembled_subproblems=subs,actual_scipy_linprog_calls=calls,empty_box_prechecks=len(prechecks),finite_map_failures=len(conflicts),nonlinear_trials=len(trials),normal_pass=sum(t['normal']<=1e-13 for t in trials),hover_pass=sum(t['hover']<=1e-8 for t in trials),phase_pass=sum(t['phase'] for t in trials),hard_pass=sum(t['hard'] for t in trials),global_box_pass=sum(t['global_box']>=0 for t in trials),soft_improving=sum(t['actual_soft_reduction']>0 for t in trials),normal_descent_pass=sum(not any('normal_descent' in g['group'] for g in t['negative_groups']) for t in trials),trial_summary=summary,trials=trials,exact_ray_conflicts=conflicts,empty_prechecks=prechecks,model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
 (D/'joint_ray_saved_scope_481.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['trial_summary','trials','exact_ray_conflicts','empty_prechecks']},indent=2))
if __name__=='__main__':main()

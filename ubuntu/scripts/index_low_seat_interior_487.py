"""Saved-only accounting and localization of completed484 trials.
No new model/root/J/LP/map/certificate/audit calculation is performed.
Original signed rows and decisions remain unchanged.
"""
from pathlib import Path
from fractions import Fraction
import json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
P=D/'interior_path_484'
read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'interior_path_exit_484.json')['actual_exit']==0
 assert not (D/'interior_saved_scope_487.json').exists()
 row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
 cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'))
 mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
 records=read(D/'contact_lift_440/records.json')
 fs,sl=np.unravel_index(cm['free'],cm['reference'].shape)
 result=read(P/'results.json');stages=[];trials=[];certs=[];fails=[]
 for sub in sorted(P.glob('linear_*')):
  k=int(sub.name.split('_')[-1]);a=dict(np.load(sub/'base/input.npz'))
  t=read(sub/'trial.json');solver=read(sub/'solver.json');stage=read(sub/'stage.json')
  out=dict(index=k,role=stage['role'],solver_status=solver['status'],actual_lp_call=solver['actual_lp_call'],linear_pass=solver.get('linear_pass'),certificate_pass=t.get('certificate_pass'),map_pass=t.get('map_pass'),root_pass=t.get('root_pass'),accepted=t['accepted'],reason=t.get('rejection'))
  if (sub/'certificate.json').exists():
   c=read(sub/'certificate.json');v=Fraction(*c['certified_relative_clearance']);own=c['limiting_owner'];owner=dict(identity=own)
   if own and own[0]=='full_linear_control':
    j=own[1];b=int(fs[j]);owner.update(state=b,slot=int(sl[j]),time_s=records[int(mesh['canonical'][b])]['time_s'])
   elif own and own[0]=='target':
    b=own[1];owner.update(state=b,slot=own[2],time_s=records[int(mesh['canonical'][b])]['time_s'])
   certs.append(dict(index=k,lp_eta=c['lp_reported_eta'],exact_eta_pair=c['certified_relative_clearance'],exact_eta_float=float(v),owner=owner,nonpositive_intervals=c['nonpositive_intervals'],passed=c['passed']))
  if (sub/'nonlinear_rows.npz').exists():
   rr=dict(np.load(sub/'nonlinear_rows.npz'));r=rr['rows'];X=np.load(sub/'root_final.npz')['X'];z=X[:,cm['control']].ravel()[cm['free']]
   planned=a['z']+sparse.load_npz(sub/'base/C.npz')@np.load(sub/'witness.npy')[:-1]
   defect=z-planned;fi=np.flatnonzero(sl==0);j=int(fi[np.argmax(abs(defect[fi]))]);b=int(fs[j]);negative=[]
   for g in dict.fromkeys(row['groups'].tolist()):
    ix=np.flatnonzero((row['groups']==g)&row['keep']);bad=ix[r[ix]<0]
    if len(bad):
     ri=int(ix[np.argmin(r[ix])]);negative.append(dict(group=str(g),hard=bool(not row['soft'][ri]),negative=len(bad),minimum_row=ri,minimum_normalized=float(r[ri]),minimum_unscaled_at_same_row=float(r[ri]*row['scale'][ri])))
   m=read(sub/'mapped.json');trials.append(dict(index=k,alpha_exact=m['alpha'],alpha_exact_one=m['alpha_exact_one'],strict_targets=m['decoded_targets_strictly_interior'],strict_drops=m['decoded_drops_strictly_interior'],normal=float(abs(rr['equalities'][:-1]).max()),hover=float(abs(rr['equalities'][-1])),gap_root=t['actual_gap_root_residual'],hard=t['summary']['hard_pass'],phase=t['summary']['monotone_phase_pass'],full_box=t['summary']['coordinate_bound_minimum'],soft=t['summary']['soft_merit'],actual_reduction=t['actual_reduction'],ratio=t['ratio'],accepted=t['accepted'],maximum_dependent_f_linearization_defect=dict(signed=float(defect[j]),absolute=float(abs(defect[j])),state=b,time_s=records[int(mesh['canonical'][b])]['time_s']),negative_groups=negative))
  if t.get('rejection'):fails.append(dict(index=k,role=stage['role'],reason=t['rejection'],solver_status=solver['status']))
  stages.append(out)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,assembled_subproblems=len(stages),actual_scipy_linprog_calls=sum(s['actual_lp_call'] for s in stages),primary_stages=sum(s['role']=='primary' for s in stages),interior_stages=sum(s['role']=='interior' for s in stages),certificates=len(certs),certificates_passed=sum(c['passed'] for c in certs),nonlinear_trials=len(trials),normal_pass=sum(t['normal']<=1e-13 for t in trials),hover_pass=sum(t['hover']<=1e-8 for t in trials),phase_pass=sum(t['phase'] for t in trials),hard_pass=sum(t['hard'] for t in trials),global_box_pass=sum(t['full_box']>=0 for t in trials),strict_target_and_drop_pass=sum(t['strict_targets'] and t['strict_drops'] for t in trials),exact_alpha_one_pass=sum(t['alpha_exact_one'] for t in trials),normal_descent_pass=sum(not any('normal_descent' in g['group'] for g in t['negative_groups']) for t in trials),soft_improving=sum(t['actual_reduction']>0 for t in trials),accepted_steps=result['accepted_steps'],trial_summary=stages,exact_certificates=certs,trials=trials,failures=fails,model_calls=0,root_calls=0,map_calls=0,optimizer_calls=0,audit_calls=0,physical_admission=False)
 assert out['actual_scipy_linprog_calls']==result['actual_lp_calls'] and len(trials)==result['nonlinear_trial_paths']
 (D/'interior_saved_scope_487.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k not in ['trial_summary','exact_certificates','trials','failures']},indent=2))
if __name__=='__main__':main()

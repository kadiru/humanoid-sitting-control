"""Saved-only accounting and dependent-f correction-box localization.
No mechanics, root, Jacobian, optimizer, geometry or audit is repeated.
"""
from pathlib import Path
from fractions import Fraction
import json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'free_gap_path_472'
read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'free_gap_saved_scope_475.json').exists()
 rows=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');cm=np.load(D/'clock_row_preparation_446/coordinate_map.npz');mesh=np.load(D/'clock_transcription_preparation_444/mesh.npz');records=read(D/'contact_lift_440/records.json')
 free_states,free_slots=np.unravel_index(cm['free'],cm['reference'].shape);subproblems=[];skips=[];trials=[]
 for P in sorted(O.glob('linear_*')):
  k=int(P.name.split('_')[-1]);solver=read(P/'solver.json');decision=read(P/'trial.json');a=np.load(P/'input.npz');meta=read(P/'metadata.json')
  subproblems.append(dict(index=k,mode=meta['mode'],status=solver['status'],actual_scipy_linprog_call=solver['status']!=-10,candidate=solver['candidate']))
  if solver['status']==-10:
   widths=a['full_step_upper']-a['full_step_lower'];bad=np.flatnonzero(widths<0);assert len(bad);j=int(bad[np.argmin(widths[bad])]);b=int(free_states[j]);slots,counts=np.unique(free_slots[bad],return_counts=True)
   skips.append(dict(index=k,empty_full_control_intervals=len(bad),slot_counts={str(int(s)):int(c) for s,c in zip(slots,counts)},worst_negative_interval_width=float(widths[j]),worst_full_control_index=j,worst_state=b,worst_time_s=records[int(mesh['canonical'][b])]['time_s'],scope='All empty intervals are dependent descent f, dimensionless. This is a wrapper box precheck, not a HiGHS infeasibility result.'))
  if (P/'nonlinear_rows.npz').exists():
   r=np.load(P/'nonlinear_rows.npz')['rows'];b=np.load(P/'root_final.npz');z=b['X'][:,cm['control']].ravel()[cm['free']];w=np.load(P/'witness.npy');planned=a['z']+sparse.load_npz(P/'C.npz')@w[:-1];fids=free_slots==0
   domain=[]
   for group in ['mesh_normal_descent','old_normal_descent','old_phase','mesh_phase']:
    ix=np.flatnonzero((rows['groups']==group)&rows['keep']);assert len(ix)
    domain.append(dict(group=group,negative=int((r[ix]<0).sum()),minimum_normalized=float(r[ix].min()),worst_row=int(ix[np.argmin(r[ix])]),minimum_unscaled=float(np.min(r[ix]*rows['scale'][ix]))))
   mapped=read(P/'mapped.json');alpha=[float(Fraction(*g['alpha'])) for g in mapped['groups'] if 'alpha' in g]
   trials.append(dict(index=k,normal_maximum=decision['total_normal_maximum'],hover_error=decision['hover_normalized_error'],gap_root_error=decision['actual_gap_root_residual'],global_control_bound_minimum=decision['summary']['coordinate_bound_minimum'],hard_pass=decision['summary']['hard_pass'],phase_pass=decision['summary']['monotone_phase_pass'],soft_violation=decision['summary']['soft_merit'],actual_soft_reduction=decision['actual_reduction'],normal_drop_alpha=float(Fraction(*mapped['drop_alpha'])),other_chart_alpha_range=[min(alpha),max(alpha)],actual_minus_linear_predicted_f_maximum=float(abs(z[fids]-planned[fids]).max()),domains=domain))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,assembled_subproblems=len(subproblems),actual_scipy_linprog_calls=sum(x['actual_scipy_linprog_call'] for x in subproblems),correction_boxes_rejected_before_scipy=len(skips),main_linear_programs_field_scope='Counts solve_new_lp wrapper invocations/assembled subproblems, including six empty-box prechecks; it is not the number of scipy.linprog calls.',nonlinear_trials=len(trials),all_new_trials_normal_hover_root_pass=all(x['normal_maximum']<=1e-13 and x['hover_error']<=1e-8 and x['gap_root_error']<=1e-13 for x in trials),all_new_trials_normal_descent_pass=all(all(g['negative']==0 for g in x['domains'] if 'normal_descent' in g['group']) for x in trials),global_box_pass_count=sum(x['global_control_bound_minimum']>=0 for x in trials),phase_pass_count=sum(x['phase_pass'] for x in trials),hard_pass_count=sum(x['hard_pass'] for x in trials),positive_soft_reduction_count=sum(x['actual_soft_reduction']>0 for x in trials),subproblems=subproblems,empty_corrections=skips,trials=trials,read_only_inspection_failure='One earlier inline metadata inspection called min on an absent mesh_gap_order group; it failed before writes. Corrected inspection used actual saved group names. No model/audit/scientific loop repeated.',model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
 (D/'free_gap_saved_scope_475.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['subproblems','empty_corrections','trials']},indent=2))
if __name__=='__main__':main()

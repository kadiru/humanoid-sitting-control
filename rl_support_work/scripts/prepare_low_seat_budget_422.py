"""Saved-array-only timing localization and new budget-domain preparation.
No FK, root, derivative model evaluation, path optimization or schedule execution.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from low_seat_budget_domain_422 import *
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'budget_preparation_422'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False)
    prior=json.loads((D/'monotone_preparation_410/preparation.json').read_text());audit=json.loads((D/'monotone_preparation_audit_411/independent_audit.json').read_text());assert prior['passed'] and audit['passed']
    f=dict(np.load(D/'monotone_preparation_410/fixture.npz'));b=dict(np.load(D/'window_path_400/initial.npz'));s=dict(np.load(D/'direct_path_418/structural.npz'));e=dict(np.load(D/'direct_path_418/event_clock.npz'))
    # Necessary duration for this SAVED waveform, not a new path or timing solve.
    C=s['C'];delta=np.diff(C,axis=0);ulp=e['ulp'];effective=b['rates']-4*ulp/.02
    upper=b['command_hi']-C[1:]-32*ulp;lower=C[1:]-b['command_lo']-32*ulp
    assert min(upper.min(),lower.min())>0 and effective.min()>0
    rate=np.max(abs(delta)/effective,axis=1);hrup=np.max(.18*np.maximum(delta,0)/upper,axis=1);hrlo=np.max(.18*np.maximum(-delta,0)/lower,axis=1)
    normal=np.zeros(114);normal[99:104]=(s['Y'][98:103,54]-s['Y'][99:104,54])/.006
    components=np.c_[np.full(114,.02),rate,hrup,hrlo,normal];required=components.max(1);groups=[(0,16),(16,64),(64,99),(99,104),(104,114)];records=[]
    for g,(a,c) in enumerate(groups):
        t=float(required[a:c].sum());ticks=int(np.ceil(t/.02));ticks=max(ticks,101 if g==3 else c-a+1)
        records.append(dict(group=g,first_segment=a,last_segment_exclusive=c,saved_time_s=float(s['dt'][a:c].sum()),conditional_minimum_time_s=t,conditional_event_tick_requirement=ticks,limiting_component_counts=np.bincount(components[a:c].argmax(1),minlength=5).tolist()))
    # Algebraic witness: required durations satisfy these individual inequalities.
    durrows=np.c_[required[:,None]*effective-delta,required[:,None]*effective+delta,upper*required[:,None]-.18*delta,lower*required[:,None]+.18*delta]
    normalrows=.006*required[99:104]-(s['Y'][98:103,54]-s['Y'][99:104,54]);assert min(durrows.min(),normalrows.min())>-1e-14
    timing=dict(scope='Conditional necessary bound for EXACT saved418 commands and gaps. Not a lower bound across geometries/forces, not a schedule, and not sufficient for continuous-window, geometry, float32 or quiet admission.',groups=records,component_names=['20ms research floor','reserved rate including4ULP/tick','upper linear target including32ULP','lower linear target including32ULP','normal6mm/s'],movement_time_lower_bound_s=float(required.sum()),with_minimum_synthetic_quiet_s=float(required.sum()+.5),event_movement_tick_lower_bound=sum(r['conditional_event_tick_requirement'] for r in records),with_minimum_quiet_ticks=sum(r['conditional_event_tick_requirement'] for r in records)+25,rate_envelope_note='4ULP PER20ms, not per geometric segment.',fixed_waveform_retiming_cannot_meet_budget=bool(required.sum()>13.),strict_sufficiency_claim=False)
    lo,hi=coordinate_bounds(b,f['shape']);hard=partition();keep=f['keep'];scale=f['scale'];slack=float(np.max(-f['iq'][keep&~hard]/scale[keep&~hard])+1e-6);seed=f['z'];y=seed[:898];assert np.array_equal(expand(y),seed)
    values,J=assemble(f['iq'],f['v'],seed,slack,scale,keep,lo,hi,f['I'],f['G']);lowerbox,upperbox=domain(seed)
    assert values.min()>=0
    rng=np.random.default_rng(422);direction=rng.normal(size=899);direction/=np.linalg.norm(direction);expanded=np.r_[direction[:898],-direction[894:898].sum()]
    separate=np.r_[(f['I']@expanded)[keep]/scale[keep]+(~hard[keep])*direction[-1],(f['G']@expanded)[:786],-(f['G']@expanded)[:786],expanded[-1],-expanded[-1]]
    error=float(abs(J@direction-separate).max());assert error<2e-12
    np.savez_compressed(O/'fixture.npz',seed=seed,slack=slack,hard=hard,keep=keep,scale=scale,lo=lo,hi=hi,values=values,jacobian=J,lower=lowerbox,upper=upperbox,direction=direction,expanded_direction=expanded,separate_direction=separate,budget_lo=BUDGET_LO,budget_hi=BUDGET_HI,budget_map=BUDGET_MAP)
    np.savez_compressed(O/'saved_timing.npz',commands=C,gaps=s['Y'][:,54],saved_dt=s['dt'],components=components,required=required,rows=durrows,normalrows=normalrows,upper=upper,lower=lower,effective_rate=effective)
    (O/'saved_timing.json').write_text(json.dumps(timing,indent=2))
    hashes={**prior['source_input_sha256'],**audit['source_sha256']}
    for p in [Path(__file__),Path('scripts/low_seat_budget_domain_422.py'),Path('scripts/low_seat_direct_constraints_416.py'),D/'monotone_preparation_410/fixture.npz',D/'monotone_preparation_audit_411/independent_audit.json',D/'direct_path_418/structural.npz',D/'direct_path_418/event_clock.npz',D/'direct_saved_constraint_groups_421.json',D/'window_path_400/initial.npz']:hashes[str(p.resolve())]=sha(p)
    for p,h in hashes.items():assert sha(p)==h,p
    r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,seed='original406 via410, not418',continuous_path_coordinates=894,independent_integer_event_budgets=4,dependent_event_budget='650-sum(first4)',epigraph_variables=1,optimizer_rows=len(values),jacobian_shape=list(J.shape),same_continuous_research_domain_as416=True,algebra_directional_error=error,original_full_derivative_error=prior['full_directional_error'],unchanged_derivative_bound=2e-6,new_model_or_derivative_evaluations=0,optimizer_calls=0,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False,source_input_sha256=hashes)
    (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in r.items() if k!='source_input_sha256'},indent=2));print(json.dumps(timing,indent=2))
if __name__=='__main__':main()

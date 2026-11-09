"""Preflight ONE new mixed-integer tangent restoration of the COMPLETE115-knot path.
Saved388 is an infeasible research seed, not a restarted nonlinear solve. All786 shape/
COM/force coordinates,109 interior timing ratios and FIVE integer event budgets are
joint variables. The linear subproblem retains all original/support rows and existing
inward envelopes. No failed force/target is clipped. Full nonlinear rows, all ideal
clock rows and sampled geometry MUST be evaluated after the one subproblem; their
failure blocks all actual schedules and physics. No second nonlinear optimizer.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from low_seat_event_clock_v2_386 import make_event,GROUPS,timing
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'integer_restoration_preparation_392'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);d=make_event();p=d['p'];f=np.load(D/'event_path_388/event_clock.npz');z=f['parameters'].copy();budgets=f['budgets'].copy();eq,full=d['evaluate'](z);saved=dict(d['last']);E0,I0=p['evaluate'](saved['x'],True);I0=I0.copy();ulp=d['ulp'];T=saved['T'];B=np.zeros((114,5))
 for g,(a,b) in enumerate(GROUPS):B[a:b,g]=.02*(saved['x'][1026+a:1026+b]-.02)/(.02*(budgets[g]-(b-a)))
 for start in [12212,14036]:
  for k in range(114):I0[start+k*16:start+(k+1)*16,1026+k]-=4*ulp/.02
 for start in [15860,17684]:
  for k in range(114):I0[start+k*16:start+(k+1)*16,1026+k]-=32*ulp
 M=np.c_[T,B];E=np.c_[E0[:,d['shape_free']],E0[:,1026:]@M];I=np.c_[I0[:,d['shape_free']],I0[:,1026:]@M];base=full[:20284];scale=d['scale'][:20284];keep=d['keep'][:20284]
 def core(v):
  x=saved['x'].copy();x[d['shape_free']]=v[:786];dt,_=timing(v[786:895],v[895:]);x[1026:]=dt;ee,raw=p['evaluate'](x);tight=d['static'].copy()
  for start in [12212,14036]:tight[start:start+1824]=(dt[:,None]/.02*4*ulp).ravel()
  for start in [15860,17684]:tight[start:start+1824]=(dt[:,None]*32*ulp).ravel()
  return ee,raw-tight
 v=np.r_[z,budgets];rng=np.random.default_rng(392);direction=rng.normal(size=900);direction[-5:]-=direction[-5:].mean();direction/=np.linalg.norm(direction);h=1e-5;ep,ip=core(v+h*direction);em,im=core(v-h*direction);fd=np.r_[(ep-em)/(2*h),(ip-im)/(2*h)];an=np.r_[E@direction,I@direction];err=float(abs(fd-an).max());assert err<2e-6
 original=np.load(D/'event_path_388/final.npz');recon=float(max(abs(saved['X']-original['full_coordinates']).max(),abs(saved['Y']-original['local_outputs']).max(),abs(full-original['ineq']).max()));assert recon<1e-10
 # One declared trust box, not a parameter sweep: phase/COM/null <=.02 in their
 # existing coordinates, logratios<=.1, integer event transfer<=5ticks per phase.
 radius=np.r_[np.full(786,.02),np.full(109,.1),np.full(5,5.)];lo=-radius.copy();hi=radius.copy()
 for k,(a,b) in enumerate(d['bounds']):
  if a is not None:lo[k]=max(lo[k],a-z[k])
  if b is not None:hi[k]=min(hi[k],b-z[k])
 # Keep at least one20ms interval per knot plus positive softmax excess; normal
 # descent retains its101-tick research reserve. All650 movement ticks conserved.
 for g,(a,b) in enumerate(GROUPS):lo[895+g]=max(lo[895+g],b-a+1-budgets[g])
 lo[898]=max(lo[898],101-budgets[3]);assert np.all(lo<=hi)
 np.savez_compressed(O/'fixture.npz',v=v,z=z,budgets=budgets,groups=GROUPS,x=saved['x'],X=saved['X'],Y=saved['Y'],eq=eq,ineq=base,raw_ineq=saved['raw'],tighten=saved['tighten'],E=E,I=I,T=T,B=B,scale=scale,keep=keep,shape_free=d['shape_free'],direction=direction,h=h,fd=fd,analytic=an,radius=radius,lower=lo,upper=hi,ulp=ulp)
 paths=[Path(__file__).resolve(),Path('scripts/low_seat_event_clock_v2_386.py').resolve(),D/'event_path_388/final.npz',D/'event_path_388/event_clock.npz',D/'event_path_audit_389/independent_audit.json'];hashes={str(q):sha(q) for q in paths}
 hashes.update(json.loads((D/'event_path_388/revision.json').read_text())['source_input_sha256'])
 for q,hv in hashes.items():assert sha(q)==hv,q
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,source_input_sha256=hashes,reconstruction_error=recon,directional_error=err,unchanged_derivative_bound=2e-6,linearized_rows=20284,retained_rows=int(keep.sum()),free_variables=900,integer_budget_variables=5,maximum_subproblem_solves=1,maximum_nodes=1000,time_limit_s=90,objective='minimize maximum normalized coordinate displacement, subject to ALL retained linearized constraints; no epigraph feasibility slack or secondary optimization',trust_radius_shape=.02,trust_radius_logratio=.1,trust_radius_budget_ticks=5,normal_budget_minimum_ticks=101,full_original_and_clock_checks_mandatory_after_solve=True,clock_rows_not_linearized_for_changing_integer_grid=True,zero_budget_change_would_not_prove_budgets_caused_prior_failure=True,actual_float32_schedule=False,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({q.name:sha(q) for q in O.iterdir() if q.is_file()},indent=2));print(json.dumps({k:v for k,v in r.items() if k!='source_input_sha256'},indent=2))
if __name__=='__main__':main()

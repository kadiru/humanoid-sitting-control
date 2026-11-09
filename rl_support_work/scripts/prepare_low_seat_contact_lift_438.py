"""New persistent-contact construction preparation; no complete path or optimizer."""
import json,datetime
from pathlib import Path
import numpy as np
from low_seat_contact_lift_438 import *
O=D/'contact_lift_preparation_438'
def main():
 O.mkdir(exist_ok=False);s=setup();w,changes=node_loads(s)
 # One fixed derivative fixture at every new equation type and both localized
 # transition mechanisms. No pose/parameter search and no kinematic projection.
 indices=[1351+16,769,912,1200,1300,1351+100]
 tests=[];analytic=[];finite=[];errs=[]
 for idx in indices:
  t,v,kind,k=s['queries'][idx];r=record(s,w,t,v,kind,k);y,J=kinematics(s,r,True);h=1e-6;fd=np.empty_like(J)
  for j in range(len(J)):
   d=np.zeros(len(J));d[j]=h;plus,*_=kinematics(s,apply_step(s,r,d));minus,*_=kinematics(s,apply_step(s,r,-d));fd[:,j]=(plus-minus)/(2*h)
  error=float(abs(J-fd).max());assert error<2e-6,(idx,error)
  tests.append(dict(record=idx,columns=len(J),error=error));analytic.append(J);finite.append(fd);errs.append(error)
 np.savez_compressed(O/'fixture.npz',weights=w,source_weights=s['weights'],points=s['points'],local=s['local'],feet=s['feet'],mapping=s['mapping'],old_points=s['P'],old_weights=s['oldw'],**{f'J_{i}':v for i,v in enumerate(analytic)},**{f'FD_{i}':v for i,v in enumerate(finite)})
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,construction=__import__('low_seat_contact_lift_438').__doc__,source430_prior_gates_passed=True,source430_extra_buffer_passed=False,canonical_vertex_identity_error_m=s['identity_error'],projection_inset_m=PROJECTION_INSET,physical_support_m=.005,prior_support_reserve_m=8*float(np.finfo(np.float32).eps),extra_buffer_dimensionless=BUFFER,extra_support_buffer_m=.005*BUFFER,nominal_root_roundoff_guard_m=2e-12,changed_nominal_nodes=changes,root_derivative_fixtures=tests,maximum_derivative_error=max(errs),unchanged_derivative_acceptance=2e-6,exact_budgets=BUDGETS,event_times=[dict(knot=k,numerator=str(s['node_times'][k].numerator),denominator=str(s['node_times'][k].denominator)) for k in [16,64,99,104,114]],vertex_times=[[str(t.numerator),str(t.denominator)] for t in s['times']],ids=s['ids'].tolist(),queries=1467,root_projection_calls=0,complete_path_constructed=False,optimizer_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,physical_admission=False,passed=True)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

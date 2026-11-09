"""New normal-drop finite-chart arithmetic fixtures ONLY. No mechanics or optimizer."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from scipy import sparse
from low_seat_exact_gap_chart_468 import *
O=D/'exact_gap_chart_preparation_468'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not O.exists();O.mkdir();d=load_domain();origin=initial(d);zero,_=map_drops(origin,np.zeros(205),d);assert zero==origin;center=fixture_center(d);pivot=d['pivot'];ids=d['independent'];trials=[];maxerr=0.;mxround=F(0)
 for k,i in enumerate(ids):
  room=min(d['upper'][i]-center[i],center[pivot]-d['lower'][pivot]);h=float(room/F(8));eps=float(room/F(10000));assert h>eps>0
  outputs=[];extra=[]
  for a in [h-eps,h,h+eps]:
   t=np.zeros(205);t[k]=a;z,info=map_drops(center,t,d);outputs.append(z);extra.append(info)
  fd=np.asarray([float((b-a)/(fq(h+eps)-fq(h-eps))) for a,b in zip(outputs[0],outputs[2])]);der=np.zeros(206);factor=float(1/(1+extra[1]['rho'])**2);der[i]=factor;der[pivot]=-factor;err=float(abs(fd-der).max());maxerr=max(maxerr,err)
  mxround=max(mxround,max(abs(z-q) for z,q in zip(outputs[1],extra[1]['unrounded'])))
  trials.append(dict(edge=i,h=h,eps=eps,outputs=[[pair(v) for v in z] for z in outputs],alpha=pair(extra[1]['alpha']),rho=pair(extra[1]['rho']),raised=extra[1]['raised'],error=err))
  (O/'ray_fixtures.json').write_text(json.dumps(trials))
 assert maxerr<2e-6,maxerr
 # Three new arbitrary joint directions test finite decoding and exact chain
 # sums. They are arithmetic fixtures only; no whole-path states are solved.
 z=center;chain=[]
 for step in range(3):
  t=np.zeros(205)
  for k,i in enumerate(ids):
   room=min(d['upper'][i]-z[i],z[i]-d['lower'][i]);t[k]=float(room)*(.03 if (k+step)%2 else -.03)
  zz,info=map_drops(z,t,d);decoded,exact=gaps(zz,d);chain.append(dict(origin=[pair(v) for v in z],tangent=t.tolist(),output=[pair(v) for v in zz],alpha=pair(info['alpha']),rho=pair(info['rho']),raised=info['raised'],gaps=decoded.tolist(),exact_gaps=[pair(v) for v in exact]));z=zz
 (O/'joint_chain.json').write_text(json.dumps(chain))
 initgap,initexact=gaps(origin,d);original=list(map(unpack,d['layout']['drops']))
 layout=dict(lower=[pair(v) for v in d['lower']],upper=[pair(v) for v in d['upper']],original_lower=[pair(v) for v in d['original_lower']],original_upper=[pair(v) for v in d['original_upper']],initial=[pair(v) for v in origin],fixture_center=[pair(v) for v in center],pivot=pivot,independent=ids,total=pair(d['total']),hover=pair(d['hover']),terminal=pair(d['terminal']),one_ulp_bound=pair(U),initial_gaps=[pair(v) for v in initexact])
 (O/'layout.json').write_text(json.dumps(layout));np.savez_compressed(O/'initial_targets.npz',gaps=initgap,normal_states=np.asarray(d['layout']['normal_states']))
 # Full331888-row derivative columns are the SAME CONTINUOUS reference basis
 # from466, before the new finite rounding. No model or new-state claim.
 cols=[i for i,c in enumerate(d['layout']['columns']) if c['kind']=='normal_drop'];assert len(cols)==205
 A=sparse.load_npz(P/'all_rows_gap_chain.npz')[:,cols];C=sparse.load_npz(P/'original_control_box_chain.npz')[:,cols];sparse.save_npz(O/'normal_drop_full_chain.npz',A);sparse.save_npz(O/'normal_drop_original_box_chain.npz',C)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,normal_edges=206,independent_drops=205,ray_fixtures=205,new_joint_maps=3,derivative_error=maxerr,unchanged_bound=2e-6,maximum_finite_rounding_error_m=float(mxround),exact_sum_every_map=True,zero_tangent_exact=True,initial_representation_maximum_drop_change_m=float(max(abs(a-b) for a,b in zip(origin,original))),independent_rounding_bound_m=float(U),pivot_rounding_bound_m=float(205*U),decoded_gap_rounding_bound_m=float(U),no_cumulative_endpoint_sum_error=True,continuous_derivative_scope='Before finite rounding only; numerical stored map not literally differentiable. Future solver must use recentered tangent and actual nonlinear checks, never a finite-map Jacobian shortcut.',normal_drop_bounds_tightened_only=True,all_old_rows_and_buffers_retained=True,model_calls=0,root_calls=0,trajectory_optimizers=0,new_complete_paths=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

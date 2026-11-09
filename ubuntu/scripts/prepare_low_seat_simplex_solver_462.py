"""Saved-matrix assembly only. No model, map fixture, root or optimizer."""
import json,datetime
from low_seat_simplex_solver_462 import *
O=D/'simplex_solver_preparation_462'
def main():
 O.mkdir(exist_ok=False);d=load_saved();groups,R=chart();out=[]
 for mode in ['predictor','correction']:
  a=assemble(d,R,d['initialX'],d['initialY'],d['initialJ'],1.,mode,None if mode=='predictor' else .15,None if mode=='predictor' else d['reference'])
  p=O/mode;p.mkdir()
  for k in ['matrix','equality','original_matrix','original_equality']:sparse.save_npz(p/(k+'.npz'),a[k])
  keys=['rhs','equality_rhs','bounds','cost','original_rhs','original_equality_rhs','original_bounds','normal_conservation_rows','r','e']
  np.savez_compressed(p/'arrays.npz',**{k:a[k] for k in keys})
  out.append(dict(mode=mode,matrix_shape=a['matrix'].shape,equality_shape=a['equality'].shape,conserved_normal_rows=int(a['normal_conservation_rows'].sum())))
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,assemblies=out,all_full_control_bounds_including_pivots=True,normal_primal_bound_unchanged=1e-13,model_calls=0,root_calls=0,optimizer_calls=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

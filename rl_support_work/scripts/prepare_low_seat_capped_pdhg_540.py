"""Prepare exact diagonal norm certificates and two NEW small numerical LP tests, once."""
from low_seat_capped_pdhg_540 import *
import datetime

def main():
 launch=read(D/'capped_pdhg_launch_540.json');assert read(D/'capped_pdhg_freeze_exit_540.json')['actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 out=D/'capped_pdhg_preparation_540';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  src=D/'capped_max_preparation_538'/role;H=sparse.load_npz(src/'hard.npz');S=sparse.load_npz(src/'coupled.npz');m=exact_metric(H,S);o=out/role;o.mkdir();np.savez(o/'metric.npz',**m)
  systems[role]=dict(hard_rows=H.shape[0],coupled_rows=S.shape[0],native_columns=H.shape[1],native_nonzeros=H.nnz+S.nnz,row_step_power_min=int(m['row_power'].min()),row_step_power_max=int(m['row_power'].max()),column_step_power_min=int(m['column_power'].min()),column_step_power_max=int(m['column_power'].max()),source_H_sha256=sha(src/'hard.npz'),source_S_sha256=sha(src/'coupled.npz'),source_arrays_sha256=sha(src/'arrays.npz'),source_coefficients_changed=False,full_size_LP_calls=0)
  print(json.dumps(dict(role=role,**systems[role])),flush=True)
 tests=[dict(name='asymmetric_sum_primary',H=[[-1,0],[0,-1]],S=[[1,1],[2,0]],bH=[-1,-1.5],bS=[0,0],L=0.,U=4.,e=1,known_x=[1.,1.5,2.5],known_hard_dual=[1.,1.],known_coupled_dual=[1.,0.],known_upper=0.,known_lower=0.,known_objective=2.5),dict(name='translated_clearance',H=[[1,1]],S=[[1,0],[-1,0],[0,1],[0,-1]],bH=[3.],bS=[2.,1.,3.,1.],L=-3.,U=0.,e=-1,known_x=[.5,1.,1.5],known_hard_dual=[0.],known_coupled_dual=[.5,.5,0.,0.],known_upper=0.,known_lower=0.,known_objective=-1.5)]
 results={}
 for t in tests:
  root=out/t['name'];root.mkdir();H=sparse.csr_matrix(np.array(t['H'],dtype=float));S=sparse.csr_matrix(np.array(t['S'],dtype=float));bH=np.array(t['bH']);bS=np.array(t['bS']);sparse.save_npz(root/'hard.npz',H);sparse.save_npz(root/'coupled.npz',S);np.savez(root/'arrays.npz',bH=bH,bS=bS,L=t['L'],U=t['U'],e=t['e']);(root/'known_optimum.json').write_text(json.dumps(t,indent=2))
  result=solve_lp(H,S,bH,bS,t['L'],t['U'],t['e'],root/'solve',small_trace=True);results[t['name']]=result;print(json.dumps(dict(test=t['name'],**result)),flush=True)
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=all(x['status']==0 for x in results.values()),specification=SPEC,systems=systems,tests=results,new_small_LP_calls=2,new_full_size_LP_calls=0,full_size_PDHG_steps=0,new_full_size_metric_certificates=2,model_calls=0,root_calls=0,finite_maps=0,physical_admission=False,new_whole_path_wrapper=False)
 (out/'preparation.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

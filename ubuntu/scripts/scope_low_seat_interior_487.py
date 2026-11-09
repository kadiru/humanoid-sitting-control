"""Saved-only scope of the final rejected484 trial's hard deficits.
Keep exact signs, distinguish the extra2^-23 buffer from the underlying row.
No model/root/J/LP/map or independent-audit replay.
"""
from pathlib import Path
import json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def main():
 p=D/'interior_constraint_scope_487.json';assert not p.exists()
 scope=json.loads((D/'interior_saved_scope_487.json').read_text());last=scope['trials'][-1];k=last['index']
 arrays=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
 matrix=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz').tocsr()
 r=np.load(D/f'interior_path_484/linear_{k:03d}/nonlinear_rows.npz')['rows'];bad=np.flatnonzero(arrays['keep']&~arrays['soft']&(r<0));rows=[]
 for i in bad:
  group=str(arrays['groups'][i]);assert group in ['mesh_command_lower','mesh_planning_lower'],group
  scale=float(arrays['scale'][i]);buffer=scale*2**-23;value=float(r[i]*scale)
  q=matrix.getrow(i);assert q.nnz==1 and q.data[0]==1
  state,feature=divmod(int(q.indices[0]),75)
  unit='rad' if group=='mesh_command_lower' or feature>=4 else 'm'
  rows.append(dict(row=int(i),group=group,state=state,feature=feature,unit=unit,buffered_normalized=float(r[i]),buffered_unscaled=value,additional_buffer_unscaled=buffer,before_additional_buffer_unscaled=value+buffer))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,last_rejected_trial=k,hard_negative_rows=rows,all_these_pre_buffer_rows_positive=all(x['before_additional_buffer_unscaled']>0 for x in rows),trial_still_rejected=True,soft_merit=last['soft'],original_rate_target_feasibility_not_inferred_from_hard_rows=True,model_calls=0,root_calls=0,optimizer_calls=0)
 p.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

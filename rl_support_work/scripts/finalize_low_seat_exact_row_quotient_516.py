"""Saved-output-only metadata completion after the terminal synthetic fixture assertion.

No grouping, matrix product, fixture calculation, LP or model is repeated.
"""
from low_seat_exact_row_quotient_v2_516 import D,SPEC,read,sha
from pathlib import Path
import json,datetime,numpy as np
from scipy import sparse
O=D/'exact_row_quotient_preparation_516'
def main():
 assert read(D/'exact_row_quotient_preparation_exit_516.json')['actual_exit']==1
 stage=read(D/'exact_row_quotient_staged_launch_516.json');assert stage['compile_actual_exit']==0
 for p,h in stage['preserved_source_output_sha256'].items():assert sha(p)==h,p
 assert not (O/'preparation.json').exists();results={}
 for mode in ['primary','witness']:
  ids=np.load(O/mode/'row_ids.npy');results[mode]={}
  for name,folder in [('original','lifted_feasibility_preparation_490'),('transformed','mixed_coordinate_preparation_504')]:
   src=D/folder/mode;M=sparse.load_npz(src/'matrix.npz');N=sparse.load_npz(O/mode/name/'matrix.npz');q=dict(np.load(O/mode/(name+'_ledger.npz')))
   results[mode][name]=dict(source=str(src),full_shape=list(M.shape),full_nnz=M.nnz,positive_ray_groups=len(q['representatives'])-len(q['contradiction_rows']),tautologies=len(q['tautology_rows']),contradictions=len(q['contradiction_rows']),own_representatives=len(q['representatives']),selected_rows=len(ids),selected_nnz=N.nnz,omitted_rows=M.shape[0]-len(ids),equality_source=str(src/'equality.npz'),equality_sha256=sha(src/'equality.npz'))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=False,scope=__doc__,saved_full_systems_complete=True,independent_full_system_audit_required=True,specification=SPEC,systems=results,
  original_producer_actual_exit=1,synthetic_fixture_passed=False,synthetic_fixture_saved_output_available=False,synthetic_fixture_repeated=False,
  failure='After all four full-system ledgers and both paired copied matrices were saved, an expected-ID assertion in the synthetic fixture failed. The transformed fixture row7 is EXACTLY two times row3 (both use nextafter at adjacent powers of two), and its RHS is tighter, so row3 must share representative7. The declared expected vector incorrectly retained3. Original synthetic computed output arrays were not saved and are UNAVAILABLE; no fixture pass is claimed and no grouping/fixture is rerun.',
  staged_scope='Read existing arrays for metadata only. Independent517 must certify EVERY real-system implication and byte-exact selected row copy before these outputs can be used. Completed full-system groupings are not repeated.',
  full_basis_or_row_transforms_rebuilt=False,lp_calls=0,model_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

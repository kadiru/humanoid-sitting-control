"""Saved-array-only stencil correction. No model/Jacobian/root rerun.
Original446 accidentally started dense approach-gap order at node64; the
unchanged original ordered-gap domain starts at node65. Remove ONLY those
extra64->65 edges. Original27708/1572 rows and every inward reserve stay exact.
Both source/result versions remain immutable and separately named.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');I=D/'clock_row_preparation_446';O=D/'clock_row_preparation_v2_446'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert read(D/'clock_row_preparation_exit_446.json')['actual_exit']==0
 for p,h in read(I/'SHA256.json').items():assert sha(I/p)==h,p
 O.mkdir(exist_ok=False);a=dict(np.load(I/'row_arrays.npz'));g=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));ts=[Fraction(int(x),int(y)) for x,y in read(D/'clock_transcription_preparation_444/exact_times.json')];M=sparse.load_npz(I/'all_rows_matrix.npz');J=sparse.load_npz(I/'global_jacobian.npz')
 start=ts[1351+65];keep=np.ones(M.shape[0],bool);removed=[]
 for row in np.flatnonzero(a['groups']=='mesh_approach_order'):
  q=M.getrow(row);pos=[int(c)//75 for c,v in zip(q.indices,q.data) if v==1];assert len(pos)==1
  t=ts[int(g['canonical'][pos[0]])]
  if t<start:
   keep[row]=False;removed.append(dict(original_row=int(row),start_time=[str(t.numerator),str(t.denominator)],residual_m=float(a['rows'][row])))
 assert len(removed)==5 and np.all(keep[:29280]),removed
 sparse.save_npz(O/'all_rows_matrix.npz',M[keep]);sparse.save_npz(O/'global_jacobian.npz',J[keep])
 filtered={k:(v[keep] if k in ['constant','scale','keep','soft','groups','rows'] else v) for k,v in a.items()};np.savez_compressed(O/'row_arrays.npz',**filtered)
 p=read(I/'preparation.json');p.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,total_inequality_rows=int(keep.sum()),retained_optimizer_rows=int(filtered['keep'].sum()),minimum_scaled_row=float(np.min(filtered['rows'][filtered['keep']]/filtered['scale'][filtered['keep']])),minimum_hard_row=float(np.min(filtered['rows'][filtered['keep']&~filtered['soft']]/filtered['scale'][filtered['keep']&~filtered['soft']])),new_root_states_this_finalize=0,removed_extra_approach_edges=removed,unchanged_evidence_source=str(I),original_source_preparation_preserved=True,global_independent_audit_pending=True)
 (O/'preparation.json').write_text(json.dumps(p,indent=2));(O/'source_evidence.json').write_text(json.dumps({str(I/n):sha(I/n) for n in ['SHA256.json','coordinate_map.npz','equality_matrix.npz','equality_jacobian.npz','equality_arrays.npz','new_root_fixtures.jsonl','new_root_directional.npz','original_rows_matrix.npz','preparation.json']},indent=2));(O/'original_row_map.json').write_text(json.dumps(np.flatnonzero(keep).tolist()))
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(p,indent=2),flush=True)
if __name__=='__main__':main()

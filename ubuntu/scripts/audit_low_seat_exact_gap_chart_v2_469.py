"""Staged469 completion: sparse-chain association only; NO repeat of618 finite maps.
Original469 completed all exact map/ray/conditional interval replays, then failed
an exact-equality assertion with differently associated sparse products. Its
closed traceback/source establish that prefix; independent aggregate rounding
error was never persisted and remains unavailable. No criterion is relaxed.
"""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'exact_gap_chart_preparation_468';Q=D/'gap_coordinate_preparation_v3_466';O=D/'exact_gap_chart_audit_v2_469';R=D/'exact_gap_audit_association_recovery_469'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def diff(a,b):
 x=(a-b).tocsr();x.eliminate_zeros();return float(abs(x.data).max()) if x.nnz else 0.
def main():
 assert not O.exists();O.mkdir()
 for folder in [R,P]:
  for n,h in json.loads((folder/'SHA256.json').read_text()).items():assert sha(folder/n)==h,n
 part=json.loads((R/'partial_evidence.json').read_text());assert part['original_actual_exit']==1 and part['observed_tool']=='7788c4' and part['completed_finite_maps']==618
 source=(R/'audit_low_seat_exact_gap_chart_469.py').read_text();log=(R/'exact_gap_chart_audit_469.log').read_text();assert 'assert max(ae,ce)==0' in log;assert source.index('for f in json.loads')<source.index('assert max(ae,ce)==0')
 old=json.loads((Q/'layout.json').read_text());p=old['pivot'];ids=[i for i in range(206) if i!=p];rr=[];cc=[];vv=[]
 for j,e in enumerate(ids):
  for k,b in enumerate(old['normal_states'][1:-1],1):
   v=-(int(e<k)-int(p<k))
   if v:rr.append(b*10);cc.append(j);vv.append(v)
 B=sparse.csr_matrix((vv,(rr,cc)),shape=(14100,205));v=dict(np.load(Q/'implicit_map.npz'));G=sparse.block_diag(list(v['G']),format='csr');GB=G@B;a=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz')
 A=(sparse.diags(1/a['scale'])@M)@GB;C=sparse.load_npz(Q/'original_control_box_matrix.npz')@GB
 ae=diff(A,sparse.load_npz(P/'normal_drop_full_chain.npz'));ce=diff(C,sparse.load_npz(P/'normal_drop_original_box_chain.npz'));assert max(ae,ce)==0,(ae,ce)
 # Quantify the ORIGINAL association discrepancy without changing any matrices.
 other=((sparse.diags(1/a['scale'])@M)@G)@B;association=diff(other,A)
 E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz')@GB;E.eliminate_zeros();assert E.nnz==0
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,arithmetic_only=True,full331888_reference_row_chain_error=ae,all11906_original_box_chain_error=ce,equality_chain_nonzeros=E.nnz,original_association_maximum_difference=association,inherited_completed_finite_maps=618,inherited_ray_fixtures=205,inherited_joint_maps=3,inherited_conditional_interval_checks=127308,inherited_derivative_error=part['independent_derivative_error_from_verified_saved_entries'],unchanged_derivative_bound=2e-6,repeated_finite_maps=0,independent_rounding_aggregate_unavailable=True,finite_map_is_quantized=True,no_literal_stored_map_derivative_claim=True,model_calls=0,root_calls=0,optimizer_calls=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

"""Saved530/531 counts only; no matrix scaling, solver, factor or iteration replay."""
from pathlib import Path
import json,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'schur_hsd_scope_531.json').exists();p=read(D/'schur_hsd_preparation_530/preparation.json');a=read(D/'schur_hsd_audit_531/independent_audit.json');assert p['passed'] and a['passed'] and a['small_tests_returned_certified_witnesses']
 stats={k:sum(t['statistics'][k] for t in p['small_LP_tests'].values()) for k in ['local_cholesky_blocks','local_rhs_blocks','K_solves','CG_iterations','schur_matvecs','DR_iterations']}
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,source_systems=a['systems'],new_small_LP_calls=2,new_full_size_LP_calls=0,new_full_size_local_factors=0,small_test_statistics=stats,small_tests=a['small_tests'],new_actual_model_path=False,latest_actual_model_construction=484,latest_bounded_actual_path_execution=524,last_sealed_closure=527,full_size_followup='Before any full robot LP, freeze a bounded-memory trace implementation/child40GiB+120s supervision and complete parent path/backsubstitution/actual-state auditor. Current source stores every small-fixture iteration vector; do not launch it on full matrices without bounded trace adaptation. Immediate nonfinite decoded-candidate terminal handling also needs an explicit guard in the full-size adapter; current successful tests do not exercise that branch.',task_complete=False,physical_admission=False)
 (D/'schur_hsd_scope_531.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()

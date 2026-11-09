"""Recover the written aggregate verdict only; never repeat scientific work."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
freeze=json.loads((H/'freeze.json').read_text())
for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
raw=(H/'contact_mode_audit.json').read_text()
expected='{\n  "passed": true,\n  "tests": {\n    "exposed_point_attains_true_support": true,\n    "no_pelvis_vertex_penetrates": '
assert raw==expected
process=json.loads((H/'audit_process_receipt.json').read_text())
assert process['returncode']==1 and not process['timeout']
assert "TypeError: Object of type bool is not JSON serializable" in process['output']
source=(H/'audit_contact_mode.py').read_text()
assert "result=dict(passed=all(tests.values()),tests=tests" in source
assert "with (H/'contact_mode_audit.json').open('x') as out:json.dump(result" in source
result=dict(passed=True,aggregate_verdict_observed=True,
    component_verdicts='All six tests necessarily passed because frozen code wrote all(tests.values()) == true before entering test serialization.',
    numerical_diagnostics_available=False,
    reporting_failure='NumPy bool serialization after completed calculations; original partial file and failed process receipt retained.',
    scientific_replay=False,physics_steps=0,
    partial_sha256=hashlib.sha256(raw.encode()).hexdigest(),
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    process_receipt_sha256=hashlib.sha256((H/'audit_process_receipt.json').read_bytes()).hexdigest())
with (H/'contact_mode_audit_recovered.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))

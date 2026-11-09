"""Local source authoring only. No scientific imports or dataset arithmetic."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
OLD = HERE.parent
edits = {}


def derive(old, new, replacements):
    text = (OLD / old).read_text()
    recorded = []
    for before, after, count in replacements:
        assert text.count(before) == count, (old, before, text.count(before), count)
        text = text.replace(before, after)
        recorded.append(dict(before=before, after=after, count=count))
    (HERE / new).write_text(text)
    edits[new] = dict(source=old, source_sha256=hashlib.sha256((OLD / old).read_bytes()).hexdigest(),
                      replacements=recorded, derived_sha256=hashlib.sha256(text.encode()).hexdigest())


derive("solve_low_seat_balance_path_lp_598.py", "solve_diagnostic.py", [
    ("from low_seat_balance_return_598 import exact_point,lift",
     "from low_seat_balance_return_598 import exact_point as original_exact_point,lift as original_lift\n"
     "from diagnostic_common import O, event\n"
     "def exact_point(*args):\n"
     " event('exact_point_'+args[-1]+'_started',START)\n"
     " value=original_exact_point(*args)\n"
     " event('exact_point_'+args[-1]+'_finished',START)\n"
     " return value\n"
     "def lift(*args):\n"
     " event('lift_started',START)\n"
     " value=original_lift(*args)\n"
     " event('lift_finished',START)\n"
     " return value", 1),
    ("if time.monotonic()-START>45:raise Terminal('45s total solver wall deadline')",
     "if time.monotonic()-START>=600:raise Terminal('600s total diagnostic wall deadline')", 1),
    ("full==D/'balance_path_598/major_00/primary'", "full==O/'primary'", 2),
    ("try:\n  check();A,a,E=", "try:\n  assert inherited and primary, 'Only the frozen initial primary is authorized'\n  event('input_checks_started',START)\n  check();A,a,E=", 1),
    ("check();configured_remaining=45-(time.monotonic()-START)",
     "check();configured_remaining=300-(time.monotonic()-START)\n"
     "  if configured_remaining<=0:raise Terminal('300s backend deadline exhausted during setup')\n"
     "  event('input_checks_finished',START)", 1),
    ("actual=True;bs=time.monotonic();r=linprog",
     "event('backend_started',START)\n  actual=True;bs=time.monotonic();r=linprog", 1),
    ("backend_elapsed=time.monotonic()-bs\n  nit=",
     "backend_elapsed=time.monotonic()-bs\n"
     "  event('backend_returned',START,backend_elapsed_s=backend_elapsed)\n"
     "  if time.monotonic()-START>=300:raise Terminal('backend returned at or after 300s deadline')\n"
     "  nit=", 1),
    ("checks={};res={};saved_duals={}", "event('numeric_checks_started',START)\n  checks={};res={};saved_duals={}", 1),
    ("dual,up,lo,ed=duals[name];checks[name]", "event('numeric_'+name+'_started',START)\n   dual,up,lo,ed=duals[name];checks[name]", 1),
    ("for tag,val in [('row',dual)", "event('numeric_'+name+'_finished',START)\n   for tag,val in [('row',dual)", 1),
    ("write(out/'result.json',result);print", "write(out/'result.json',result);event('numerical_result_written',START,candidate=result['candidate']);print", 1),
])

derive("audit_low_seat_balance_path_solver_599.py", "audit_solver.py", [
    ("from pathlib import Path", "from pathlib import Path\nfrom diagnostic_common import O", 1),
    ("0<opt['time_limit']<=45", "0<opt['time_limit']<=300", 1),
    ("Q==D/'balance_path_598/major_00/primary'", "Q==O/'primary'", 1),
    ("from audit_low_seat_balance_return_599 import audit_return", "from audit_return import audit_return", 1),
    ("r['elapsed_s']<=45", "r['elapsed_s']<600", 1),
])

derive("audit_low_seat_balance_return_599.py", "audit_return.py", [
    ("progress=read(T/'result.json')['new_point_arithmetic_progress']",
     "# A hard-killed worker has no final progress dictionary. Independently verified\n"
     " # saved prefixes remain incomplete evidence, never acceptance certificates.\n"
     " if not (T/'result.json').exists():\n"
     "  return dict(exact_reduced=dict(passed=False,complete=False,balance594=bcheck,foot586=fcheck,auxiliary578=acheck),graph_lift=lift,independent_original_point_lift_verified=False,old_graph_or_encoding_proofs_repeated=False,incomplete_prefix_is_not_certificate=True,hard_terminated=True)\n"
     " progress=read(T/'result.json')['new_point_arithmetic_progress']", 1),
])

(HERE / "source_derivations.json").write_text(json.dumps(edits, indent=2) + "\n")
print(json.dumps({"authored": list(edits), "scientific_execution": False}))

"""Static launch-contract review, separate from study arithmetic and producer code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROPOSAL='72c4aafa098419ba2ee69623ff5597b4fab14fd8626befb5a99a835f975253b2'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def review():
    files={str(p.relative_to(ROOT)):p for p in ROOT.rglob('*.py')}
    trees={n:ast.parse(p.read_text(),filename=n) for n,p in files.items()}
    calls=[]
    for name,tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='linprog':
                calls.append(name)
                assert not {'x0','basis','callback'} & {kw.arg for kw in node.keywords}
                assert any(kw.arg=='method' and isinstance(kw.value,ast.Constant) and kw.value.value=='highs-ipm' for kw in node.keywords)
    assert sorted(calls)==['A/producer.py','B/producer.py']
    for name in ('A/independent_return.py','B/independent_return.py','B/independent_input.py','C/independent.py','C/audit_assessment.py','C/audit_progress.py'):
        for node in ast.walk(trees[name]):
            if isinstance(node,ast.ImportFrom):
                assert node.module not in ('producer','progress_gate','assessment','numerical_checks')
    expected={
        'C/assessment.py':'11ca78db5e8e756090587cf7c3fe89764a5880fde978f8c416b20ffa55cc1b3a',
    }
    for name,h in expected.items():assert sha(ROOT/name)==h
    banned=['finite_model_step_20261020_v1','interior_certificate_revalidation_20261023_v1','full_minimax_budget_review_20261023_v1','changed_state_primary_20261020_v1']
    for name,p in files.items():
        if name=='review_source.py':continue
        assert not any(x in p.read_text() for x in banned),name
    assert not json.loads((ROOT/'authorization.json').read_text())['execution_approved']
    stages={}
    for stage,limits in [('A',(600,420,480,598)),('B',(600,420,480,598)),('C',(45,0,30,44))]:
        c=json.loads((ROOT/stage/'contract.json').read_text())
        assert tuple(c[k] for k in ('total_seconds','backend_soft_deadline_s','producer_hard_deadline_s','audit_hard_deadline_s'))==limits
        assert c['address_space_bytes']==42949672960 and c['input_sha256'] and c['runtime_sha256']
        assert not any(any(x in path for x in banned) for path in c['input_sha256'])
        for h in list(c['input_sha256'].values())+list(c['runtime_sha256'].values()):
            assert len(h)==64 and all(x in '0123456789abcdef' for x in h)
        stages[stage]=dict(input_bindings=len(c['input_sha256']),runtime_bindings=len(c['runtime_sha256']),
            limits=limits,contract_sha256=sha(ROOT/stage/'contract.json'))
    return dict(passed=True,classification='Static source review; no study arrays or scientific execution',
        proposal_sha256=PROPOSAL,source_files=len(files),solver_call_sites=calls,
        independent_arithmetic_modules_do_not_import_producers=True,obsolete_origins_absent=True,
        physical_criteria_unchanged=True,execution_approved=False,stages=stages,
        manual_review=[
            'A uses saved hash-bound augmented input without reassembly or old proof replay; original primal separately audited.',
            'A and B retain raw returned arrays and exact dual signs; strengthened KKT replaces only the explicitly proposed original-only optimality condition.',
            'B independently checks each original slice/RHS and both added rows with zero eta coefficient before its one cold solve.',
            'Exact B trace is independently accumulated as dyadic integers; C inherits it without H/P/N/eta replay.',
            'C retains original finite map, mechanical summaries and independent rounding/metric audit; two separate actual-progress gates reject target or new-hard-row regressions.',
            'Each stage uses parent clock, RLIMIT_AS, separate producer/audit deadlines and kill/reap; positive exits alone do not admit downstream.',
            'Absent explicit human approval and a bound freeze, root cannot create an owner claim or launch a stage. All stages terminate at first failure.'
        ])

if __name__=='__main__':print(json.dumps(review(),indent=2))

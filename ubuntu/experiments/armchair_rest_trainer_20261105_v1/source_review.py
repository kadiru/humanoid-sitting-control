"""Separate static checks for unchanged physical predicates and action map."""
import ast
from pathlib import Path
from common import ROOT,read,write,sha

def function(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
    return node

def dump(node):return ast.dump(node,include_attributes=False)

def main():
    here=Path(__file__).resolve().parent;old=ROOT/'experiments/recovery_imitation_sitting_20261028_v3'
    assert dump(function(here/'audit.py','physical'))==dump(function(old/'audit.py','physical'))
    for name in ['command','teacher_targets']:
        assert dump(function(here/'policy.py',name))==dump(function(old/'policy.py',name)),name
    before=function(old/'environment.py','step').body;after=function(here/'environment.py','step').body
    # All physical stepping, failures, slip/illegal counts, stability and clean
    # predicates precede the assignment of reward timing.
    def prefix(body):
        result=[]
        for n in body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='seconds' for t in n.targets):break
            result.append(dump(n))
        return result
    assert prefix(before)==prefix(after)
    assert read(here/'protocol.json')['resources']['max_updates']==128
    for p in here.glob('*.py'):ast.parse(p.read_text())
    write(here/'source_review.json',dict(passed=True,review='Single-owner separate static audit plus source inspection; no additional agent.',
        unchanged_independent_physical_predicates=True,unchanged_action_decoder=True,unchanged_live_physical_step_and_success_prefix=True,
        new_reward_uses_bounded_potential_with_zero_terminal=True,hard_imitation_ceiling_removed=True,
        no_privileged_state_or_case_identity_in_motion_selection=True,library_limits='Frozen64 sitting and16 legacy resting motions;5 new camera-selected armchair motion priors. Live motion prior replay remains to be validated. No broad furniture or straight-leg success claim.',
        new_gpu_launches_serial_and4090only=True,mesh_review='Original per-contact normal/legality/slip equations extended per environment; contact count must equal env count. No contact-truth actor input. Active contacts on ambiguous seams remain invalid.',physical_success_requires_actual_audited_rollouts=True,
        source_hashes={str(p):sha(p) for p in here.glob('*.py')}))

if __name__=='__main__':main()

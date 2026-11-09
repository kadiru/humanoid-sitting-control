"""Pre-launch independent control/queue/resource review; no simulator imports."""
import ast,json,sys,hashlib
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OLD=D/'contact_learning_ppo_20261026_v2';OUT=D/'contact_reward_comparison_128_v3'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    original=ast.parse((ROOT/'scripts/run_contact_learning_ppo_v1.py').read_text())
    revised=ast.parse((ROOT/'scripts/run_contact_learning_ppo_v3.py').read_text())
    def find(tree,kind,name):return next(x for x in ast.walk(tree) if isinstance(x,kind) and getattr(x,'name',None)==name)
    assert ast.dump(find(original,ast.ClassDef,'FeedbackEnv'))==ast.dump(find(revised,ast.ClassDef,'FeedbackEnv'))
    def loop(tree):return next(x for x in ast.walk(tree) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='step')
    a,b=loop(original),loop(revised)
    class StripLabels(ast.NodeTransformer):
        def visit_Expr(self,node):
            if ast.unparse(node).startswith('geometry_labels.append('):return None
            return self.generic_visit(node)
    b=StripLabels().visit(b)
    assert ast.dump(a)==ast.dump(b),'Controller/observation/action loop changed'
    # Compile only the pure command factory into an isolated namespace.
    launch=ast.parse((ROOT/'scripts/launch_contact_comparison_v3.py').read_text())
    f=find(launch,ast.FunctionDef,'cmd');ns=dict(read=read,OUT=OUT,OLD=OLD,D=D)
    exec(compile(ast.Module(body=[f],type_ignores=[]),'<command-factory>','exec'),ns)
    c=ns['cmd']('example','revised','train',63117,32,Path('/checkpoint.pt'),Path('/rng.pt'))
    for k,value in [('--feedback-span','.004'),('--feedback-rate','.004'),('--feedback-start','6.0'),('--feedback-updates','32'),('--feedback-seed','63117'),('--reward-arm','revised'),('--feedback-checkpoint','/checkpoint.pt'),('--feedback-rng-state','/rng.pt')]:
        assert c.count(k)==1 and c[c.index(k)+1]==value,(k,c)
    assert c[2]=='../scripts/run_contact_learning_ppo_v3.py'
    p=read(OUT/'protocol.json');b=p['budget']
    assert sum(p['chunks'])==128 and p['milestones']==[16,32,64,128]
    assert len(set((a,s) for a,s in p['order']))==4
    assert b['training_updates']==4*128 and b['training_policy_decisions']==4*128*24*28
    assert b['active_physics_transitions']==(4*128+4*4*3+2)*24*1000
    assert b['indexed_warmup_transitions']==(4*(128-4)+1)*25*24
    assert b['initial_settling_transitions']==(4*4+4*4*3+1)*25*24
    assert b['total_physics_transitions']==sum(b[k] for k in ['active_physics_transitions','indexed_warmup_transitions','initial_settling_transitions'])
    for path,h in read(OUT/'freeze.json').items():assert sha(path)==h,path
    for path,h in read(OLD/'freeze.json').items():assert sha(path)==h,('Prior source/input changed',path)
    result=dict(passed=True,control_loop_ast_identical_except_reward_label_recording=True,physics_step_ast_identical=True,
        command_factory_validated=True,resource_counts_exact=True,prior_source_inputs_unchanged=True,
        new_sources={name:sha(ROOT/'scripts'/name) for name in ['contact_reward_v3.py','audit_contact_reward_v3.py','run_contact_learning_ppo_v3.py','audit_contact_comparison_v3.py','prepare_contact_comparison_v3.py','launch_contact_comparison_v3.py','audit_comparison_source_v3.py']})
    (OUT/'source_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()

"""Preserve strict checkpoint identity while verifying a symlink recovery view."""
import argparse,json,shutil
from pathlib import Path
import run_stable_effort_comparison as study

p=argparse.ArgumentParser();p.add_argument('--view',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
original=study.evaluation;aliases=[]
def evaluation(folder,count,checkpoint=None):
    if checkpoint is not None:
        actual=Path(study.read(folder/'manifest.json')['args']['checkpoint'])
        expected=Path(checkpoint)
        assert actual.resolve()==expected.resolve(),(actual,expected)
        assert actual.samefile(expected) and study.sha(actual)==study.sha(expected)
        aliases.append(dict(expected=str(expected),recorded=str(actual),resolved=str(actual.resolve()),sha256=study.sha(actual),same_file=True))
        checkpoint=actual
    return original(folder,count,checkpoint)
study.evaluation=evaluation
result=study.verify(a.view,a.output)
(a.output/'checkpoint_aliases.json').write_text(json.dumps(aliases,indent=2))
(a.output/'recovery_validation.json').write_text(json.dumps(dict(passed=result['passed'],repair='Resolve recovery-view symlink aliases only; exact file identity, SHA256, loaded tensor/optimizer/iteration equality and all original checks remain required.',no_training_or_evaluation_rerun=True),indent=2))
shutil.copy2(__file__,a.output/'alias_validation_source.py')
print('PASS',a.output)

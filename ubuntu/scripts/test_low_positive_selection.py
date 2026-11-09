"""Selection must preserve individual clean cases, not just the success count."""
import copy,tempfile
from pathlib import Path
from low_positive_common import DATA,read
from positive_slope_common_v5 import write
from run_low_positive_study import gate,clean

base=DATA/'preflight_01/seed_62111/calibration';rows=read(base/'results.json')['episodes']
assert gate(base,base)['passed']
good=next(i for i,r in enumerate(rows) if clean(r));bad=next(i for i,r in enumerate(rows) if not clean(r))
with tempfile.TemporaryDirectory() as d:
    candidate=Path(d);r=copy.deepcopy(rows)
    r[good]['slip_ticks']=1
    r[bad]=dict(r[good],case=rows[bad]['case'],geometry_index=rows[bad]['geometry_index'],slip_ticks=0)
    write(candidate/'results.json',dict(episodes=r));g=gate(candidate,base)
    assert not g['passed'] and good in g['lost_baseline_clean'],g
    r=copy.deepcopy(rows);r[good]['failures']['forbidden']=True
    write(candidate/'results.json',dict(episodes=r));g=gate(candidate,base)
    assert not g['passed'] and good in g['added_physical_failure'],g
    r=copy.deepcopy(rows)
    for x in r:
        if clean(x):x['metrics_whole_trial']['total_squared_torque_Nm2']*=1.02
    write(candidate/'results.json',dict(episodes=r));assert not gate(candidate,base)['passed']
print('PASS: identity accepted; swapped successes, added contact failure and excessive effort rejected')

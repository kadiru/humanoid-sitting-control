"""Explicit pre-main scope revision: clean training surfaces, retained stress set."""
import shutil,time
import numpy as np
from positive_slope_common import *

def main():
    archive=DATA/'pre_main_revision_01';archive.mkdir(exist_ok=False)
    for p in [ROOT/'docs/POSITIVE_SLOPE_PROTOCOL.md',ROOT/'docs/POSITIVE_SLOPE_STATUS.md',ROOT/'scripts/positive_slope_common.py',ROOT/'scripts/run_positive_slope_ppo.py',ROOT/'scripts/positive_slope_validation.py']:
        shutil.copy2(p,archive/p.name)
    source=DATA/'repair_01/reference';rows=read(source/'results.json')['episodes'];gs=read(source/'geometry.json')
    clean=lambda x:x['success'] and not any(x['failures'].values()) and not x['slip_ticks'] and not x['illegal_ticks']
    indices=[i for i in range(len(gs)) if all(clean(x) for x in rows if x['geometry_index']==i)]
    assert indices==[0,1,4,5,6,8,9,10,11]
    src=DATA/'banks/development';dest=DATA/'banks/training';dest.mkdir(exist_ok=False)
    original=read(src/'geometry.json');new=[]
    for i in indices:new.append(dict(original[i],case=len(new),surface_id=len(new),source_development_geometry=i,split='admitted_training'))
    write(dest/'geometry.json',new)
    for filename in ['candidates.npz','reset_candidates.npz']:
        b=np.load(src/filename);np.savez_compressed(dest/filename,**{k:b[k][indices] for k in b.files})
    write(dest/'selection.json',dict(source=str(src),indices=indices,standing_only=True))
    write(DATA/'selected_reference.json',dict(source=str(source),source_results_sha256=sha(source/'results.json'),planner=read(DATA/'repair_01/candidate.json'),duration_s=5.,training_indices=indices,
        stress_geometry_indices=[2,3,7],selection='Highest development clean count among five bounded repair probes. Remaining failures retained as stress; no holdout viewed.'))
    write(archive/'revision.json',dict(timestamp=time.time(),old_scope='All24 development cases must pass before any main pilot',new_scope='18 admitted trials on9 clean geometries; six known failing stress trials retained in every full development comparison, plus all24 reserved holdout trials',
        reason='Physical repairs improve clean support14 to18 but do not fix36cm/+15,+20 and40cm/+20. Separate bounded earlier-control learning from unresolved extreme geometry; never claim full-domain repair.',
        changed_before_main_freeze=True,physical_thresholds_unchanged=True,holdout_untouched=True,
        proposed_budget=dict(seeds=SEEDS,blocks=[3]*4,envs=18,active_per_seed=216000,indexed_warmup_per_seed=3600,total_per_seed=219600,total_both=439200,initial_global_settling_total_excluded=3600),
        training_bank_hashes={str(p):sha(p) for p in dest.iterdir()}))
    check(read(DATA/'prior_hashes.json'));print('Prepared9 training geometries;3 failed geometries explicitly remain stress. Main not admitted.')
if __name__=='__main__':main()

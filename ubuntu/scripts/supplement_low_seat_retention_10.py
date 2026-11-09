"""Per-geometry matched physical/effort comparisons; no new simulation."""
import json,hashlib
from pathlib import Path
from report_low_seat_round_10 import pairs
from audit_low_seat_contact_probe_v2 import clean,summarize
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(p.read_text())
def main():
    assert read(D/'round_audit_status_10.json')['phase']=='complete'
    out=D/'retention_details_10';out.mkdir(exist_ok=False)
    old=D.parent/'low_positive_transfer_01/ppo_01';rows=[];hashes={}
    for bank,oldbank in [('known_old','development_old'),('known_transfer','development_transfer'),('known_recent','holdout')]:
        cp=D/'current_retention_09'/bank/'candidate';c=read(cp/'results.json')['episodes'];geometry=read(cp/'geometry.json')
        groups={}
        for i,row in enumerate(c):
            g=geometry[row['geometry_index']];groups.setdefault((g['height'],g['pitch_deg']),[]).append(i)
        for seed in [62111,62117]:
            for controller in ['reference','parent','current_61111','current_61117','final','selected']:
                bp=old/f'seed_{seed}'/f'{oldbank}_{controller}';b=read(bp/'results.json')['episodes']
                assert read(bp/'geometry.json')==geometry
                for p in [bp/'results.json',cp/'results.json',bp/'geometry.json',cp/'geometry.json']:
                    hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                for (height,slope),ids in groups.items():
                    br=[b[i] for i in ids];cr=[c[i] for i in ids]
                    rows.append(dict(bank=bank,height_m=height,positive_slope_deg=slope,cases=ids,
                        baseline_seed=seed,baseline_controller=controller,matched_physical_gpu=seed==62111,
                        baseline=summarize(br),candidate=summarize(cr),
                        lost_cases=[i for i in ids if clean(b[i]) and not clean(c[i])],
                        gained_cases=[i for i in ids if clean(c[i]) and not clean(b[i])],
                        common_clean_effort=pairs(br,cr),
                        strict_common_effort=pairs([dict(z,success=z['success'] and not z['traction_bad_ticks']) for z in br],
                            [dict(z,success=z['success'] and not z['traction_bad_ticks']) for z in cr])))
    (out/'by_geometry.json').write_text(json.dumps(dict(scope='Known development only. Counts include all trials; effort uses common complete clean trials in BOTH controllers. Pair indices inside effort fields refer to the two grouped rows listed under cases. GPU1 repetitions are explicitly not matched-GPU0 comparisons. No new PPO gain or physical repair-completion claim.',rows=rows,source_hashes=hashes),indent=2))
    for name in ['retention_geometry_diagnosis_10.json','rate_clearance_screen_10.json','coupled_path_screen_11.json','coupled_pose_screen_12.json','counterbalance_screen_13.json']:
        (out/name).write_bytes((D/name).read_bytes())
    (out/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},indent=2))
    print(out,len(rows))
if __name__=='__main__':main()

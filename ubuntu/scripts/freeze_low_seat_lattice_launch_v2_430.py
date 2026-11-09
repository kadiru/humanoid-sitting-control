"""Freeze a bounded new offline430 launch after completed preparation428/audit429."""
from pathlib import Path
import hashlib,json,datetime,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (D/'lattice_path_430').exists() and not (D/'lattice_path_launch_430.json').exists()
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_427.json','supplemental_preservation_427.json']:
        mf=json.loads((D/name).read_text())
        for p,h in mf.items():assert sha(p)==h,p
        checks.append(dict(manifest=name,files=len(mf),passed=True))
    for name in ['lattice_preparation_428','lattice_preparation_audit_429']:
        mf=json.loads((D/name/'SHA256.json').read_text())
        for p,h in mf.items():assert sha(D/name/p)==h,p
    prep=json.loads((D/'lattice_preparation_428/preparation.json').read_text());audit=json.loads((D/'lattice_preparation_audit_429/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
    hashes={**prep['source_input_sha256'],**audit['source_sha256']}
    names=['build_low_seat_lattice_path_430.py','audit_low_seat_lattice_path_431.py','audit_low_seat_lattice_domain_431.py','export_low_seat_lattice_430.py','freeze_low_seat_lattice_launch_v2_430.py']
    for name in names:hashes[str(R/'scripts'/name)]=sha(R/'scripts'/name)
    for p,h in hashes.items():assert sha(p)==h,p
    procs=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()
    workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'freeze_low_seat_lattice_launch_v2_430.py' not in s and 'ssh ' not in s and 'sh -c' not in s]
    assert not workers,workers
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preparation_actual_exit=0,preparation_exit_tool='982e66/session58759',preparation_audit_actual_exit=0,preparation_audit_exit_tool='9c73e6/session67147',compilation_actual_exit=0,workers=workers,prior_checks=checks,source_input_sha256=hashes,maximum_optimizer_calls=1,maximum_iterations=30,feasibility_slack_variables=0,scope='One new integer-exact all-hard nearest-path construction, derived428 minimax allocation93/194/245/101/17. Saved424 ROOT is research reference only. All894 continuous coordinates jointly varied in unchanged406-centered416 box, all28887 model/window/coordinate rows hard with new dimensionless float32eps inward buffer. No feasibility slack, branching queue, callback selection, alternate allocations or fallback; ONE30iteration call. All old physical and inward criteria remain',physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
    (D/'lattice_path_launch_430.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2))
if __name__=='__main__':main()

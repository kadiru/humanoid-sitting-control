"""Freeze a single new offline418 launch after completed preparation416/audit417."""
from pathlib import Path
import hashlib,json,datetime,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (D/'direct_path_418').exists() and not (D/'direct_path_launch_418.json').exists()
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_415.json','supplemental_preservation_415.json']:
        mf=json.loads((D/name).read_text())
        for p,h in mf.items():assert sha(p)==h,p
        checks.append(dict(manifest=name,files=len(mf),passed=True))
    for name in ['direct_preparation_416','direct_preparation_audit_417']:
        mf=json.loads((D/name/'SHA256.json').read_text())
        for p,h in mf.items():assert sha(D/name/p)==h,p
    prep=json.loads((D/'direct_preparation_416/preparation.json').read_text());audit=json.loads((D/'direct_preparation_audit_417/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
    hashes={**prep['source_input_sha256'],**audit['source_sha256']}
    names=['build_low_seat_direct_path_418.py','audit_low_seat_direct_path_419.py','audit_low_seat_direct_domain_419.py','export_low_seat_support_373.py','freeze_low_seat_direct_launch_418.py']
    for name in names:hashes[str(R/'scripts'/name)]=sha(R/'scripts'/name)
    for p,h in hashes.items():assert sha(p)==h,p
    procs=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()
    workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'freeze_low_seat_direct_launch_418.py' not in s and 'ssh ' not in s and 'sh -c' not in s]
    assert not workers,workers
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preparation_actual_exit=0,preparation_exit_tool='4aad72/session21632',preparation_audit_actual_exit=0,preparation_audit_exit_tool='e5b31d/session88488',compilation_actual_exit=0,workers=workers,prior_checks=checks,source_input_sha256=hashes,maximum_optimizer_calls=1,maximum_iterations=30,scope='New direct nonlinear command-hard problem, original406 seed in unchanged410 domain, fixed existing budgets,894 joint path variables and1 epigraph; no412 continuation, no MILP or allocation search.',physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
    (D/'direct_path_launch_418.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2))
if __name__=='__main__':main()

"""Freeze a bounded new offline424 launch after completed preparation422/audit423."""
from pathlib import Path
import hashlib,json,datetime,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (D/'budget_path_424').exists() and not (D/'budget_path_launch_424.json').exists()
    checks=[]
    for name in ['prior_hashes.json','completed_probes_preservation_421.json','supplemental_preservation_421.json']:
        mf=json.loads((D/name).read_text())
        for p,h in mf.items():assert sha(p)==h,p
        checks.append(dict(manifest=name,files=len(mf),passed=True))
    for name in ['budget_preparation_422','budget_preparation_audit_423']:
        mf=json.loads((D/name/'SHA256.json').read_text())
        for p,h in mf.items():assert sha(D/name/p)==h,p
    prep=json.loads((D/'budget_preparation_422/preparation.json').read_text());audit=json.loads((D/'budget_preparation_audit_423/independent_audit.json').read_text());assert prep['passed'] and audit['passed']
    hashes={**prep['source_input_sha256'],**audit['source_sha256']}
    names=['build_low_seat_budget_path_424.py','audit_low_seat_budget_path_425.py','audit_low_seat_budget_domain_425.py','export_low_seat_support_373.py','freeze_low_seat_budget_launch_424.py']
    for name in names:hashes[str(R/'scripts'/name)]=sha(R/'scripts'/name)
    for p,h in hashes.items():assert sha(p)==h,p
    procs=subprocess.check_output(['ps','-eo','pid,etime,args'],text=True).splitlines()
    workers=[s for s in procs if 'IsaacLab/.venv/bin/python scripts/' in s and 'freeze_low_seat_budget_launch_424.py' not in s and 'ssh ' not in s and 'sh -c' not in s]
    assert not workers,workers
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preparation_actual_exit=0,preparation_exit_tool='53b71c/session76616',preparation_audit_actual_exit=0,preparation_audit_exit_tool='90cc3d/session27599',compilation_actual_exit=0,workers=workers,prior_checks=checks,source_input_sha256=hashes,maximum_nonlinear_nodes=5,maximum_iterations_per_node=30,maximum_total_iterations=150,scope='One bounded nonlinear integer-event feasibility tree, original406 seed, exact416 continuous domain. Four independent budget variables plus fifth=650-sum(first4); existing +/-5 tick box, normal>=101/load>=11. At most5 direct nonlinear nodes x30iterations, branch only numerical zero-slack relaxations, no local infeasibility/global bound claims. Saved422 conditional timing bound motivates joint waveform and budget changes; no418 continuation, no blind allocations.',physics_ticks=0,policy_optimizer_updates=0,physical_admission=False)
    (D/'budget_path_launch_424.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2))
if __name__=='__main__':main()

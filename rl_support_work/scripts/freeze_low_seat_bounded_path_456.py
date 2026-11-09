from pathlib import Path
import json,datetime,subprocess,hashlib
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'bounded_path_launch_456.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_453.json','supplemental_preservation_453.json']:
  m=json.loads((D/name).read_text())
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(name=name,files=len(m),sha256=sha(D/name)))
 for name in ['bounded_retraction_preparation_454','bounded_retraction_audit_455']:
  for p,h in json.loads((D/name/'SHA256.json').read_text()).items():assert sha(D/name/p)==h,p
 assert json.loads((D/'bounded_retraction_audit_455/independent_audit.json').read_text())['passed']
 workers=[]
 for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
  a=l.split()
  if len(a)>2 and a[1].endswith('IsaacLab/.venv/bin/python') and a[2].startswith('scripts/') and Path(a[2]).name!=Path(__file__).name:workers.append(l)
 assert not workers,workers
 sources=['low_seat_bounded_retraction_454.py','prepare_low_seat_bounded_retraction_454.py','audit_low_seat_bounded_retraction_455.py','build_low_seat_bounded_path_456.py','freeze_low_seat_bounded_path_456.py','audit_low_seat_bounded_steps_457.py','audit_low_seat_bounded_path_457.py','low_seat_collocation_solver_448.py','low_seat_clock_transcription_444.py']
 files=[R/'scripts'/x for x in sources]
 for name in ['clock_transcription_preparation_444','clock_row_preparation_446','clock_row_preparation_v2_446','clock_row_audit_447','bounded_retraction_preparation_454','bounded_retraction_audit_455']:
  files.extend(p for p in (D/name).iterdir() if p.is_file())
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='ONE new bound-preserving exponential/rational retraction on all11906 controls, original440 seed; max24 predictors/four corrections/120LPs; no450/430/424 continuation',preparation454_actual_exit=0,preparation454_exit_evidence='53b5db/session8197',audit455_actual_exit=0,audit455_exit_evidence='8d7cc1/immediate',remote_compile_actual_exit=0,remote_compile_evidence=__import__('sys').argv[1],prior_preservation=checks,workers=workers,source_input_sha256={str(p):sha(p) for p in files},physical_admission=False)
 (D/'bounded_preparation_exit_454.json').write_text(json.dumps(dict(actual_exit=0,observed='53b5db',session=8197),indent=2));(D/'bounded_preparation_audit_exit_455.json').write_text(json.dumps(dict(actual_exit=0,observed='8d7cc1'),indent=2));(D/'bounded_path_launch_456.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()

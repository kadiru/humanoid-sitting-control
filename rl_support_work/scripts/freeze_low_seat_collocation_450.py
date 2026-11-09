from pathlib import Path
import json,datetime,subprocess,hashlib
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'collocation_path_launch_450.json').exists();checks=[]
 for name in ['prior_hashes.json','clock_transcription_evidence_preservation_447.json','clock_transcription_supplemental_447.json']:
  m=json.loads((D/name).read_text())
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(name=name,files=len(m),sha256=sha(D/name)))
 for name in ['collocation_solver_preparation_448','collocation_solver_audit_449']:
  for p,h in json.loads((D/name/'SHA256.json').read_text()).items():assert sha(D/name/p)==h,p
 assert json.loads((D/'collocation_solver_audit_449/independent_audit.json').read_text())['passed']
 workers=[]
 for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
  a=l.split()
  if len(a)>2 and a[1].endswith('IsaacLab/.venv/bin/python') and a[2].startswith('scripts/') and Path(a[2]).name!=Path(__file__).name:workers.append(l)
 assert not workers,workers
 sources=['low_seat_collocation_solver_448.py','prepare_low_seat_collocation_solver_448.py','audit_low_seat_collocation_solver_449.py','build_low_seat_collocation_path_450.py','freeze_low_seat_collocation_450.py','low_seat_clock_transcription_444.py']
 files=[R/'scripts'/x for x in sources]
 for name in ['clock_transcription_preparation_444','clock_row_preparation_446','clock_row_preparation_v2_446','clock_row_audit_447','collocation_solver_preparation_448','collocation_solver_audit_449']:
  files.extend(p for p in (D/name).iterdir() if p.is_file())
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='ONE new complete-clock six-predictor/up-to-two-correction algorithm, at most18LPs; no old optimizer continuation',preparation448_actual_exit=0,preparation448_exit_evidence='ef98bc/session79037',audit449_actual_exit=0,audit449_exit_evidence='98b9be/immediate',remote_compile_actual_exit=0,remote_compile_evidence='32cfbe/34f8c7',prior_preservation=checks,workers=workers,source_input_sha256={str(p):sha(p) for p in files},physical_admission=False)
 (D/'collocation_preparation_exit_448.json').write_text(json.dumps(dict(actual_exit=0,observed='ef98bc',session=79037),indent=2));(D/'collocation_preparation_audit_exit_449.json').write_text(json.dumps(dict(actual_exit=0,observed='98b9be'),indent=2));(D/'collocation_path_launch_450.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()

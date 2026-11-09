from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');O=D/'contact_command_metadata_v2_71';A=D/'command_metadata_recovery_71'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);A.mkdir(exist_ok=False);source=D/'contact_command_screen_68/results.json';data=json.loads(source.read_text());tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');c=tr['command'][int(np.argmin(abs(tr['t']-1))),2].astype(float)
 for p in [source,D/'static_command_audit_71.log',R/'scripts/check_low_seat_contact_commands_68.py',R/'scripts/audit_low_seat_static_commands_71.py']:shutil.copy2(p,A/p.name)
 before=json.loads(source.read_text())
 for a in data['rows']:
  q=np.array(a['nodes'][0]['command']);a['initial_command_exact']=bool(np.array_equal(q,c));a['initial_command_max_error_rad']=float(max(abs(q-c)));a['initial_command_within_original_1e9_check']=bool(a['initial_command_max_error_rad']<1e-9);assert a['initial_command_within_original_1e9_check'];a['scope']+=' Exact metadata corrected: anchored initial commands differ from recorded commands by a floating-point residue; the original1e-9 numerical check passed but bitwise equality did not. Arrays are unchanged; no execution or physical-prefix claim.'
 for a,b in zip(data['rows'],before['rows']):assert a['nodes']==b['nodes']
 data['metadata_correction']=dict(original=str(source),original_sha256=sha(source),reason='Independent audit caught initial_command_exact=True despite5.55e-17rad residual. Preserve false exact gate, report unchanged original1e-9 numerical check separately. No tolerance/command/model/physics change.')
 (O/'results.json').write_text(json.dumps(data,indent=2));record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=data['metadata_correction'],all_command_force_and_kinematic_arrays_unchanged=True,physics_ticks=0,optimizer_updates=0,original_audit_assertion_preserved=True);(O/'recovery.json').write_text(json.dumps(record,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));(A/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in A.iterdir() if p.is_file()},indent=2));print(json.dumps(record),flush=True)
if __name__=='__main__':main()

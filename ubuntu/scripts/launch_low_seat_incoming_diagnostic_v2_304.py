"""ONE13s recording-only prefix of234, not a controller candidate or repair."""
from pathlib import Path
import datetime,hashlib,json,re,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return h.hexdigest()
def main():
 assert not (D/'incoming_diagnostic_v2_303').exists() and not (D/'incoming_revision_v2_304.json').exists()
 proc=subprocess.run(['ps','-eo','pid,args'],capture_output=True,text=True,check=True).stdout
 assert not [p for p in proc.splitlines() if re.search(r'^\s*\d+\s+\S*python\S* .*scripts/(run|refine|optimize|audit|diagnose|schedule|check|screen)_low_seat_\S+\.py',p)]
 subprocess.run([str(R/'IsaacLab/.venv/bin/python'),'-m','py_compile']+[str(R/'scripts'/n) for n in ['low_seat_incoming_recorder_v2_301.py','check_low_seat_incoming_recorder_v2_302.py','run_low_seat_incoming_diagnostic_v2_303.py','audit_low_seat_incoming_v2_305.py']],check=True)
 prep=read(D/'incoming_recorder_preparation_v2_302/results.json');assert prep['passed']
 assert all(sha(p)==h for p,h in prep['source_sha256'].items())
 sources=dict(read(D/'centered_probe_234/revision.json')['sources'])
 for n in ['low_seat_private_recorder_198.py','low_seat_drive_recorder_244.py','check_low_seat_drive_recorder_245.py','low_seat_contact_recorder_257.py','check_low_seat_contact_recorder_258.py','low_seat_incoming_recorder_v2_301.py','check_low_seat_incoming_recorder_v2_302.py','run_low_seat_incoming_diagnostic_v2_303.py','audit_low_seat_incoming_v2_305.py',Path(__file__).name]:sources[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for n in ['IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py','IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py','IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py','IsaacLab/source/isaaclab/isaaclab/actuators/actuator_kernels.py','IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py','IsaacLab/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py','IsaacLab/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.3.2+110.3.0.lx64.r.cp312.u7f4/omni/physics/tensors/api.py']:sources[str(R/n)]=sha(R/n)
 assert all(sha(p)==h for p,h in sources.items())
 checks=[]
 for n in ['prior_hashes.json','completed_probes_preservation_300.json','supplemental_preservation_300.json']:
  m=read(D/n);bad=[p for p,h in m.items() if not Path(p).exists() or sha(p)!=h];checks.append(dict(manifest=n,checked=len(m),mismatches=bad));assert not bad,bad[:3]
 (D/'prior_recheck_before_incoming_v2_304.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
 scope='Adds incoming-joint-wrench every5ms and independent stiffness/damping/armature/effort-limit/friction backend readbacks at preflight/8s/13s using private destination descriptors; actual joint/body order checked, all allocated timestamped and direct public buffers byte-checked unchanged. Transmitted total joint wrench is NOT isolated motor effort. Distinct contact-wrench recording-only13s prefix of centered234: intercept EXISTING normal/friction contact query returns with separate force/application points; no added query, average-point wrench approximation or control use. Also retains private current link transforms/velocities and backend drive targets, exact implicit kernel inputs/outputs and live authored solver settings. No public-state getter or replacement of runtime inputs. All234 controls, guards, physics/materials and parent unchanged.15600active+600settling,zero complete20s episodes/optimizer/holdout. Full24-case prefix must replay EXACTLY; no admission or equality waiver.'
 revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=scope,sources=sources,physics_gpu=1,active_ticks=15600,initial_settling=600,full20s_episodes=0,optimizer_updates=0,fresh_holdout=False,physical_admission=False)
 (D/'incoming_revision_v2_304.json').write_text(json.dumps(revision,indent=2))
 shutil.copy2(D/'status.json',D/'status_before_incoming_v2_304.json');m=read(D/'status.json');m.update(updated_utc=revision['utc'],phase='incoming_diagnostic_v2_303_running',physical_jobs_running=True,latest_running_candidate='incoming_diagnostic_v2_303',repair_complete=False,main_training_started=False,fresh_holdout_started=False);(D/'status.json').write_text(json.dumps(m,indent=2))
 with (R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md').open('a') as f:f.write('\n\nDiagnostic304: '+scope+'\n')
 r=subprocess.run(['bash','scripts/run_resting_gpu1.sh','../scripts/run_low_seat_incoming_diagnostic_v2_303.py'],cwd=R)
 (D/'incoming_process_exit_v2_304.json').write_text(json.dumps(dict(returncode=r.returncode,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_pending=True),indent=2))
 assert r.returncode==0
if __name__=='__main__':main()

"""Wait for both independent verifiers, then export a self-contained evidence bundle."""
import hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/split_support_curriculum_02');STUDY=DATA/'ppo_01';OUT=DATA/'final_export_01'
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def main():
 while True:
  states={s:read(STUDY/s/'status.json') for s in ['sitting','leaning']}
  if any(x['phase']=='execution_failed' for x in states.values()):
   write(DATA/'status.json',dict(phase='needs_execution_repair',surface_states=states));return 1
  if all(x['phase']=='verified_complete' for x in states.values()):break
  time.sleep(30)
 subprocess.run([sys.executable,str(ROOT/'scripts/render_split_report.py'),'--data',str(STUDY),'--output',str(OUT)],check=True)
 subprocess.run([sys.executable,str(ROOT/'scripts/render_split_replay.py'),'--data',str(STUDY),'--output',str(OUT/'motion_replay.html')],check=True)
 for surface in ['sitting','leaning']:
  folder=STUDY/surface;copy(folder/'protocol.json',OUT/surface/'protocol.json');shutil.copytree(folder/'frozen',OUT/surface/'frozen')
  p=read(folder/'protocol.json')
  for seed in p['seeds']:
   state=read(folder/f'seed_{seed}.json');copy(folder/f'seed_{seed}.json',OUT/surface/f'seed_{seed}.json')
   for h in state['history']:
    d=folder/h['train']
    for name in ['verification.json','progress.json','manifest.json','results.json','first_failures.json']:copy(d/name,OUT/surface/'training'/d.name/name)
   copy(Path(state['last_checkpoint']),OUT/surface/'checkpoints'/f'{seed}_final.pt')
  for label in ['fixed','old_reference','reference']+[f'policy_{seed}' for seed in p['seeds']]:
   dest='PPO_'+label.removeprefix('policy_') if label.startswith('policy_') else label
   for name in ['trace.npz','startup.npz','geometry.json','static_inertial_model.json']:
    copy(folder/label/name,OUT/surface/dest/name)
  copy(folder/'verified_01/verification_source.py',OUT/surface/'verification_source.py')
 for n in ['prior_preservation_before.json','old_reference_instrumentation_parity.json','leaning_repair_manifest_02.json']:copy(DATA/n,OUT/'validation'/n)
 for surface,suffix in [('sitting','01'),('leaning','02')]:
  copy(DATA/f'{surface}_preflight_verified_{suffix}/verification.json',OUT/'validation'/f'{surface}_preflight.json')
 for n in ['SPLIT_SUPPORT_PROTOCOL.md','SPLIT_LEANING_REPAIR.md']:copy(ROOT/'docs'/n,OUT/'docs'/n)
 # Link the inspected offline replay without altering frozen scientific artifacts.
 p=OUT/'report.html';p.write_text(p.read_text().replace('<h1>','<p><a href="motion_replay.html">Open matched motion replay</a></p><h1>',1))
 manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()};write(OUT/'SHA256.json',manifest)
 write(DATA/'status.json',dict(phase='verified_results_ready_for_visual_review',output=str(OUT),surfaces=states,files=len(manifest),note='Agent must visually inspect, copy to Mac, update handoffs and deliver comparison; no policy promoted.'))
 print(OUT,flush=True);return 0
if __name__=='__main__':sys.exit(main())

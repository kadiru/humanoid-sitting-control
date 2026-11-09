"""Bounded validation of each surface on its own GPU; no study training here."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/split_support_curriculum_02')
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def run(surface,gpu,label,mode,extra=None,profile=None):
 out=DATA/f'{surface}_{label}';cmd=['bash','scripts/'+('run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'),'../scripts/run_split_leaning.py','--bank','/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01','--output',str(out),'--mode',mode,'--surface',surface,'--profiles',str(DATA/'split_profiles'/(profile or f'{surface}_selected_02.json')),'--num-envs','14','--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']+(extra or [])
 write(DATA/f'{surface}_preflight_status_02.json',dict(phase='running',active=out.name));write(DATA/(out.name+'.command.json'),cmd)
 with (DATA/(out.name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 write(DATA/(out.name+'.exit.json'),dict(returncode=p.returncode));assert p.returncode==0,out.name
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--surface',required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
 try:
  out=run(a.surface,a.gpu,'reference_validation_02','reference',['--diagnostics']);r=json.loads((out/'results.json').read_text());assert r['count']==r['successes']==14, 'Corrected reference did not pass all14 starts'
  run(a.surface,a.gpu,'reset_isolation_02','reference',['--reset-probe'])
  smoke=run(a.surface,a.gpu,'optimizer_smoke_02','train',['--start-stage','0','--num-envs','16','--iterations','2'])
  out=run(a.surface,a.gpu,'smoke_evaluation_02','evaluate',['--checkpoint',str(smoke/'final.pt')]);r=json.loads((out/'results.json').read_text());assert r['count']==r['successes']==14, 'Small optimizer smoke did not retain reference successes'
  write(DATA/f'{a.surface}_preflight_status_02.json',dict(phase='complete_pending_verification'))
 except Exception as e:
  write(DATA/f'{a.surface}_preflight_status_02.json',dict(phase='failed',error=repr(e)));raise
if __name__=='__main__':main()

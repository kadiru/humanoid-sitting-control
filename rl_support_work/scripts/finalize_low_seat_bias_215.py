"""Independent audit and complete-attempt video; never reruns physics."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json,os,subprocess,traceback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');S=D/'bias_finalization_status_215.json'
def main():
    assert not S.exists()
    assert json.loads((D/'status_bias_probe_210.json').read_text())['phase']=='physics_complete_audit_pending'
    state=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase='running',physics_repair=False,optimizer_updates=0)
    S.write_text(json.dumps(state,indent=2))
    names=['audit_low_seat_bias_211.py','audit_low_seat_bias_dual_212.py','render_low_seat_all_attempts_213.py','verify_low_seat_all_attempts_214.py',Path(__file__).name]
    sources={str(R/'scripts'/n):hashlib.sha256((R/'scripts'/n).read_bytes()).hexdigest() for n in names}
    (D/'bias_finalization_sources_215.json').write_text(json.dumps(sources,indent=2))
    def work(label,steps):
        results=[]
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='1')
        for name,extra in steps:
            with (D/(name+'.log')).open('x') as f:
                p=subprocess.run([str(R/'IsaacLab/.venv/bin/python'),str(R/'scripts'/name),*extra],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
            results.append(dict(script=name,exit_code=p.returncode))
            (D/f'bias_{label}_status_215.json').write_text(json.dumps(dict(phase='failed' if p.returncode else 'running',steps=results),indent=2))
            if p.returncode:break
        passed=len(results)==len(steps) and all(r['exit_code']==0 for r in results)
        out=dict(phase='complete' if passed else 'failed',steps=results,passed=passed)
        (D/f'bias_{label}_status_215.json').write_text(json.dumps(out,indent=2));return out
    with ThreadPoolExecutor(2) as ex:
        a=ex.submit(work,'audit',[('audit_low_seat_bias_211.py',[]),('audit_low_seat_bias_dual_212.py',['--folder',str(D/'bias_probe_210')])])
        v=ex.submit(work,'video',[('render_low_seat_all_attempts_213.py',[]),('verify_low_seat_all_attempts_214.py',[])])
        state.update(audit=a.result(),video=v.result())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    state.update(phase='complete' if state['audit']['passed'] and state['video']['passed'] else 'needs_recovery',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_unchanged=True,physical_repair=False)
    S.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
if __name__=='__main__':main()

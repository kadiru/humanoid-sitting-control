"""Finite sequential owner; no relaunch, no promotion, only the4090."""
import os,sys,fcntl,time,signal,subprocess,traceback,shutil
from pathlib import Path
from common import ROOT,DATA,read,write,sha
from audit import evaluate
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/armchair_rest_ppo_20261105_v1';GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
def main():
    assert not (OUT/'training_receipt.json').exists(),'NO_AUTOMATIC_RETRY'
    locks=[]
    for p in [DATA/'contact_parallel_gpu_v1/transition.lock',DATA/'contact_ppo_scientific_owner.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    frozen=read(HERE/'freeze.json');start=time.monotonic();active=None;child=None;completed=[]
    def verify():
        for p,h in frozen.items():assert sha(p)==h,('FROZEN_SOURCE_CHANGED',p)
    def status(state,**kw):write(OUT/'training_status.json',dict(state=state,owner_pid=os.getpid(),active=active,completed=completed,elapsed_s=time.monotonic()-start,**kw))
    def resource():
        assert time.monotonic()-start<43200,'OWNER_TIME'
        assert shutil.disk_usage(OUT).free>20_000_000_000,'DISK_FREE'
        mem={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines())}
        assert mem['MemAvailable']>8_000_000_000,'RAM_FREE'
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<20_000_000_000,'OUTPUT_SIZE'
        temp=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits']).strip());assert temp<85,('GPU_TEMP',temp)
    def run(label,bank,checkpoint,train=False,mesh=False):
        nonlocal active,child
        verify();resource()
        assert not subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader']).strip(),'4090_BUSY'
        cmd=['bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(HERE/'run_training.py'),'--bank',str(bank),'--output',str(OUT/label),'--mode','train' if train else 'evaluate','--checkpoint',str(checkpoint),'--seed','65011','--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
        if mesh:cmd+=['--mesh']
        if train:cmd+=['--updates','128','--demonstrations',str(OUT/'prepared/demonstrations.npz')]
        write(OUT/(label+'.command.json'),cmd);t=time.monotonic()
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
            active=dict(label=label,pid=child.pid);status('running')
            while child.poll() is None:
                time.sleep(10);resource()
                assert time.monotonic()-t<(30000 if train else 3600),('CHILD_TIMEOUT',label)
            write(OUT/(label+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t))
        active=None
        assert child.returncode==0,('CHILD_FAILED',label,child.returncode)
        if not train:evaluate(OUT/label)
        completed.append(label);status('running')
    verify();assert read(OUT/'trainer_preflight.json')['passed'];assert read(HERE/'source_review.json')['passed']
    write(OUT/'training_receipt.json',dict(owner_pid=os.getpid(),freeze_sha256=sha(HERE/'freeze.json'),protocol_sha256=sha(HERE/'protocol.json'),seed=65011,gpu=GPU,scientific_owners=1))
    try:
        prepared=OUT/'prepared'
        run('prior_armchair11',prepared/'evaluation',prepared/'initial.pt',mesh=True)
        audit=read(OUT/'prior_armchair11/independent_audit.json');assert any(audit.get('validated_clean',audit['clean'])),'NO_SUCCESSFUL_PRIOR_ROLLOUT'
        run('training_65011',prepared/'train',prepared/'initial.pt',train=True,mesh=True)
        final=OUT/'training_65011/final.pt'
        run('learned_armchair11',prepared/'evaluation',final,mesh=True)
        run('learned_sitting72',ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting',final)
        prior=read(OUT/'prior_armchair11/independent_audit.json');rest=read(OUT/'learned_armchair11/independent_audit.json');sit=read(OUT/'learned_sitting72/independent_audit.json')
        protected=read(ROOT/'experiments/motion_prior_ppo_20261029_v1/protected_sitting_cases.json')
        write(OUT/'comparison.json',dict(prior_armchair=prior,learned_armchair=rest,learned_sitting=sit,protected_definition=protected,physical_thresholds_unchanged=True,automatically_promoted=False))
        status('complete')
    except BaseException as e:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status('stopped',error=repr(e));write(OUT/'training_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()

"""Finish diagnosis/video/report after existing198 and199; no physics launch."""
from pathlib import Path
import datetime,json,os,subprocess,time
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');PY=R/'IsaacLab/.venv/bin/python';S=D/'private_finalization_status_v2_204.json'
def status(phase,**kw):S.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,physics_launch=False,optimizer_updates=0,**kw),indent=2))
def start(script,log):
    env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    return subprocess.Popen([str(PY),str(R/'scripts'/script)],cwd=R,env=env,stdout=(D/log).open('w'),stderr=subprocess.STDOUT)
def main():
    assert not S.exists();status('waiting_for_existing198_199')
    for _ in range(180):
        if (D/'private_audit_v2_199/independent_audit.json').exists():break
        time.sleep(10)
    else:raise RuntimeError('Existing198/199 did not produce audit; inspect rather than restart')
    a=json.loads((D/'private_audit_v2_199/independent_audit.json').read_text());assert a['all_substep_raw_exact']
    assert a['postflight_completion_checked_from_saved_arrays']
    status('diagnosis_and_video',exact_prefix_replay=a['exact_prefix_replay'])
    jobs=[start('diagnose_low_seat_private_v2_200.py','private_diagnosis_200.log'),start('render_low_seat_all_attempts_201.py','all_attempts_video_201.log')]
    codes=[p.wait() for p in jobs];assert codes==[0,0],codes
    status('independent_video_audit',exact_prefix_replay=a['exact_prefix_replay'])
    assert start('verify_low_seat_all_attempts_202.py','all_attempts_video_audit_202.log').wait()==0
    assert start('report_low_seat_private_v2_203.py','private_report_203.log').wait()==0
    status('complete',exact_prefix_replay=a['exact_prefix_replay'],physical_admission=False,repair_complete=False)
if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e));raise

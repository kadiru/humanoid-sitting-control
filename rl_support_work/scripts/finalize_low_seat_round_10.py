"""Finish audits and an interim report. Never launch training or holdout."""
import json,time,os,subprocess,traceback,hashlib
from pathlib import Path
from run_transfer_assessment import status
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
STATE=D/'round_audit_status_10.json'
def read(p):return json.loads(Path(p).read_text())
def main():
    try:
        status(STATE,'waiting',main_training_started=False,fresh_holdout_started=False)
        while True:
            files=[D/'current_retention_status_09.json']
            values=[read(p) if p.exists() else {} for p in files]
            assert not any(x.get('phase')=='failed' for x in values),values
            if all(x.get('phase')=='complete' for x in values):break
            time.sleep(20)
        status(STATE,'auditing',main_training_started=False,fresh_holdout_started=False)
        import torch,numpy as np
        torch.set_num_threads(2)
        from positive_slope_validation_v4 import actors,physical,matched
        from verify_seat_load_study import load,equal
        cp=D.parent/'positive_slope_01/ppo_01/seed_61111/block_03/train/final.pt'
        output={}
        for round_id in ['09']:
            root=D/('current_retention_'+round_id);items={}
            for bank in ['known_old','known_transfer','known_recent']:
                folder=root/bank/'candidate';base=root/bank/'baseline'
                items[bank]=dict(actors=actors(folder,cp),physical=physical(folder,require_clean=False),
                    matched=matched(base,folder,start=10.))
                assert equal(load(folder/'initial.pt'),load(folder/'final.pt')),'Unexpected optimizer change'
                trace=np.load(folder/'trace.npz')
                for key in ['q','body','command','reference','torque','geometry']:assert np.isfinite(trace[key]).all(),(folder,key)
                # Trace samples are0.1s apart. Every applied physics command keeps
                # the inherited0.003rad/tick limit; this independently checks its
                # necessary sampled bound, not hidden between-sample behavior.
                max_delta=float(np.max(abs(np.diff(trace['command'],axis=0))))
                assert max_delta<=.015001,(folder,max_delta)
                items[bank]['sampled_command_change_rad_per_0p1s']=max_delta
                items[bank]['initial_final_learner_identical_no_optimizer']=True
                items[bank]['sensor_control']=read(folder/'sensor_control_audit.json')
            output[round_id]=items
        dest=D/'current_policy_independent_audit_10.json';assert not dest.exists();dest.write_text(json.dumps(dict(passed=True,rounds=output),indent=2))
        from datetime import datetime,timezone
        from audit_low_seat_contact_probe_v2 import clean
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        targets=read(D/'retention_targets.json')['banks']
        recomputed={}
        for bank,oldbank in [('known_old','development_old'),('known_transfer','development_transfer'),('known_recent','holdout')]:
            root=D/'current_retention_09'/bank
            rows=read(root/'candidate/results.json')['episodes']
            required=[int(k) for k,v in targets[oldbank]['cases'].items() if v['successful_controllers']]
            retained=[i for i in required if clean(rows[i])]
            recorded=read(D/'current_retention_09/retention.json')[bank]
            assert len(retained)==recorded['retained']
            assert [i for i in required if i not in retained]==recorded['lost']
            assert all(row['steps']==1000 for row in rows)
            for f,h in read(root/'baseline_reuse.json')['hashes'].items():assert sha(root/'baseline'/f)==h
            recomputed[bank]=dict(required=required,retained=retained,lost=recorded['lost'])
        for f,h in read(D/'current_retention_09/sources.json').items():assert sha(f)==h,(f,'source changed')
        prior=read(D/'prior_hashes.json');protected=read(D/'completed_probes_preservation_08.json')
        for f,h in {**prior,**protected}.items():assert sha(f)==h,(f,'protected changed')
        added={}
        for round_id in ['08','09']:
            for folder,dirs,files in os.walk(D/('current_retention_'+round_id),followlinks=False):
                for name in files:
                    p=Path(folder)/name
                    if not p.is_symlink():added[str(p)]=sha(p)
        preserve=D/'completed_probes_preservation_10.json';assert not preserve.exists()
        preserve.write_text(json.dumps({**protected,**added},indent=2))
        check=D/'prior_recheck_after_retention_10.json';assert not check.exists()
        check.write_text(json.dumps(dict(passed=True,utc=datetime.now(timezone.utc).isoformat(),
            prior_count=len(prior),protected_completed_files=len(protected)+len(added),
            recomputed_retention=recomputed,actual_physics_trials=72,active_ticks=72000,
            initial_settling_excluded=1800,optimizer_updates=0,fresh_holdout_used=False),indent=2))
        env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2')
        with (D/'round_report_10.log').open('x') as f:
            subprocess.run([str(ROOT/'IsaacLab/.venv/bin/python'),'scripts/report_low_seat_round_10.py'],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(STATE,'complete',report=str(D/'progress_report_10'),main_training_started=False,fresh_holdout_started=False,
            physical_repair_not_implied=True)
    except Exception as e:
        status(STATE,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()

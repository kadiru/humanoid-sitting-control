"""One finite local artifact transfer; no scheduler, GPU use, or scientific work."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
REMOTE='/home/kadir/humanoid-sitting/outputs/furniture_validation_20261028_v2'

def save(value):
    p=HERE/'delivery_status.json';q=p.with_suffix('.tmp')
    q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)

def main():
    start=time.monotonic();failures=0
    while time.monotonic()-start<52*3600:
        try:
            p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','kfuUbuntu','cat '+REMOTE+'/status.json'],capture_output=True,text=True,timeout=30,check=True)
            state=json.loads(p.stdout);failures=0
            save(dict(state=state['state'],owner=state,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),recurring_automation=False))
            if state['state'] in ['complete','stopped']:
                local=HERE/'results';local.mkdir(exist_ok=True)
                # Preserve large trajectory arrays remotely; copy the complete
                # scores, contact records, logs, review frames and video locally.
                subprocess.run(['rsync','-a','--exclude=trajectory.npz','kfuUbuntu:'+REMOTE+'/',str(local)+'/'],timeout=1800,check=True)
                if state['state']=='complete':
                    manifest=json.loads((local/'video_manifest.json').read_text());video=local/'furniture_collisions_and_forces.mp4'
                    assert hashlib.sha256(video.read_bytes()).hexdigest()==manifest['sha256']
                    save(dict(state='video_delivered',video=str(video),sha256=manifest['sha256'],duration_s=53,
                        collision_validated=manifest['collision_validated'],robot_trials=manifest['robot_trials'],robot_clean=manifest['robot_clean'],recurring_automation=False))
                else:save(dict(state='validation_stopped',failure=json.loads((local/'failure.json').read_text()),local_evidence=str(local),recurring_automation=False))
                return
        except Exception as e:
            failures+=1;save(dict(state='transfer_check_failed',error=repr(e),consecutive_failures=failures,recurring_automation=False))
            if failures>=10:raise
        time.sleep(60)
    raise TimeoutError('Finite52-hour delivery deadline; no indefinite polling')

if __name__=='__main__':main()

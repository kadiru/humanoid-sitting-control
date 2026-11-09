"""Finite local delivery collector; no simulation/training/recurring automation."""
import datetime,hashlib,html,json,os,shlex,subprocess,time,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;RESULTS=HERE/'results';REMOTE='/home/kadir/humanoid-sitting/outputs/furniture_perception_repair_20261029_v1'
DEADLINE=18*3600+300

def read(path):return json.loads(path.read_text())
def write(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
def fetch(remote,local):
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10','kfuUbuntu:'+remote,str(local)],check=True,timeout=180)
def page(status,terminal=False,error=None):
    state=status.get('state','waiting_for_ppo');stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    complete=state=='complete' and (RESULTS/'media/updated_furniture_depth_contacts.mp4').exists()
    titles={'waiting_for_ppo':'Waiting for PPO to finish','retention72':'Checking all72 sitting cases','furniture_rollouts':'Testing seats,arms andbacks','body_support':'Testing body-shape support','rendering':'Rendering the updated video','complete':'Fresh furniture retest complete','stopped':'Retest stopped: evidence preserved'}
    text=html.escape(titles.get(state,state));summary=''
    if (RESULTS/'summary.json').exists():
        s=read(RESULTS/'summary.json');r=s['retention72'];rows=s['furniture']
        summary='<p>Corrected-perception72-case result: <b>'+str(r['clean'])+'/72 clean</b>; '+str(len(r['repairs']))+' original failures repaired; '+str(len(r['regressions']))+' protected-case regressions.</p>'
        summary+='<p>Furniture: '+str(sum(x['robot']['clean'] is True for x in rows))+'/14 validated clean attempts. '+str(sum(x['robot']['clean'] is None for x in rows))+' contact scores remain unvalidated.</p>'
    media='<video controls preload="metadata" src="results/media/updated_furniture_depth_contacts.mp4" poster="results/media/robot_099.png"></video>' if complete else '<div class="pending">The fresh video is not available yet. This page will update after the finite retest finishes.</div>'
    detail=html.escape(json.dumps(status,indent=2))
    failure='<p class="error">'+html.escape(error or '')+'</p>' if error else ''
    refresh='' if terminal else '<meta http-equiv="refresh" content="45">'
    content='<!doctype html><html><head><meta charset="utf-8">'+refresh+'<title>Furniture perception and contact retest</title><style>body{font:18px/1.5 system-ui;background:#eef2f5;color:#20394b;max-width:1200px;margin:32px auto;padding:0 24px}h1{font-size:30px}video,img{width:100%;border-radius:12px}p{max-width:1000px}.pending{padding:28px;background:white;border-radius:12px}.error{color:#a12222}small{color:#597080}pre{font-size:13px;white-space:pre-wrap}summary{cursor:pointer}</style></head><body><h1>'+text+'</h1><p>4090 only. The4070 stays with AIWorker. Original action limits and physical success criteria are unchanged.</p>'+summary+failure+media+'<p>Depth alignment andsurface recognition are separate from successful sitting. Body-shape loadtests use controlledfixtures; they do not prove whole-robot balance.</p><details><summary>Saved-frame diagnosis, before the fresh retest</summary><p>Green shows the selected depth patch. All42 saved captures aligned within0.3mm after correcting camera pose andpixelcenters. Nineof14 layouts pass the unchanged single-plane gate; curved surfaces remain explicitly flagged.</p><img src="results/saved_recognition_preview.jpg"></details><details><summary>Execution status</summary><pre>'+detail+'</pre></details><p><small>Updated '+stamp+'. One finite continuation; no recurring scheduledtask. Video will be structurally checked; final visual review is still required.</small></p></body></html>'
    temp=HERE/'review.tmp.html';temp.write_text(content);temp.replace(HERE/'review.html')

def main():
    start=time.monotonic();write(HERE/'delivery_receipt.json',dict(pid=os.getpid(),finite=True,deadline_seconds=DEADLINE,simulation_launches=0))
    errors=0
    while time.monotonic()-start<DEADLINE:
        try:
            text=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','kfuUbuntu','cat '+shlex.quote(REMOTE+'/status.json')],text=True,timeout=30)
            status=json.loads(text);write(RESULTS/'status.json',status);errors=0
            if status['state']=='stopped':
                fetch(REMOTE+'/failure.json',RESULTS/'failure.json');page(status,True,read(RESULTS/'failure.json')['error']);return
            if status['state']=='complete':
                for name in ['summary.json','resource_accounting.json','policy_bindings.json','policy_selection.json','probe_plan_summary.json']:
                    fetch(REMOTE+'/'+name,RESULTS/name)
                (RESULTS/'media').mkdir(exist_ok=True)
                fetch(REMOTE+'/media/video_manifest.json',RESULTS/'media/video_manifest.json')
                for name in ['updated_furniture_depth_contacts.mp4','depth_recognition.jpg','robot_000.png','robot_049.png','robot_099.png','robot_199.png']:
                    fetch(REMOTE+'/media/'+name,RESULTS/'media'/name)
                p=RESULTS/'media/updated_furniture_depth_contacts.mp4';h=hashlib.sha256(p.read_bytes()).hexdigest()
                assert h==read(RESULTS/'media/video_manifest.json')['sha256']
                write(HERE/'delivery_complete.json',dict(complete=True,sha256=h,video=str(p),visual_review_pending=True))
                page(status,True);return
            page(status)
        except Exception as e:
            errors+=1;write(HERE/'delivery_error.json',dict(error=repr(e),consecutive_errors=errors,traceback=traceback.format_exc()))
            if errors>=3:
                page({'state':'delivery_error'},True,'Local delivery failed; remote work was not interrupted. '+repr(e));return
        time.sleep(45)
    page({'state':'delivery_deadline'},True,'Finite delivery deadline reached. Remote evidence remains preserved.')
if __name__=='__main__':main()

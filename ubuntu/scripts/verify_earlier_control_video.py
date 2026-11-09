"""Independent episode coverage and outcome-label audit for the short video."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def verify(folder,expected_paths):
    m=read(folder/'coverage.json');assert m['source_paths']==expected_paths and not m['smoke_only']
    expected={}
    for bank,items in expected_paths.items():
        for key,path in items.items():
            for row in read(Path(path)/'results.json')['episodes']:
                success=row['success'] and not any(row['failures'].values()) and row['slip_ticks']==0 and row['illegal_ticks']==0
                expected[(bank,key,row['case'])]=('SUCCESS' if success else 'FAILURE',row)
    actual={}
    for entry in m['episodes']:
        key=(entry['bank'],entry['controller'],entry['case']);assert key not in actual
        assert entry['label']==expected[key][0];actual[key]=entry
        for k in ['slip_ticks','illegal_ticks']:assert entry[k]==expected[key][1][k]
        assert entry['margin_ticks']==expected[key][1]['traction_bad_ticks']
    assert set(actual)==set(expected)
    pages={(p['bank'],p['case']):p for p in m['pages']};assert len(pages)==len(m['pages'])
    assert set(pages)=={(b,c) for b,k,c in expected}
    for (bank,key,case),entry in actual.items():
        page=pages[(bank,case)];trace=np.load(Path(entry['source'])/'trace.npz');tick=np.load(Path(entry['source'])/'every_tick.npz')
        required={0,199}
        for bad in [~tick['no_slip'][:,case],tick['wrong'][:,case]>10,tick['failure'][:,case]]:
            idx=np.flatnonzero(bad)
            if len(idx):required.add(int(np.argmin(abs(trace['t']-tick['t'][idx[0]]))))
        assert required.issubset(set(page['trace_indices'])) and len(page['trace_indices'])==page['frames']
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,nb_frames','-of','json',str(folder/'summary.mp4')]))
    assert 0<float(probe['format']['duration'])<60 and abs(float(probe['format']['duration'])-m['duration_s'])<.04
    assert probe['streams'][0]['codec_name']=='h264'
    assert int(probe['streams'][0]['nb_frames'])==round(m['fps']*m['duration_s'])
    return dict(passed=True,episodes=len(actual),successes=sum(e['label']=='SUCCESS' for e in actual.values()),failures=sum(e['label']=='FAILURE' for e in actual.values()),conditions=len(pages),seconds=float(probe['format']['duration']),codec='h264',all_first_event_frames_included=True,sha256=hashlib.sha256((folder/'summary.mp4').read_bytes()).hexdigest())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);a=p.parse_args();r=verify(a.folder,read(a.verification)['paths']);(a.folder/'verification.json').write_text(json.dumps(r,indent=2));print(r)

"""Resume authorized work after each independent assessment, with hard verification gates."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_transfer_assessment import DATA,ROOT,status,run

def main():
    p=argparse.ArgumentParser();p.add_argument('--surface',choices=['sitting','leaning'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    folder=DATA/'pipeline'/a.surface;folder.mkdir(parents=True,exist_ok=False)
    try:
        status(folder/'status.json','waiting_for_assessment')
        while True:
            file=DATA/'assessment'/a.surface/'status.json'
            state=json.loads(file.read_text())
            if state['phase']=='assessment_complete':break
            assert state['phase']!='execution_failed',state
            time.sleep(10)
        status(folder/'status.json','verifying_assessment')
        run(folder,'assessment_verification',[sys.executable,str(ROOT/'scripts/verify_transfer_assessment.py'),'--data',str(DATA/'assessment'/a.surface),'--output',str(DATA/'assessment'/a.surface/'verified_01')])
        status(folder/'status.json','optimizer_preflight')
        run(folder,'preflight',[sys.executable,str(ROOT/'scripts/run_transfer_effort_study.py'),'--surface',a.surface,'--gpu',str(a.gpu),'--preflight'])
        status(folder/'status.json','training_and_matched_evaluation')
        run(folder,'study',[sys.executable,str(ROOT/'scripts/run_transfer_effort_study.py'),'--surface',a.surface,'--gpu',str(a.gpu)])
        status(folder/'status.json','verified_complete')
    except Exception as e:
        status(folder/'status.json','execution_or_validation_failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()

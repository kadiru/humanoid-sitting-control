"""Persistent baseline evaluations only; no automatic training or tuning."""
import argparse,traceback
from positive_slope_common import *
from run_transfer_assessment import run,status
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);a=p.parse_args()
folder=DATA/'assessment';folder.mkdir(exist_ok=True);state=DATA/f'assessment_gpu{a.gpu}.json'
try:
    jobs=['reference','60117','60111'] if a.gpu==0 else ['59117','59111']
    for kind in jobs:
        status(state,'evaluating',controller=kind)
        run(folder,kind,baseline(folder,a.gpu,kind,kind))
    check(read(DATA/'prior_hashes.json'));status(state,'complete',controllers=jobs)
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise

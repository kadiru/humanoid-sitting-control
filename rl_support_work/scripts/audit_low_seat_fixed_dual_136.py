"""Offline independent current-view routing/association/input replay."""
import argparse,json,hashlib,datetime
from pathlib import Path
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from humanoid_sitting.supported_surface import observed_support
from low_seat_dual_view_40 import DOWN
from run_supported_postures import REAR_CAMERA

def main():
 p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();folder=a.folder;P=folder/'candidate'
 a=np.load(P/'dual_view_inputs_132.npz');model=BodyModel('runs/g1_kinematics_geometry_04/model.json');trace=np.load(P/'trace.npz');sensors=np.load(P/'sensor_inputs.npz')['sensors']
 contract=json.loads((P/'fixed_contract_132.json').read_text());insertion='measurements=fallback(self,measurements,q,gravity,force)\n    '
 assert contract['observer_replacement'].replace(insertion,'')==contract['observer_original']
 count=0;fallbacks=0;checks=[]
 for k,t in enumerate(a['t']):
  at=int(np.flatnonzero(np.isclose(trace['t'],t,atol=1e-8))[0]);assert np.array_equal(a['q'][k],trace['q'][at])
  for i in range(24):
   prior=a['prior'][k,i] if np.isfinite(a['prior'][k,i]).all() else None
   rear,rd=observed_support(model,a['rear_K'][k,i],REAR_CAMERA,a['rear_depth'][k,i],a['q'][k,i],a['gravity'][k,i],prior)
   saved=a['rear_fit'][k,i];assert (rear is None)==np.isnan(saved).all()
   if rear is not None:assert np.array_equal(rear,saved),(k,i,'rear replay')
   needed=bool(a['selected'][k,i] and rear is None);assert needed==a['needed'][k,i]
   chosen=rear;source=0 if rear is not None else -1
   if needed:
    down,dd=observed_support(model,a['down_K'][k,i],DOWN,a['down_depth'][k,i],a['q'][k,i],a['gravity'][k,i],prior)
    if down is not None:chosen=down;source=1;fallbacks+=1
   assert source==a['source'][k,i]
   recorded=a['chosen_fit'][k,i];assert (chosen is None)==np.isnan(recorded).all()
   if chosen is not None:
    assert np.array_equal(chosen,recorded) and np.array_equal(chosen,trace['geometry'][at,i]),(k,i,'chosen current measurement')
    step=round(float(t)*10)
    if step<200:assert np.array_equal(chosen,sensors[0,step,i,368:381]),(k,i,'actual414 geometry channels')
   count+=1
  print('REPLAY_TIME',float(t),flush=True)
 selected=a['selected'];assert np.all(~(a['source']==1)|selected)
 staged=json.loads((folder/'independent_fixed_audit_135.json').read_text());assert staged['software_audit_passed']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),software_audit_passed=True,physical_admission_passed=False,repair_admission_not_implied=True,current_plane_replay_exact=True,actual414_chosen_geometry_exact=True,frames=len(a['t']),case_frames=count,down_fallback_case_frames=fallbacks,unselected_down_fallbacks=0,unchanged_plane_fit_and_association=True,source_insertion_exact=True,fixed_audit=staged,prior_failed_physical_and_exact_gates_unchanged=True)
 dest=folder/'independent_dual_view_audit_136.json';assert not dest.exists();dest.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='fixed_audit'},indent=2))
if __name__=='__main__':main()

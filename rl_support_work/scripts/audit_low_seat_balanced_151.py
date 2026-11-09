"""Independent inputs, path, commands, sampled guards and physical outcomes."""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
from validate_low_positive_feedback_v3 import validate
from audit_low_seat_contact_probe_v2 import summarize,clean
from low_seat_balanced_runtime_v2_148 import build_plan
from low_seat_gated_command_128 import GatedCommandPath
from low_seat_measured_guard_130 import MeasuredGuard
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');F=D/'balanced_force_probe_150';P=F/'candidate'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    outpath=F/'independent_balanced_audit_151.json';assert not outpath.exists();torch.set_num_threads(2)
    for p,h in read(F/'revision.json')['sources'].items():assert sha(p)==h,p
    validation=validate(P);assert validation['actor_steps']==validation['critic_steps']==0
    contract=read(P/'fixed_contract_132.json');cfg=contract['config'];m=BodyModel(cfg['cooked_model']);meta=read(cfg['cooked_model']);ine=read(cfg['inertia']);cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json')
    tr=np.load(P/'trace.npz');sen=np.load(P/'sensor_inputs.npz')['sensors'];ticks=np.load(P/'fixed_ticks_132.npz')
    plans=read(P/'fixed_plans_132.json');selected=sorted(x['case'] for x in plans if x['plan']['model_passed']);assert selected==[0,1,6,7]
    replays=[];engines={};guards={};targets={};bycase={}
    for row in plans:
        i=row['case'];x=row['inputs'];at=int(np.flatnonzero(tr['t']==1.)[0]);assert row['tick']==50
        for k in ['q','geometry','command']:assert np.array_equal(np.array(x[k]),tr[k][at,i]),(i,k)
        assert np.array_equal(np.array(x['gravity'],np.float32),sen[0,10,i,339:342]);assert x['reference'][0]==float(tr['estimated_root_x'][at,i])
        plan=build_plan(m,meta,ine,cal,np.array(x['q'],np.float32),np.array(x['gravity'],np.float32),np.array(x['geometry'],np.float32),np.array(x['reference']),np.array(x['command'],np.float32),x['endpoint'])
        expected=np.array([e['load_checks'][0]['command'] for e in plan['path']]);saved=np.array([e['load_checks'][0]['command'] for e in row['plan']['path']])
        err=float(np.max(abs(expected-saved)));assert err<2e-7
        assert plan['summary']==row['plan']['summary']
        engines[i]=GatedCommandPath(row['plan'],m);guards[i]=MeasuredGuard(m,row['plan'],np.array(x['geometry'],np.float32));targets[(i,50)]=np.array(x['command'],np.float32);bycase[i]=row
        replays.append(dict(case=i,command_max_error_rad=err,model_passed=plan['model_passed']));print('PLANNER',i,err,flush=True)
    records=read(P/'fixed_guard_records_132.json');guard_errors=[];pauses={i:0 for i in selected};loadmax={i:0. for i in selected};last_progress={}
    for r in records:
        i=r['case'];tick=r['tick'];engine=engines[i]
        target=engine.advance(r['progress_allowed'],r['estimated_support']['max_normal_bw'])
        assert engine.last_info==r['command_path'],(i,tick,'guarded command arithmetic')
        targets[(i,tick)]=target;last_progress[i]=dict(arc=engine.position,full_arc=float(engine.arc[-1]),fraction=float(engine.position/engine.arc[-1]))
        pauses[i]+=int(not r['progress_allowed']);loadmax[i]=max(loadmax[i],r['command_path'].get('demanded_normal_bw',0.))
        if tick%5==0 and tick<1000:
            at=int(np.flatnonzero(np.isclose(tr['t'],tick*.02,atol=1e-10))[0]);step=tick//5
            assert np.array_equal(np.array(r['geometry'],np.float32),tr['geometry'][at,i])
            a=guards[i].observe(tr['q'][at,i],sen[0,step,i,339:342],tr['geometry'][at,i]);saved=r['measured_guard']
            assert a['available']==saved['available']
            if a['available']:
                keys=['anchor_disagreement_m','foot_orientation_disagreement_rad','imu_disagreement_rad','plane_height_error_m','plane_slope_error_deg','pelvis_gap_m','patch_margin_m','housing_clearance_m']
                e=max(abs(a[k]-saved[k]) for k in keys);assert e<1e-9,(i,tick,e)
                assert a['anchor_valid']==saved['anchor_valid'];guard_errors.append(e)
    assert len(ticks['t'])==1000 and np.isfinite(ticks['after']).all()
    expected=np.zeros_like(ticks['selected']);expected[50:,selected]=True;assert np.array_equal(expected,ticks['selected'])
    cap=ticks['cap'];assert np.all(abs(cap-.003)<2e-10)
    delta=abs(ticks['after']-ticks['before']);assert np.all(delta<=cap[:,:,None]+3e-7)
    assert np.array_equal(ticks['before'][:,:,23:],ticks['after'][:,:,23:])
    span=torch.tensor(read(P/'bridge_contract.json')['span'],dtype=torch.float32,device='cuda:0')
    limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
    lo=torch.tensor([limits[n][0]+.03 for n in m.names[:23]],dtype=torch.float32,device='cuda:0');hi=torch.tensor([limits[n][1]-.03 for n in m.names[:23]],dtype=torch.float32,device='cuda:0')
    filter_errors=[]
    for tick in range(50,1000):
        for i in selected:
            ref=torch.tensor(targets[(i,tick)],device='cuda:0');prev=torch.tensor(ticks['before'][tick,i],device='cuda:0');actions=torch.tensor(ticks['actions'][tick,i],device='cuda:0')
            desired=ref[:23]+span*torch.tanh(actions);assert torch.all((desired>=lo)&(desired<=hi))
            prev[:23]+=(.1*(desired.clamp(min=lo,max=hi)-prev[:23])).clamp(-float(cap[tick,i]),float(cap[tick,i]))
            error=float(np.max(abs(prev.cpu().numpy()-ticks['after'][tick,i])));assert error==0.,(tick,i,error)
            filter_errors.append(error)
    starts={}
    for name in ['startup.npz','traction_startup.npz','bridge_startup.npz']:
        b=np.load(F/'baseline'/name);c=np.load(P/name);starts[name]={k:bool(np.array_equal(b[k],c[k])) for k in b.files};assert all(starts[name].values())
    baseline=np.load(F/'baseline/trace.npz');best=np.load(D/'normal_landing3s_06/candidate/trace.npz');prefix={};unchanged={};other=[i for i in range(24) if i not in selected]
    for k in ['body','q','command','reference','geometry','torque']:
        mask=tr['t']<=1.;prefix[k]=bool(np.array_equal(baseline[k][mask],tr[k][mask]));assert prefix[k]
        unchanged[k]=dict(exact=bool(np.array_equal(best[k][:,other],tr[k][:,other])),max_error=float(np.max(abs(best[k][:,other]-tr[k][:,other]))))
    cr=read(P/'results.json')['episodes'];br=read(D/'normal_landing3s_06/candidate/results.json')['episodes'];assert len(cr)==24 and all(r['steps']==1000 for r in cr)
    pairs={}
    for label,indices in [('common_clean',[i for i in range(24) if clean(br[i]) and clean(cr[i])]),('strict_both',[i for i in range(24) if clean(br[i]) and clean(cr[i]) and not br[i]['traction_bad_ticks'] and not cr[i]['traction_bad_ticks']])]:
        pairs[label]=dict(cases=indices,windows={})
        for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
            pairs[label]['windows'][window]={}
            for key in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']:
                a=np.array([br[i][window][key] for i in indices]);b=np.array([cr[i][window][key] for i in indices]);pairs[label]['windows'][window][key]=dict(n=len(indices),baseline=float(a.mean()) if len(a) else None,candidate=float(b.mean()) if len(b) else None,percent_change=float((b.mean()/a.mean()-1)*100) if len(a) else None)
    exact=all(x['exact'] for x in unchanged.values())
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit_completed=True,passed=exact,software_audit_passed=True,exact_preservation_gate_passed=exact,
        physical_admission_passed=False,actual_input_action_reward_audit=validation,planner_replay=replays,
        every20ms_selected_filter_replay_exact=True,filter_replay_device=torch.cuda.get_device_name(0),selected_filter_records=len(filter_errors),every20ms_cap_checked=True,max_step_rad=float(delta.max()),fixed14fingers_exact=True,
        sampled_current_guard_replay=dict(samples=len(guard_errors),max_error=max(guard_errors) if guard_errors else None,scope='Saved0.1s raw sensor/pose samples only; every20ms guard permission/nominal progression replayed separately, not all20ms raw estimator inputs.'),
        exact_starts=starts,locked_first1=prefix,unselected_full20_exact=unchanged,paused_guard_ticks=pauses,max_assumed_normal_load_bw=loadmax,last_path_progress=last_progress,
        best_probe=summarize(br),candidate=summarize(cr),lost_best_clean_cases=[i for i in range(24) if clean(br[i]) and not clean(cr[i])],gained_best_clean_cases=[i for i in range(24) if not clean(br[i]) and clean(cr[i])],
        remaining_target_cases=[i for i in [0,1,2,3,6,7,10,11] if not clean(cr[i])],added_physical_cases=[i for i in range(24) if not any(br[i]['failures'].values()) and any(cr[i]['failures'].values())],paired_effort=pairs,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(P/'fixed_contract_132.json'):sha(P/'fixed_contract_132.json')})
    outpath.write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['candidate','lost_best_clean_cases','gained_best_clean_cases','remaining_target_cases','added_physical_cases','exact_preservation_gate_passed']}),flush=True)
if __name__=='__main__':main()

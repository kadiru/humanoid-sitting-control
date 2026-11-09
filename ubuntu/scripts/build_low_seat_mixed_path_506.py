"""ONE bounded partial-kinematic/native-auxiliary interior-point minimax run; no root or physical execution."""
from low_seat_mixed_execution_506 import *
O=D/'mixed_path_506'
def main():
    launch=read(D/'mixed_path_launch_506.json');assert launch['compile_actual_exit']==0 and launch['dependencies_actual_exit_verified']
    for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
    O.mkdir(exist_ok=False);d=load();ev=Evaluation(d);coords=Coordinates();started=time.monotonic()
    f=d['initial'].copy();y=d['seed']['Y'].copy();J=d['seed']['J'].copy();drops=d['drops'].copy()
    np.savez_compressed(O/'initial.npz',features=f,actualY=y,J=J,X=f[:,:24]);write(O/'initial_drop_state.json',[pair(x) for x in drops]);write(O/'specification.json',SPEC)
    initial,_=assess(d,f,y);history=[];trust=1.;lps=0;assemblies=0;maps=0;trials=0;accepted=0;certfail=0;mapfail=0;last=None;stop='declared major cap'
    for outer in range(SPEC['max_major_iterations']):
        b=blocks(d,f,y,J);s=intervals(d,f,drops,trust);sub=O/f'major_{outer:02d}';save_context(sub,d,f,y,J,drops,trust,b,s);S,a=coords.get(b,sub)
        start,_=assess(d,f,y);mu=start['phi'];record=dict(major=outer,trust=trust,start=start,trials=[],accepted=False)
        w,meta=solve_mixed(sub/'primary',d,b,s,S,a);assemblies+=1;lps+=int(meta['actual_lp_call']);record['primary']=meta;terminal=meta['terminal_failure'];eligible=not terminal
        if eligible:
            theta=modeled_phi(d,b,w[:-1]);eligible=theta<mu;record.update(primary_modeled_phi=theta,primary_eligible=bool(eligible))
        else:record['primary_eligible']=False
        if eligible:
            ceiling=(mu+theta)/2;record['witness_ceiling']=ceiling;w,meta=solve_mixed(sub/'interior',d,b,s,S,a,ceiling);assemblies+=1;lps+=int(meta['actual_lp_call']);record['interior']=meta;terminal=meta['terminal_failure']
            if not terminal:
                for k in range(SPEC['max_dyadic_trials']):
                    folder=sub/f'trial_{k}';folder.mkdir();t=w[:-1]*(2.**-k);np.save(folder/'direction.npy',t)
                    cert=certificate(s,t);write(folder/'certificate.json',cert);at=dict(dyadic=k,scale=2.**-k,certificate_pass=cert['passed'],accepted=False);record['trials'].append(at)
                    if not cert['passed']:
                        certfail+=1;at['reason']='exact interval certificate rejected unchanged dyadic direction';write(folder/'decision.json',at);continue
                    maps+=1;assert maps<=120
                    try:nf,nd,m=decode(d,f,drops,t,s)
                    except (ValueError,AssertionError) as exc:
                        mapfail+=1;at.update(map_pass=False,reason=type(exc).__name__+': '+str(exc));write(folder/'decision.json',at);continue
                    at['map_pass']=True;np.savez_compressed(folder/'mapped.npz',features=nf);write(folder/'mapped.json',m)
                    ny,changed=ev.evaluate(f[:,:24],nf[:,:24],y,'value');trials+=1
                    state,rows=assess(d,nf,ny);modelphi=encoded_prediction(d,f,nf,y,J);pred=mu-modelphi;actual=mu-state['phi'];ratio=actual/pred if pred>0 else None
                    take=bool(state['auxiliary_acceptance_domain_pass'] and pred>0 and actual>0 and ratio>=.1)
                    at.update(summary=state,predicted_phi=modelphi,predicted_reduction=pred,actual_reduction=actual,ratio=ratio,accepted=take,changed_value_states=len(changed))
                    np.savez_compressed(folder/'actual.npz',Y=ny,changed=changed,**rows);write(folder/'decision.json',at)
                    if take:
                        nJ,jchanged=ev.evaluate(f[:,:24],nf[:,:24],J,'jacobian');np.savez_compressed(folder/'accepted_J.npz',J=nJ,changed=jchanged)
                        f,y,J,drops=nf,ny,nJ,nd;accepted+=1;last=[outer,k];record['accepted']=True
                        if ratio>.75:trust=min(1.,2*trust)
                        break
        history.append(record);write(sub/'record.json',record)
        with (O/'history.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        final,_=assess(d,f,y);print(json.dumps(dict(major=outer,actual_lps=lps,maps=maps,actual_trial_states=trials,accepted=accepted,phi=final['phi'],actual_hard_min=final['actual_hard_min'],actual_gate=final['actual_numerical_admission_gate'],elapsed_s=time.monotonic()-started)),flush=True)
        if terminal:stop='terminal LP failure: no optimal independently certified reconstructed witness; no retry';break
        if final['actual_numerical_admission_gate']:stop='all actual numerical rows and unit closure passed; geometry/filter still pending';break
        if not record['accepted']:trust*=.5
        if trust<1/32:stop='declared trust floor';break
    final,rows=assess(d,f,y)
    target=np.c_[f[:,71],f[:,3],f[:,16:24]]
    np.savez_compressed(O/'final.npz',features=f,X=f[:,:24],Y=y,J=J,target=target,rows=rows['actual_rows'],equalities=rows['actual_equalities'],**rows)
    write(O/'final_drop_state.json',[pair(v) for v in drops])
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,specification=SPEC,stop=stop,major_iterations=len(history),assembled_subproblems=assemblies,actual_lp_calls=lps,
        finite_maps=maps,actual_trial_evaluations=trials,certificate_failures=certfail,map_failures=mapfail,accepted_solver_iterates=accepted,last_accepted=last,initial=initial,final=final,
        wrapper_counts=ev.counts,new_local_block_solves=coords.solves,new_inverse_certificates=coords.certificates,initial_inverse_inherited=True,accepted_solver_iterate_is_not_actual_path=True,initial_actualX_Y_J_inherited=True,initial_root_repeated=False,
        final_actual_state_changed=bool(np.any(f[:,:24]!=d['seed']['X'])),sampled_geometry_pending=True,independent_audit_pending=True,
        actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,physical_admission=False,elapsed_s=time.monotonic()-started)
    write(O/'results.json',result);write(O/'SHA256.json',{str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()});print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

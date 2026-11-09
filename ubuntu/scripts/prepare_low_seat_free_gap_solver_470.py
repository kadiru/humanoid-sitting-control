"""Saved-matrix preparation for new free-gap JOINT LPs; NO root/optimizer/candidate."""
import json,datetime
from low_seat_free_gap_solver_470 import *
O=D/'free_gap_solver_preparation_470'
def main():
 assert read(D/'exact_gap_chart_audit_exit_469.json')['actual_exit']==0
 assert read(D/'exact_gap_chart_audit_v2_469/independent_audit.json')['passed'];O.mkdir(exist_ok=False)
 d=load_saved();X=d['initialX'];Y=d['initialY'];J=d['initialJ'];drops=initial(d['gap_domain']);target=d['initial_target'];g,_=gaps(drops,d['gap_domain']);target[d['gap_layout']['normal_states'],0]=g
 a=assemble(d,X,Y,J,target,drops,1.);save_assembly(a,O/'predictor');merit=max(0.,-a['r'][d['a']['soft']&d['a']['keep']].min());c=assemble(d,X,Y,J,target,drops,1.,'correction',merit/2,a['z']);save_assembly(c,O/'correction')
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,reference='Original440 saved mechanics/J only. New target gaps are NOT yet lifted; this is LP assembly validation, not candidate feasibility.',predictor_shape=a['matrix'].shape,corrector_shape=c['matrix'].shape,homogeneous_equalities=a['equality'].shape[0],independent_coordinates=d['B'].shape[1],full_control_bound_rows=2*len(a['z']),dependent_f_bounds_retained=True,normal_drop_bound_rows=412,all331888_final_rows_retained=True,all_buffers_retained=True,specification=SPEC,model_calls=0,root_calls=0,lp_calls=0,new_complete_paths=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()

"""Saved-only exact box-support preparation index; no arithmetic or solver replay."""
from pathlib import Path
import json,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def main():
 p=read(D/'box_support_preparation_546/preparation.json');a=read(D/'box_support_audit_547/independent_audit.json')
 assert p['passed'] and a['passed'] and read(D/'box_support_audit_exit_547.json')['actual_exit']==0
 s=list(p['systems'].values())
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,systems=p['systems'],audit_systems=a['systems'],new_exact_box_systems=2,new_mixed_rows_tested=sum(x['mixed_rows_tested'] for x in s),new_proved_removed_rows=sum(x['proved_removed_rows'] for x in s),new_negative_box_support_tests=sum(x['negative_box_support_tests'] for x in s),missing_endpoint_untested=sum(x['missing_endpoint_untested'] for x in s),new_exact_support_terms=sum(x['exact_support_terms'] for x in s),native_variables_removed=0,scalar_variables_removed=0,old_scientific_calculations_repeated=False,lp_calls=0,small_examples=0,linear_solves=0,gram_factors=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,new_solver=False,new_whole_path_wrapper=False,new_robot_candidate=False,new_actual_path=False,physical_admission=False,task_complete=False,limits='Only exact-real implications of unchanged source inequalities. All unary premises retained; no row scaling, coordinate or box change. No numerical-solver performance or feasibility evidence; old542 timed-out internals remain unusable.')
 q=D/'box_support_scope_547.json';assert not q.exists();q.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['systems','audit_systems']},indent=2),flush=True)
if __name__=='__main__':main()

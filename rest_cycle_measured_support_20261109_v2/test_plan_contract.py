"""Independent malformed-input checks; no physics or model admission claim."""
import copy
import unittest
import numpy as np
from plan_contract import check_plan


def example():
    nodes=[]
    for i in range(3):
        nodes.append(dict(q=[i*.001]*23+[0.]*14,command=[i*.001]*23+[0.]*14,
            root=np.eye(4).tolist(),load=i*.2,gap=.012*(2-i),marker_gap=.012*(2-i),
            model_effort_ratio=.8,model_cop=[0.,0.,0.],model_tau=[0.]*23,
            phase='approach' if i==0 else 'landing'))
    back=copy.deepcopy(nodes[::-1])
    def schedule(rows):
        current=np.array([r['command'] for r in rows[:-1]])
        commands=np.array([r['command'] for r in rows[1:]])
        desired=current[:,:23]+10*(commands[:,:23]-current[:,:23])
        return dict(commands=commands.tolist(),desired=desired.tolist(),seconds=.04)
    return dict(nodes=nodes,return_nodes=back,forward=schedule(nodes),backward=schedule(back),
                nominal_rest_settle_s=1.54,nominal_return_settle_s=2.54,
                rest_load=.4,rest_model_effort_ratio=.8)


def check(p):
    return check_plan(p,[-2.]*23,[2.]*23,[0.]*37)


class PlanContractChecks(unittest.TestCase):
    def test_well_formed_contract_passes_only_structure(self):
        r=check(example());self.assertTrue(r['passed']);self.assertFalse(r['physical_success'])
    def test_late_nonfinite_load_is_rejected_before_minmax(self):
        for bad in (float('nan'),float('inf'),-float('inf')):
            p=example();p['nodes'][1]['load']=bad
            self.assertFalse(check(p)['passed'])
    def test_support_force_cannot_pull_or_exceed_original_range(self):
        for bad in (-.001,.80001):
            p=example();p['nodes'][1]['load']=bad
            self.assertFalse(check(p)['passed'])
    def test_actual_joint_margin_is_checked_independently_of_commands(self):
        p=example();p['nodes'][1]['q'][0]=1.999
        self.assertFalse(check(p)['passed'])
    def test_nonfinite_model_outputs_are_rejected(self):
        for key in ('model_cop','model_tau','q','command'):
            p=example();p['nodes'][1][key][0]=float('nan')
            self.assertFalse(check(p)['passed'])
    def test_root_must_be_a_proper_rigid_transform(self):
        for location in ((0,0),(3,3)):
            p=example();p['nodes'][1]['root'][location[0]][location[1]]=2.
            self.assertFalse(check(p)['passed'])
    def test_timing_claim_cannot_replace_command_count(self):
        for direction in ('forward','backward'):
            p=example();p[direction]['seconds']=0.
            self.assertFalse(check(p)['passed'])
    def test_nominal_timing_cannot_be_forged(self):
        for key in ('nominal_rest_settle_s','nominal_return_settle_s'):
            p=example();p[key]=0.
            self.assertFalse(check(p)['passed'])
    def test_endpoint_associations_are_ordered(self):
        p=example();p['nodes'][1]['command'][0]=.0015
        self.assertFalse(check(p)['passed'])
    def test_command_detour_cannot_borrow_straight_segment_geometry(self):
        p=example();p['nodes'].pop(1)
        # Filter and endpoint association still pass, but this interior command
        # no longer lies at the declared geometric interpolation fraction.
        p['forward']['commands'][0][0]=.0008
        commands=np.asarray(p['forward']['commands'])
        previous=np.vstack([p['nodes'][0]['command'],commands[:-1]])
        p['forward']['desired']=(previous[:,:23]+10*(commands[:,:23]-previous[:,:23])).tolist()
        r=check(p)
        self.assertFalse(r['passed'])
        self.assertTrue(any('declared_interpolation' in e for e in r['errors']))
    def test_return_must_join_the_same_pose_not_just_command(self):
        p=example();p['return_nodes'][0]['root'][0][3]=.001
        self.assertFalse(check(p)['passed'])
    def test_rest_load_metadata_must_match_rest_node(self):
        p=example();p['rest_load']=.1
        self.assertFalse(check(p)['passed'])
    def test_fixed_fingers_cannot_be_changed(self):
        p=example();p['forward']['commands'][0][23]=.001
        self.assertFalse(check(p)['passed'])
    def test_original_filter_limits_remain_required(self):
        p=example();p['forward']['desired'][0][0]=.1
        self.assertFalse(check(p)['passed'])
    def test_missing_paths_and_fields_fail_closed(self):
        for key in ('nodes','forward','rest_load'):
            p=example();del p[key]
            self.assertFalse(check(p)['passed'])
    def test_input_is_not_repaired_or_mutated(self):
        p=example();before=copy.deepcopy(p);check(p)
        self.assertEqual(p,before)


if __name__=='__main__':unittest.main()

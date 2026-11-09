"""Behavioral reward tests against the exact expression executed by the simulator."""
import ast,unittest
from pathlib import Path
import torch

def reward_function():
    tree=ast.parse(Path(__file__).with_name('run_transfer_effort.py').read_text())
    target=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="self.args.objective == 'effort'"][0]
    # Bind only the reward's explicit labels, never simulator state or actor inputs.
    source='def reward(total,reference,good,alive,first,phase,tracking,balance,transfer,friction_cost,delta):\n'
    source+='    from types import SimpleNamespace\n    self=SimpleNamespace(reward_reference=reference,alive=alive,phase=lambda:phase)\n    DT=.02\n'
    source+='\n'.join('    '+line for stmt in target.body for line in ast.unparse(stmt).splitlines())+'\n    return reward\n'
    scope={};exec(compile(source,'actual_reward_expression','exec'),scope);return scope['reward']

class RewardTests(unittest.TestCase):
    def run_reward(self,efforts,good=True,alive=True,first=False):
        n=len(efforts);t=lambda x:torch.full((n,),x)
        return reward_function()(torch.tensor(efforts),t(100.),t(good),t(alive),t(first),t(6.),t(1.),t(1.),t(1.),t(0.),torch.zeros(n,23))
    def test_lower_whole_body_torque_is_rewarded(self):
        r=self.run_reward([50.,80.,100.,120.]);self.assertTrue(bool((r[:-1]>r[1:]).all()))
        self.assertAlmostEqual(float(r[1]-r[2]),.02*2*.2,places=6)
    def test_losing_support_cannot_collect_savings(self):
        r=self.run_reward([0.,100.,10000.],good=False);self.assertTrue(torch.equal(r,r[0].expand_as(r)))
        self.assertLess(float(r[0]),float(self.run_reward([10000.])[0]))
    def test_failure_has_no_effort_bonus(self):
        r=self.run_reward([0.,10000.],good=False,alive=False,first=True)
        self.assertTrue(torch.equal(r,torch.full_like(r,-5.)))
    def test_calibration_changes_reward_not_physical_metric(self):
        r=self.run_reward([100.,100000.]);self.assertTrue(torch.isfinite(r).all())

class RetentionTests(unittest.TestCase):
    @staticmethod
    def row(success,effort=100.,failure=False):
        return dict(success=success,failures={'fall':failure},metrics_whole_trial={'total_squared_torque_Nm2':effort})
    def test_lower_effort_cannot_hide_lost_success(self):
        from run_transfer_effort_study import retention
        baseline=[self.row(True),self.row(True)]
        result=retention([self.row(True,1.),self.row(False,0.)],baseline,baseline)
        self.assertFalse(result['eligible']);self.assertEqual(result['lost_success_conditions'],[1])
    def test_new_failure_rejected_even_on_previously_unsupported_case(self):
        from run_transfer_effort_study import retention
        baseline=[self.row(True),self.row(False)]
        result=retention([self.row(True),self.row(False,failure=True)],baseline,baseline)
        self.assertFalse(result['eligible']);self.assertEqual(result['new_physical_failures'],[1])
if __name__=='__main__':unittest.main()

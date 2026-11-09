import unittest
import torch
from humanoid_sitting.support_capped_impact import CappedImpactStudy, SupportedHoldTicks
from humanoid_sitting.support_objective import support_objective


class CappedImpactTests(unittest.TestCase):
    def sample(self):
        m={k:torch.tensor([v,v]) for k,v in dict(gravity_z=-1.,speed=0.,angular_speed=0.,
            support=.75,goal_error2=0.,tilt2=0.,leg_effort2=.1,effort2=.1,peak_force_ratio=.75,vz=0.).items()}
        m.update({k:torch.tensor([v,v]) for k,v in dict(inside=True,height_ok=True,failure=False).items()})
        return m

    def call(self,obj,m,alive=None,dt=.02):
        return obj(m,torch.zeros(2,23),torch.zeros(2,23),torch.ones(2,dtype=torch.bool) if alive is None else alive,dt)

    def test_exact_150_ticks_and_interruption(self):
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            timer=SupportedHoldTicks(); good=torch.ones(2,dtype=torch.bool,device=device)
            for _ in range(149):timer.update(good)
            self.assertEqual(timer.count(0),149)
            timer.update(good); self.assertEqual(timer.count(0),150)
            timer.reset(torch.tensor([0],device=device)); timer.update(good)
            self.assertEqual(timer.count(0),1); self.assertEqual(timer.count(1),151)
            timer.update(torch.zeros_like(good)); timer.update(good)
            self.assertEqual(int(timer.current[1]),1); self.assertEqual(timer.count(1),151)
            timer.reset(); self.assertEqual(timer.count(0),0)

    def test_legacy_exact_reward_parity(self):
        m=self.sample();m['peak_force_ratio'][:]=2.5
        actual=self.call(CappedImpactStudy('legacy'),m); expected=self.call(support_objective,m)
        self.assertTrue(torch.equal(actual[0],expected[0]))
        self.assertTrue(all(torch.equal(v,expected[2][k]) for k,v in actual[2].items()))

    def test_only_impact_changes(self):
        m=self.sample();m['peak_force_ratio'][:]=1.5
        actual=self.call(CappedImpactStudy('capped'),m); expected=self.call(support_objective,m)
        self.assertTrue(torch.equal(actual[1],expected[1]))
        for k,v in actual[2].items():
            if k!='impact':self.assertTrue(torch.equal(v,expected[2][k]),k)

    def test_peak_cap_duration_independence_and_monotonicity(self):
        for dt,repeats in ((.02,1),(.01,2),(.02,20)):
            obj=CappedImpactStudy('capped');m=self.sample();charge=0.
            for force in (1.1,1.5,1.2,2.,10.,100.):
                m['peak_force_ratio'][:]=force
                for _ in range(repeats):
                    delta=float(self.call(obj,m,dt=dt)[2]['impact'][0])*dt
                    self.assertLessEqual(delta,0);charge+=delta
                    self.assertGreaterEqual(charge,-5.000001)
            self.assertAlmostEqual(charge,-5.,places=5)

    def test_independent_reset_and_failed_episode(self):
        obj=CappedImpactStudy('capped');m=self.sample();m['peak_force_ratio'][:]=2.
        self.call(obj,m);obj.reset(torch.tensor([0]))
        charge=self.call(obj,m)[2]['impact']*.02
        self.assertTrue(torch.equal(charge,torch.tensor([-5.,0.])))
        reward,good,_=self.call(obj,m,torch.zeros(2,dtype=torch.bool))
        self.assertTrue(torch.equal(reward,torch.zeros(2)));self.assertFalse(good.any())

    def test_failure_penalty_retained_and_accounting(self):
        obj=CappedImpactStudy('capped');m=self.sample();m['failure'][:]=True;m['peak_force_ratio'][:]=20.
        reward,good,_=self.call(obj,m)
        self.assertFalse(good.any());self.assertTrue((reward < -9).all())
        totals=obj.pop_totals();actual=totals.pop('actual_total')
        self.assertAlmostEqual(actual,sum(totals.values()),places=5)
        self.assertEqual(totals['failure'],-10.)


if __name__=='__main__':unittest.main()

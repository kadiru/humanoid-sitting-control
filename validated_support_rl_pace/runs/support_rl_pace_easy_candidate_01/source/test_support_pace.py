"""Real-checkpoint equivalence, causal progression, filtering and expansion tests."""
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from humanoid_sitting.deployment import DeploymentStudent
from humanoid_sitting.soft_landing import ContactEstimator, SoftLandingController
from humanoid_sitting.support_pace import PaceController, update_pace, migrate_state
ROOT=Path(__file__).resolve().parents[1]
PARENT=Path('/data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt')

class PaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.policy,_=DeploymentStudent.load(ROOT/'runs/deployment_status_mask_05/student.pt','cpu')
        cls.contact=ContactEstimator.load(ROOT/'runs/contact_estimator_02/contact.pt','cpu')

    def test_neutral_matches_frozen_controller_over_history(self):
        old=SoftLandingController(self.policy,self.contact,mode='actions')
        new=PaceController(self.policy,self.contact,mode='actions')
        a=b=None;generator=torch.Generator().manual_seed(51601)
        with torch.inference_mode():
            for _ in range(80):
                x=self.policy.mean[None,None,:]+.1*self.policy.std[None,None,:]*torch.randn(4,1,109,generator=generator)
                x[...,11:13]=1.3;x[...,108]=1.
                new.progress_rate=torch.ones(4)
                y,a=old(x,a);z,b=new(x,b)
                torch.testing.assert_close(y,z,rtol=0,atol=0)
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(old.last_contact,new.last_contact,rtol=0,atol=0)

    def test_rate_applies_before_decoder_without_slowing_contact(self):
        normal=PaceController(self.policy,self.contact,mode='actions')
        slow=PaceController(self.policy,self.contact,mode='actions')
        with torch.inference_mode():
            x=self.policy.mean[None,None,:].repeat(4,1,1);x[...,11:13]=1.3
            old=torch.randn(1,4,128)*.1
            slow.progress_rate=torch.full((4,),.25)
            _,a=normal(x,old);_,b=slow(x,old)
            torch.testing.assert_close(b,old+.25*(a-old),rtol=1e-5,atol=1e-7)
            torch.testing.assert_close(normal.last_contact,slow.last_contact,rtol=0,atol=0)
            self.assertGreater(float((a-b).abs().max()),.001)

    def test_filter_bounded_neutral_and_refusal_holds(self):
        rate=torch.ones(4);ready=torch.tensor([True,True,False,True])
        torch.testing.assert_close(update_pace(rate,torch.zeros(4),ready),rate,rtol=0,atol=0)
        for _ in range(150):
            new=update_pace(rate,torch.tensor([-2.,2.,-2.,-1.]),ready)
            self.assertTrue(((new>=.25)&(new<=1.)).all())
            self.assertLessEqual(float((new-rate).abs().max()),.040001)
            self.assertEqual(float(new[2]),1.)
            rate=new
        self.assertLess(float(rate[0]),.25001);self.assertEqual(float(rate[1]),1.)

    def test_expansion_preserves_policy_and_critic_function(self):
        parent=torch.load(PARENT,map_location='cpu',weights_only=False)
        for group,actor in [('actor_state_dict',True),('critic_state_dict',False)]:
            old=parent[group];template={k:v.clone() for k,v in old.items()}
            for key in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
                template[key]=torch.empty(1,145)
            template['mlp.0.weight']=torch.empty(128,145)
            if actor:
                template['mlp.4.weight']=torch.empty(24,128);template['mlp.4.bias']=torch.empty(24)
                template['distribution.log_std_param']=torch.empty(24)
            new=migrate_state(old,template,actor)
            x=old['obs_normalizer._mean']+torch.randn(32,144)*old['obs_normalizer._std']
            def forward(state,obs):
                z=(obs-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+1e-8)
                for i in (0,2):z=F.elu(F.linear(z,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias']))
                return F.linear(z,state['mlp.4.weight'],state['mlp.4.bias'])
            for pace in (.25,.6,1.):
                z=forward(new,torch.cat((x,torch.full((32,1),pace)),1));y=forward(old,x)
                torch.testing.assert_close(z[:,:y.shape[1]],y,rtol=1e-5,atol=1e-6)
                if actor:self.assertEqual(float(z[:,-1].abs().max()),0.)
            if actor:self.assertAlmostEqual(float(new['distribution.log_std_param'][-1].exp()),.5)

if __name__=='__main__':unittest.main()

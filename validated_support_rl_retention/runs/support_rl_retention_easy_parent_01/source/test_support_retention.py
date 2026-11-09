"""Update parity, frozen-reference isolation, and analytic KL gradient checks."""
import copy
from pathlib import Path
import unittest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from humanoid_sitting.support_retention import RetentionPPO

PARENT=Path('/data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt')

class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.parent=torch.load(PARENT,map_location='cpu',weights_only=False)

    def algorithm(self,kind=RetentionPPO,coefficient=0.,freeze=False):
        obs=TensorDict({'policy':torch.zeros(4,144)},batch_size=[4])
        groups={'actor':['policy'],'critic':['policy']}
        actor=MLPModel(obs,groups,'actor',23,hidden_dims=[128,128],obs_normalization=True,
            distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=.04,std_type='log'))
        critic=MLPModel(obs,groups,'critic',1,hidden_dims=[128,128],obs_normalization=True)
        actor.load_state_dict(self.parent['actor_state_dict']);critic.load_state_dict(self.parent['critic_state_dict'])
        storage=RolloutStorage('rl',4,8,obs,[23],'cpu')
        alg=kind(actor,critic,storage,num_learning_epochs=3,num_mini_batches=4,
                learning_rate=3e-5,schedule='fixed',entropy_coef=0.)
        if kind is RetentionPPO:alg.configure_retention(self.parent['actor_state_dict'],coefficient,freeze)
        return alg

    def observations(self):
        state=self.parent['actor_state_dict']
        x=state['obs_normalizer._mean']+torch.randn(4,144)*state['obs_normalizer._std']*.1
        return TensorDict({'policy':x},batch_size=[4])

    def collect(self,alg):
        torch.manual_seed(51701)
        with torch.inference_mode():
            obs=self.observations()
            for tick in range(8):
                alg.act(obs);obs=self.observations()
                alg.process_env_step(obs,torch.randn(4),torch.tensor([tick==7,False,False,False]),{})
            alg.compute_returns(obs)

    def test_zero_weight_matches_upstream_exactly(self):
        baseline=self.algorithm(PPO);candidate=self.algorithm()
        self.collect(baseline);self.collect(candidate)
        torch.manual_seed(51703);a=baseline.update()
        torch.manual_seed(51703);b=candidate.update()
        for key in a:self.assertEqual(a[key],b[key])
        for name in ('actor','critic'):
            for key,value in getattr(baseline,name).state_dict().items():
                torch.testing.assert_close(value,getattr(candidate,name).state_dict()[key],rtol=0,atol=0)
        for i,state in baseline.optimizer.state_dict()['state'].items():
            for key,value in state.items():
                torch.testing.assert_close(value,candidate.optimizer.state_dict()['state'][i][key],rtol=0,atol=0)

    def test_reference_and_normalizers_frozen_during_actual_update(self):
        alg=self.algorithm(coefficient=1.,freeze=True)
        before={name:copy.deepcopy(getattr(alg,name).state_dict()) for name in ('actor','critic','reference')}
        self.collect(alg);result=alg.update()
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
        for key,value in before['reference'].items():torch.testing.assert_close(value,alg.reference.state_dict()[key],rtol=0,atol=0)
        for name in ('actor','critic'):
            after=getattr(alg,name).state_dict()
            for key,value in before[name].items():
                if key.startswith('obs_normalizer.'):torch.testing.assert_close(value,after[key],rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(value,after[key]) for key,value in before[name].items() if key.startswith('mlp.')))
        optimized={id(p) for g in alg.optimizer.param_groups for p in g['params']}
        self.assertTrue(all(not p.requires_grad and p.grad is None and id(p) not in optimized for p in alg.reference.parameters()))

    def test_kl_gradient_pulls_mean_toward_parent_without_rng_side_effect(self):
        alg=self.algorithm(coefficient=1.);obs=self.observations()
        with torch.no_grad():alg.actor.mlp[-1].bias.add_(.01)
        alg.actor(obs,stochastic_output=True)
        rng=torch.get_rng_state().clone();kl=alg.reference_kl(obs)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        expected=(.5*(.01/alg.actor.output_std).square()).sum(-1).mean()
        torch.testing.assert_close(kl,expected,rtol=1e-4,atol=1e-6)
        kl.backward()
        self.assertTrue((alg.actor.mlp[-1].bias.grad>0).all())
        self.assertTrue(all(p.grad is None for p in alg.reference.parameters()))

    def test_nonzero_weight_changes_actual_ppo_update(self):
        a=self.algorithm(coefficient=0.,freeze=True);b=self.algorithm(coefficient=1.,freeze=True)
        self.collect(a);self.collect(b)
        torch.manual_seed(51709);a.update()
        torch.manual_seed(51709);b.update()
        self.assertTrue(any(not torch.equal(x,y) for x,y in zip(a.actor.parameters(),b.actor.parameters())))

if __name__=='__main__':unittest.main()

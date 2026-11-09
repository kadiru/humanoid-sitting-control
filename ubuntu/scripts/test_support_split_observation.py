"""Distance/patch independence, missing-data semantics, and migration tests."""
import unittest
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from humanoid_sitting.support_split_observation import (
    distance_features, expand_parent_state, install_split_linear)

PARENT = '/data/humanoid-sitting-perception/support_rl_independent_train_04/final.pt'


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.r = dict(stamp=1., valid=True, normal_P=[0,0,1], clearance_m=.13,
                      measurement_age_s=.08, allowance_m=.015)

    def test_distance_does_not_require_patch(self):
        v = distance_features(self.r, 1.)
        self.assertEqual(v[4], 1.)
        self.assertAlmostEqual(v[0], .13, places=7)
        # Signed distance is retained even below the fitted plane.
        self.assertLess(distance_features(self.r | dict(clearance_m=-.01), 1.)[0], 0.)

    def test_unavailable_never_leaks_previous_values(self):
        for change in (dict(valid=False), dict(stamp=.98), dict(allowance_m=.02),
                       dict(measurement_age_s=.12), dict(normal_P=[0,0,0]),
                       dict(clearance_m=float('nan')), dict(normal_P=None)):
            v = distance_features(self.r | change, 1.)
            np.testing.assert_array_equal(v[:5], np.zeros(5))
            self.assertEqual(v[5], 1.)
            self.assertTrue(np.isfinite(v).all())


class MigrationTests(unittest.TestCase):
    def test_parent_actions_and_values_preserved_and_new_channels_learn(self):
        parent = torch.load(PARENT, map_location='cpu', weights_only=False)
        for device in ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []):
            for role, outputs in [('actor',23), ('critic',1)]:
                old_obs = TensorDict({'policy':torch.zeros(16,144,device=device)}, [16])
                new_obs = TensorDict({'policy':torch.zeros(16,151,device=device)}, [16])
                kwargs = dict(hidden_dims=[128,128],obs_normalization=True)
                if role == 'actor':
                    kwargs['distribution_cfg'] = dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',
                                                      init_std=.04,std_type='log')
                old = MLPModel(old_obs,{role:['policy']},role,outputs,**kwargs).to(device)
                new = MLPModel(new_obs,{role:['policy']},role,outputs,**kwargs).to(device)
                state = parent[role+'_state_dict']; old.load_state_dict(state)
                new.load_state_dict(expand_parent_state(state)); install_split_linear(new)
                old.eval(); new.eval()
                with torch.no_grad():
                    for _ in range(10):
                        old_obs['policy'] = (state['obs_normalizer._mean'] +
                            torch.randn(16,144)*state['obs_normalizer._std']*.3).to(device)
                        new_obs['policy'] = torch.cat((old_obs['policy'],torch.randn(16,7,device=device)),1)
                        torch.testing.assert_close(old(old_obs),new(new_obs),rtol=0,atol=0)
                optimizer = torch.optim.Adam(new.parameters(),lr=3e-5)
                before = new.mlp[0].weight[:,144:].detach().clone()
                new(new_obs).square().mean().backward(); optimizer.step()
                self.assertTrue((new.mlp[0].weight[:,144:] != before).any())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

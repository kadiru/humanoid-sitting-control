# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Restricted RSL-RL 5.5.1 PPO update with a frozen parent-policy KL penalty.

The update below follows the installed rsl_rl.algorithms.ppo.PPO.update. It
intentionally rejects extensions not exercised here. No installed package is
modified. Weight zero is tested against upstream PPO for exact update parity.
"""
import copy
import torch
from rsl_rl.algorithms import PPO


class RetentionPPO(PPO):
    def configure_retention(self, parent_state, coefficient, freeze_normalization=True):
        assert coefficient >= 0
        assert not self.actor.is_recurrent and not self.critic.is_recurrent
        assert not self.rnd and not self.symmetry and not self.is_multi_gpu
        assert not self.use_mixed_precision and self.schedule == 'fixed'
        self.reference = copy.deepcopy(self.actor)
        self.reference.load_state_dict(parent_state)
        self.reference.requires_grad_(False).eval()
        self.retention_coefficient = float(coefficient)
        self.freeze_normalization = freeze_normalization

    def reference_kl(self, observations):
        # Deterministic reference forward consumes no sampling RNG and uses its
        # own frozen sensor normalizer. Only the current policy gets gradients.
        with torch.no_grad():
            self.reference.distribution.update(self.reference(observations))
            parent = self.reference.output_distribution_params
        return self.actor.get_kl_divergence(parent, self.actor.output_distribution_params).mean()

    def update(self):
        totals = dict(value=0., surrogate=0., entropy=0., parent_kl=0.)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages-batch.advantages.mean())/(batch.advantages.std()+1e-8)
            self.actor(batch.observations, masks=batch.masks,
                       hidden_state=batch.hidden_states[0], stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            entropy = self.actor.output_entropy
            ratio = torch.exp(log_prob-torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages)*ratio
            clipped = -torch.squeeze(batch.advantages)*torch.clamp(ratio,1.-self.clip_param,1.+self.clip_param)
            surrogate_loss = torch.max(surrogate,clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = batch.values+(values-batch.values).clamp(-self.clip_param,self.clip_param)
                value_loss = torch.max((values-batch.returns).pow(2),(value_clipped-batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns-values).pow(2).mean()
            kl = self.reference_kl(batch.observations)
            loss = surrogate_loss+self.value_loss_coef*value_loss-self.entropy_coef*entropy.mean()
            if self.retention_coefficient:
                loss = loss+self.retention_coefficient*kl
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
            self.optimizer.step()
            for key,value in [('value',value_loss),('surrogate',surrogate_loss),('entropy',entropy.mean()),('parent_kl',kl)]:
                totals[key] += value.item()
        if not self.freeze_normalization:
            obs = self.storage.observations.flatten(0,1)
            self.actor.update_normalization(obs)
            self.critic.update_normalization(obs)
        count = self.num_learning_epochs*self.num_mini_batches
        self.storage.clear()
        return {key:value/count for key,value in totals.items()}

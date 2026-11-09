"""PPO update excluding reset-warmup samples from every optimized statistic.

Actor and critic are separate networks. A disabled actor has grad=None, so
Adam cannot move it through retained momentum. No normalization, recurrence,
mixed precision, symmetry or RND is supported by this bounded study.
"""
import torch

def optimize_batch(alg,batch,actor_enabled=True):
    active=batch.observations['learn'].flatten().bool()
    if not active.any():return None
    obs=batch.observations[active]
    target=batch.returns[active];oldvalue=batch.values[active]
    value=alg.critic(obs)
    clipped=oldvalue+(value-oldvalue).clamp(-alg.clip_param,alg.clip_param)
    value_loss=torch.maximum((value-target).square(),(clipped-target).square()).mean()
    loss=alg.value_loss_coef*value_loss
    surrogate=value_loss.new_zeros(());entropy=value_loss.new_zeros(());kl=value_loss.new_zeros(())
    if actor_enabled:
        alg.actor(obs,stochastic_output=True)
        logp=alg.actor.get_output_log_prob(batch.actions[active])
        ratio=(logp-batch.old_actions_log_prob[active].flatten()).exp()
        advantage=batch.advantages[active].flatten()
        surrogate=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(1-alg.clip_param,1+alg.clip_param)).mean()
        entropy=alg.actor.output_entropy.mean()
        with torch.no_grad():
            kl=alg.actor.get_kl_divergence(tuple(p[active] for p in batch.old_distribution_params),alg.actor.output_distribution_params).mean()
        loss=loss+surrogate-alg.entropy_coef*entropy
    alg.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if actor_enabled:torch.nn.utils.clip_grad_norm_(alg.actor.parameters(),alg.max_grad_norm)
    else:assert all(p.grad is None for p in alg.actor.parameters())
    torch.nn.utils.clip_grad_norm_(alg.critic.parameters(),alg.max_grad_norm)
    alg.optimizer.step()
    return dict(value=float(value_loss.detach()),surrogate=float(surrogate.detach()),entropy=float(entropy.detach()),active_kl=float(kl),active_samples=int(active.sum()))

def install_active_update(alg,enabled):
    assert not alg.actor.is_recurrent and not alg.critic.is_recurrent
    assert not alg.rnd and not alg.symmetry and not alg.is_multi_gpu and not alg.use_mixed_precision
    assert alg.schedule=='fixed'
    def update():
        mask=alg.storage.observations['learn'].bool()
        advantages=alg.storage.advantages
        # RSL computes returns under inference_mode and may allocate advantages
        # there. Normalization must preserve that tensor's inference contract.
        with torch.inference_mode():
            chosen=advantages[mask]
            if chosen.numel()>1:
                advantages[mask]=(chosen-chosen.mean())/(chosen.std(unbiased=False)+1e-8)
        results=[]
        for batch in alg.storage.mini_batch_generator(alg.num_mini_batches,alg.num_learning_epochs):
            row=optimize_batch(alg,batch,enabled())
            if row is not None:results.append(row)
        alg.storage.clear()
        return {k:sum(r[k] for r in results)/max(1,len(results)) for k in ['value','surrogate','entropy','active_kl','active_samples']}
    alg.update=update

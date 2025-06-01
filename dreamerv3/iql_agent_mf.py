import re
from functools import partial as bind

import embodied
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ruamel.yaml as yaml
import pdb

from . import jaxagent
from . import jaxutils
from . import nets
from . import ninjax as nj

f32 = jnp.float32
treemap = jax.tree_util.tree_map
sg = lambda x: treemap(jax.lax.stop_gradient, x)
cast = jaxutils.cast_to_compute
sample = lambda dist: {
    k: v.sample(seed=nj.seed()) for k, v in dist.items()}


@jaxagent.Wrapper
class Agent(nj.Module):

  configs = yaml.YAML(typ='safe').load(
      (embodied.Path(__file__).parent / 'configs.yaml').read())

  def __init__(self, obs_space, act_space, config):
    self.obs_space = {
        k: v for k, v in obs_space.items() if not k.startswith('log_')}
    self.act_space = {
        k: v for k, v in act_space.items() if k != 'reset'}
    self.config = config
    enc_space = {
        k: v for k, v in obs_space.items()
        if k not in ('is_first', 'is_last', 'is_terminal', 'reward') and
        not k.startswith('log_') and re.match(config.enc.spaces, k)}
    embodied.print('Encoder:', {k: v.shape for k, v in enc_space.items()})

    # World Model
    #self.enc = {
    #    'simple': bind(nets.SimpleEncoder, **config.enc.simple),
    #}[config.enc.typ](enc_space, name='enc')

    # Actor
    kwargs = {}
    kwargs['shape'] = {
        k: (*s.shape, s.classes) if s.discrete else s.shape
        for k, s in self.act_space.items()}
    kwargs['dist'] = {
        k: config.actor_dist_disc if v.discrete else config.actor_dist_cont
        for k, v in self.act_space.items()}
    self.actor = nets.MLP(**kwargs, **config.actor, name='actor')
    print(self.actor.dist)
    #print(affafwffwfawf)
    self.retnorm = jaxutils.Moments(**config.retnorm, name='retnorm')
    self.valnorm = jaxutils.Moments(**config.valnorm, name='valnorm')
    self.advnorm = jaxutils.Moments(**config.advnorm, name='advnorm')

    # Critic
    self.value = nets.MLP((), name='value', **self.config.value)
    #self.slowvalue = nets.MLP(
    #        (), name='slowvalue', **self.config.value, dtype='float32')
    self.critic = nets.MLP((), name='critic', **self.config.critic)
    self.slowcritic = nets.MLP(
        (), name='slowcritic', **self.config.critic, dtype='float32')
    self.updaterQ = jaxutils.SlowUpdater(
        self.critic, self.slowcritic,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='updaterQ')

    '''
    self.updaterV = jaxutils.SlowUpdater(
        self.value, self.slowvalue,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='updaterV')
    '''

    # Optimizer
    kw = dict(config.opt)
    lr = kw.pop('lr')
    if config.separate_lrs:
      lr = {f'agent/{k}': v for k, v in config.lrs.items()}
    self.opt = jaxutils.Optimizer(lr, **kw, name='opt')
    self.modules = [
        #self.enc, 
        self.actor, self.critic, self.value]
    scales = self.config.loss_scales.copy()
    self.scales = scales

  @property
  def policy_keys(self):
    return '/(enc|dyn|actor)/'

  @property
  def aux_spaces(self):
    spaces = {}
    spaces['stepid'] = embodied.Space(np.uint8, 20)
    if self.config.replay_context:
      latdtype = jaxutils.COMPUTE_DTYPE
      latdtype = np.float32 if latdtype == jnp.bfloat16 else latdtype
      spaces['deter'] = embodied.Space(latdtype, self.config.dyn.rssm.deter)
      spaces['stoch'] = embodied.Space(np.int32, self.config.dyn.rssm.stoch)
    return spaces

  def init_policy(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    prevlat = {
        'deter': jnp.zeros(batch_size),
        'stoch': jnp.zeros(batch_size),
    }
    return (prevlat, prevact)

  def init_train(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    prevlat = {
        'deter': jnp.zeros(batch_size),
        'stoch': jnp.zeros(batch_size),
    }
    return prevlat, prevact

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def policy(self, obs, carry, mode='train'):
    self.config.jax.jit and embodied.print(
        'Tracing policy function', color='yellow')
    prevlat, prevact = carry
    obs = self.preprocess(obs)
    #embed = obs['vector']
    embed = self.enc(obs, bdims=1)
    prevact = jaxutils.onehot_dict(prevact, self.act_space)
    out = {'stoch': embed, 'deter': embed}
    actor = self.actor(out, bdims=1)
    print(actor)
    lat, act = prevlat, sample(actor)

    outs = {}
    if self.config.replay_context:
      outs.update({k: out[k] for k in self.aux_spaces if k != 'stepid'})
      outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)

    outs['finite'] = {
        '/'.join(x.key for x in k): (
            jnp.isfinite(v).all(range(1, v.ndim)),
            v.min(range(1, v.ndim)),
            v.max(range(1, v.ndim)))
        for k, v in jax.tree_util.tree_leaves_with_path(dict(
            obs=obs, prevlat=prevlat, prevact=prevact,
            embed=embed, act=act, out=out, lat=lat,
        ))}

    assert all(
        k in outs for k in self.aux_spaces
        if k not in ('stepid', 'finite', 'is_online')), (
              list(outs.keys()), self.aux_spaces)

    act = {
        k: jnp.nanargmax(act[k], -1).astype(jnp.int32)
        if s.discrete else act[k] for k, s in self.act_space.items()}
    return act, outs, (lat, act)

  def train(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing train function', color='yellow')
    data = self.preprocess(data)
    stepid = data.pop('stepid')

    mets, (out, carry, metrics) = self.opt(
        self.modules, self.loss, data, carry, has_aux=True
    )
    metrics.update(mets)
    self.updaterQ()
    #self.updaterV()
    outs = {}

    if self.config.replay_context:
      outs['replay'] = {'stepid': stepid}
      outs['replay'].update({
          k: out['replay_outs'][k] for k in self.aux_spaces if k != 'stepid'})
      outs['replay']['stoch'] = jnp.argmax(
          outs['replay']['stoch'], -1).astype(jnp.int32)

    return outs, carry, metrics

  def loss(self, data, carry, update=True):
    metrics, losses, dists = {}, {}, {}
    prevlat, prevact = carry

    # Replay rollout
    prevacts = {
        k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1)
        for k in self.act_space}
    prevacts = jaxutils.onehot_dict(prevacts, self.act_space)
    #embed = self.enc(data)
    embed = data['vector']
    newlat = prevlat
    outs = {'deter': embed, 'stoch': embed}
    #newlat, outs = self.dyn.observe(prevlat, prevacts, embed, data['is_first'])
    '''
    rew_feat = outs if self.config.reward_grad else sg(outs)
    dists = dict(
        **self.dec(outs),
        reward=self.rew(rew_feat, training=True),
        cont=self.con(outs, training=True))
    losses = {k: -v.log_prob(f32(data[k])) for k, v in dists.items()}
    if self.config.contdisc:
      del losses['cont']
      softlabel = data['cont'] * (1 - 1 / self.config.horizon)
      losses['cont'] = -dists['cont'].log_prob(softlabel)
    dynlosses, mets = self.dyn.loss(outs, **self.config.rssm_loss)
    losses.update(dynlosses)
    metrics.update(mets)
    '''

    rew = data['reward'] 
    con = f32(~data['is_terminal'])
    acts = {'action': data['action']}
    critic = self.critic({**outs, **acts})
    value = self.value({**outs})
    discount = (1 - 1 / self.config.horizon) 

    # Value
    cur_q = sg(critic.mean())
    #slowvalue = self.slowvalue({**outs})
    adv_weight = jnp.where(value.mean() < cur_q, 0.7, 0.3)
    losses['value'] = -(
        adv_weight * value.log_prob(cur_q)
        #self.config.slowreg * value.log_prob(sg(slowvalue.mean()))
    )
  
    # Critic
    slowcritic = self.slowcritic({**outs, **acts})
    next_outs = {k: v[:, 1:] for (k, v) in outs.items()}
    next_v = self.value({**next_outs}).mean()
    ret = rew[:, :-1] + con[:, :-1] * discount * next_v
    isvalid = jnp.logical_or(~data['is_last'], data['is_terminal']) 
    #jax.debug.print('CON {x} {y}', x=con[:, 1:].min(), y=con[:, 1:].max())
    ret_padded = jnp.concatenate([ret, 0*ret[:, -1:]], 1)
    losses['critic'] = sg(f32(isvalid))[:, :-1] * -(
        critic.log_prob(sg(ret_padded)) +
        self.config.slowreg * critic.log_prob(sg(slowcritic.mean()))
    )[:, :-1]

    # Actor
    #roffset, rscale = self.retnorm(ret, update)
    adv = (critic.mean() - value.mean())
    exp_a = jnp.minimum(jnp.exp(3.0 * adv), 100.)
    actor = self.actor(outs)
    logpi = sum([v.log_prob(sg(acts[k])) for k, v in actor.items()])
    losses['actor'] = -(
        sg(exp_a) * logpi
    )

    val = value.mean()
    ents = {k: v.entropy() for k, v in actor.items()}
    # Metrics
    metrics.update({f'{k}_loss': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics.update(jaxutils.tensorstats(rew, 'rew'))
    metrics.update(jaxutils.tensorstats(val, 'val'))
    metrics.update(jaxutils.tensorstats(ret, 'ret'))
    metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
    metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()
    for k, space in self.act_space.items():
      act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
      metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
      if hasattr(actor[k], 'minent'):
        lo, hi = actor[k].minent, actor[k].maxent
        rand = ((ents[k] - lo) / (hi - lo)).mean(
            range(2, len(ents[k].shape)))
        metrics.update(jaxutils.tensorstats(rand, f'rand/{k}'))
      metrics.update(jaxutils.tensorstats(ents[k], f'ent/{k}'))
    metrics['data_rew/max'] = jnp.abs(data['reward']).max()
    metrics['pred_rew/max'] = jnp.abs(rew).max()
    metrics['data_rew/mean'] = data['reward'].mean()
    metrics['pred_rew/mean'] = rew.mean()
    metrics['data_rew/std'] = data['reward'].std()
    metrics['pred_rew/std'] = rew.std()
    if 'reward' in dists:
      stats = jaxutils.balance_stats(dists['reward'], data['reward'], 0.1)
      metrics.update({f'rewstats/{k}': v for k, v in stats.items()})
    if 'cont' in dists:
      stats = jaxutils.balance_stats(dists['cont'], data['cont'], 0.5)
      metrics.update({f'constats/{k}': v for k, v in stats.items()})
    metrics['activation/embed'] = jnp.abs(embed).mean()
    # metrics['activation/deter'] = jnp.abs(replay_outs['deter']).mean()

    # Combine
    losses = {k: v * self.scales[k] for k, v in losses.items()}
    #jax.debug.print("Losses {x}", x=jax.tree_util.tree_map(jnp.mean, losses))
    loss = jnp.stack([v.mean() for k, v in losses.items()]).sum()
    newact = {k: data[k][:, -1] for k in self.act_space}
    outs = {'replay_outs': outs, 'prevacts': prevacts, 'embed': embed}
    outs.update({f'{k}_loss': v for k, v in losses.items()})
    carry = (newlat, newact)

    #jax.debug.print("AFTERREW {x} {y}", x=data['reward'].min(), y=data['reward'].max())
    return loss, (outs, carry, metrics)

  def report(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing report function', color='yellow')
    if not self.config.report:
      return {}, carry
    metrics = {}
    data = self.preprocess(data)

    # Train metrics
    _, (outs, carry_out, mets) = self.loss(data, carry, update=False)
    metrics.update(mets)

   # Grad norms per loss term
    if self.config.report_gradnorms:
      for key in self.scales:
        try:
          lossfn = lambda data, carry: self.loss(
              data, carry, update=False)[1][0][f'{key}_loss'].mean()
          grad = nj.grad(lossfn, self.modules)(data, carry)[-1]
          metrics[f'gradnorm/{key}'] = optax.global_norm(grad)
        except KeyError:
          print(f'Skipping gradnorm summary for missing loss: {key}')

    return metrics, carry_out

  def preprocess(self, obs):
    spaces = {**self.obs_space, **self.act_space, **self.aux_spaces}
    result = {}
    for key, value in obs.items():
      if key.startswith('log_') or key in ('reset', 'key', 'id'):
        continue
      space = spaces[key]
      if len(space.shape) >= 3 and space.dtype == jnp.uint8:
        value = cast(value) / 255.0
      result[key] = value
    result['cont'] = 1.0 - f32(result['is_terminal'])
    return result

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
actor_sample = nets.actor_sample
actor_log_prob = nets.actor_log_prob

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
    dec_space = {
        k: v for k, v in obs_space.items()
        if k not in ('is_first', 'is_last', 'is_terminal', 'reward') and
        not k.startswith('log_') and re.match(config.dec.spaces, k)}
    embodied.print('Encoder:', {k: v.shape for k, v in enc_space.items()})
    embodied.print('Decoder:', {k: v.shape for k, v in dec_space.items()})

    # World Model
    self.enc = {
        'simple': bind(nets.SimpleEncoder, **config.enc.simple),
    }[config.enc.typ](enc_space, name='enc')
    self.dec = {
        'simple': bind(nets.SimpleDecoder, **config.dec.simple),
    }[config.dec.typ](dec_space, name='dec')
    self.dyn = {
        'rssm': bind(nets.RSSM, **config.dyn.rssm),
    }[config.dyn.typ](name='dyn')
    self.rew = nets.MLP((), **config.rewhead, name='rew')
    self.con = nets.MLP((), **config.conhead, name='con')

    if self.config.ac_grads == 'none':
        self.sg = sg
    else:
        self.sg = (lambda x: x)

    # Actor
    self.actor_dists = {k: config.actor_dist_disc if v.discrete else config.actor_dist_cont for k, v in self.act_space.items()}
    kwargs = {}
    kwargs['shape'] = {
        k: (*s.shape, s.classes) if s.discrete else s.shape
        for k, s in self.act_space.items()}
    kwargs['dist'] = self.actor_dists
    self.actor = nets.MLP(**kwargs, **config.actor, name='actor')
    #self.fakeactor = nets.MLP(**kwargs, **config.actor, name='fakeactor', dtype='float32')
    self.retnorm = jaxutils.Moments(**config.retnorm, name='retnorm')
    self.valnorm = jaxutils.Moments(**config.valnorm, name='valnorm')
    self.advnorm = jaxutils.Moments(**config.advnorm, name='advnorm')

    # Critic
    self.value = nets.MLP((), name='value', **self.config.value)
    self.slowvalue = nets.MLP(
            (), name='slowvalue', **self.config.value, dtype='float32')
    self.critic = nets.MLP((), name='critic', **self.config.critic)
    self.slowcritic = nets.MLP(
        (), name='slowcritic', **self.config.critic, dtype='float32')
    self.updaterQ = jaxutils.SlowUpdater(
        self.critic, self.slowcritic,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='updaterQ')

    self.updaterV = jaxutils.SlowUpdater(
        self.value, self.slowvalue,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='updaterV')

    #self.updaterP = jaxutils.SlowUpdater(
    #    self.actor, self.fakeactor,
    #    1.0,
    #    1,
    #    name='updaterP')

    # Optimizer
    kw = dict(config.opt)
    lr = kw.pop('lr')
    if config.separate_lrs:
      lr = {f'agent/{k}': v for k, v in config.lrs.items()}
    self.opt = jaxutils.Optimizer(lr, **kw, name='opt')
    self.modules = [
        self.enc, self.dyn, self.dec, self.rew, self.con,
        self.actor, self.critic, self.value
    ]
    self.ac_modules = [
        self.actor, self.critic, self.value
    ]
    scales = self.config.loss_scales.copy()
    cnn = scales.pop('dec_cnn')
    mlp = scales.pop('dec_mlp')
    scales.update({k: cnn for k in self.dec.imgkeys})
    scales.update({k: mlp for k in self.dec.veckeys})
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
    return (self.dyn.initial(batch_size), prevact)

  def init_train(self, batch_size):
    prevact = {
        k: jnp.zeros((batch_size, *v.shape), v.dtype)
        for k, v in self.act_space.items()}
    return (self.dyn.initial(batch_size), prevact)

  def init_report(self, batch_size):
    return self.init_train(batch_size)

  def policy(self, obs, carry, mode='train'):
    self.config.jax.jit and embodied.print(
        'Tracing policy function', color='yellow')
    prevlat, prevact = carry
    obs = self.preprocess(obs)
    embed = self.enc(obs, bdims=1)
    prevact = jaxutils.onehot_dict(prevact, self.act_space)
    lat, out = self.dyn.observe(
        prevlat, prevact, embed, obs['is_first'], bdims=1)
    actor = self.actor(out, bdims=1)
    if mode == 'eval':
        act = {}
        for k in actor:
            if self.actor_dists[k] == 'tanh_normal':
                act[k] = jnp.tanh(actor[k].mode())
            else:
                act[k] = actor[k].mode()
    else:
        act = sample(actor)

    outs = {}
    if self.config.replay_context:
      outs.update({k: out[k] for k in self.aux_spaces if k != 'stepid'})
      outs['stoch'] = jnp.argmax(outs['stoch'], -1).astype(jnp.int32)

    outs['finite'] = {
        '/'.join(x.key for x in k): (
            jnp.isfinite(v).all(range(1, v.ndim)),
            v.min(range(1, v.ndim)),
            v.max(range(1, v.ndim)))
        for k, v in jax.tree_util.tree_flatten_with_path(dict(
            obs=obs, prevlat=prevlat, prevact=prevact,
            embed=embed, act=act, out=out, lat=lat,
        ))[0]}

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

    if self.config.replay_context:
      K = self.config.replay_context
      data = data.copy()
      context = {
          k: data.pop(k)[:, :K] for k in self.aux_spaces if k != 'stepid'}
      context['stoch'] = f32(jax.nn.one_hot(
          context['stoch'], self.config.dyn.rssm.classes))
      prevlat = self.dyn.outs_to_carry(context)
      prevact = {k: data[k][:, K - 1] for k in self.act_space}
      carry = prevlat, prevact
      data = {k: v[:, K:] for k, v in data.items()}
      stepid = stepid[:, K:]

    if self.config.reset_context:
      keep = jax.random.uniform(
          nj.seed(), data['is_first'][:, :1].shape) > self.config.reset_context
      data['is_first'] = jnp.concatenate([
          data['is_first'][:, :1] & keep, data['is_first'][:, 1:]], 1)

    if self.config.run.model_checkpoint:
        mets, outs = self.opt(
            [self.ac_modules],
            [self.loss],
            data, carry, has_aux=True,
        )
    else:
        mets, outs = self.opt(
            [self.modules],
            [self.loss],
            data, carry, has_aux=True
        )
    out, carry, metrics = outs['outs'], outs['carry'], outs['metrics']
    #jax.debug.print("METS {x}", x=jax.tree_util.tree_map(jnp.mean, mets))
    metrics.update(mets)
    self.updaterQ()
    #self.updaterV()
    #self.updaterP()
    outs = {}

    if self.config.replay_context:
      outs['replay'] = {'stepid': stepid}
      outs['replay'].update({
          k: out['replay_outs'][k] for k in self.aux_spaces if k != 'stepid'})
      outs['replay']['stoch'] = jnp.argmax(
          outs['replay']['stoch'], -1).astype(jnp.int32)

    if self.config.replay.fracs.priority > 0:
      bs = data['is_first'].shape
      if self.config.replay.priosignal == 'td':
        priority = out['critic_loss'][:, 0].reshape(bs)
      elif self.config.replay.priosignal == 'model':
        terms = [out[f'{k}_loss'] for k in (
            'rep', 'dyn', *self.dec.veckeys, *self.dec.imgkeys)]
        priority = jnp.stack(terms, 0).sum(0)
      elif self.config.replay.priosignal == 'all':
        terms = [out[f'{k}_loss'] for k in (
            'rep', 'dyn', *self.dec.veckeys, *self.dec.imgkeys)]
        terms.append(out['actor_loss'][:, 0].reshape(bs))
        terms.append(out['critic_loss'][:, 0].reshape(bs))
        priority = jnp.stack(terms, 0).sum(0)
      else:
        raise NotImplementedError(self.config.replay.priosignal)
      assert stepid.shape[:2] == priority.shape == bs
      outs['replay'] = {'stepid': stepid, 'priority': priority}

    return outs, carry, metrics

  def loss(self, data, carry, update=True):
    metrics = {}
    prevlat, prevact = carry

    # Replay rollout
    prevacts = {
        k: jnp.concatenate([prevact[k][:, None], data[k][:, :-1]], 1)
        for k in self.act_space}
    prevacts = jaxutils.onehot_dict(prevacts, self.act_space)
    embed = self.enc(data)
    newlat, outs = self.dyn.observe(prevlat, prevacts, embed, data['is_first'])
    dists = dict(
        **self.dec(outs),
        reward=self.rew(outs, training=True),
        cont=self.con(outs, training=True)
    )
    losses = {k: -v.log_prob(f32(data[k])) for k, v in dists.items()}
    if self.config.contdisc:
      del losses['cont']
      softlabel = data['cont'] * (1 - 1 / self.config.horizon)
      losses['cont'] = -dists['cont'].log_prob(softlabel)
    #weight = f32(~data['is_first'])
    losses['cont'] = losses['cont']# * weight
    losses['reward'] = losses['reward']# * weight

    dynlosses, mets = self.dyn.loss(outs, **self.config.rssm_loss)
    losses.update(dynlosses)
    metrics.update(mets)
    replay_outs = {k: v for (k, v) in outs.items()}

    # Imagination rollout
    def imgstep(carry, _):
      lat, act = carry
      lat, out = self.dyn.imagine(lat, act, bdims=1)
      dist = self.actor(sg(out), bdims=1)
      act, logpi, logpi_sg = actor_sample(dist, self.actor_dists)
      return (lat, act), (self.sg(out), self.sg(act), logpi, logpi_sg)

    if self.config.imag_start == 'all':
      B, T = data['is_first'].shape
      startlat = self.dyn.outs_to_carry(treemap(
          lambda x: x.reshape((B * T, 1, *x.shape[2:])), replay_outs))
      startout = treemap(
          lambda x: x.reshape((B * T, *x.shape[2:])),
          replay_outs)
    elif self.config.imag_start == 'last':
      startlat = newlat
      startout, startrew, startcon = treemap(
          lambda x: x[:, -1], (replay_outs, rew, con))
    
    if self.config.imag_repeat > 1:
      N = self.config.imag_repeat
      startlat, startout = treemap(
          lambda x: x.repeat(N, 0), (startlat, startout))
    startact, startlogpi, startlogpi_sg = actor_sample(self.actor(startout, bdims=1), self.actor_dists)
    _, (outs, acts, logpis, logpis_sg) = jaxutils.scan(
        imgstep, (startlat, startact),
        jnp.arange(self.config.imag_length), self.config.imag_unroll)
    outs, acts, logpis, logpis_sg = treemap(lambda x: x.swapaxes(0, 1), (outs, acts, logpis, logpis_sg))
    outs, acts, logpis, logpis_sg = treemap(
        lambda first, seq: jnp.concatenate([first, seq], 1),
        treemap(lambda x: x[:, None], (startout, startact, startlogpi, startlogpi_sg)), (outs, acts, logpis, logpis_sg)
    )

    # Annotate
    rew = self.rew(outs).mean()[:, 1:]
    con = self.con(outs).mean()[:, 1:]
    con_padded = jnp.concatenate([jnp.ones_like(con[:, -1:]), con], 1)
    
    # Return
    critic = self.critic({**outs, **sg(acts)})
    slowcritic = self.slowcritic({**outs, **sg(acts)})
    value = self.value({**outs})
    voffset, vscale = self.valnorm.stats()
    val = critic.mean() * vscale + voffset
    slowval = slowcritic.mean() * vscale + voffset
    tarval = slowval if self.config.slowtar else val
    discount = 1 if self.config.contdisc else 1 - 1 / self.config.horizon
    weight = jnp.cumprod(discount * con_padded, 1) / discount
    
    '''
    masks = [jnp.ones_like(con[:, -1])]
    for t in range(self.config.imag_length):
      masks.append(masks[t] * con[:, t])
    mask = jnp.stack(masks, axis=1)
    rets = [tarval[:, -1]];
    lamb = self.config.return_lambda; lamb_weight = 1.0
    for t in reversed(range(self.config.imag_length)):
      q_cur = tarval[:, t]
      q_next = (mask[:, t, None] * rew[:, t, None] + mask[:, t+1, None] * discount * rets[-1])
      next_value = (q_cur + lamb * lamb_weight * q_next) / (1 + lamb * lamb_weight)
      rets.append(next_value)
      lamb_weight = 1.0 + lamb * lamb_weight
    ret = jnp.stack(list(reversed(rets)), 1)
    '''
    
    rets = [tarval[:, -1]]
    disc = con * discount
    lam = self.config.return_lambda
    interm = rew[:, :, None] + (1 - lam) * disc[:, :, None] * tarval[:, 1:]
    for t in reversed(range(disc.shape[1])):
      rets.append(interm[:, t] + disc[:, t, None] * lam * rets[-1])
    ret = jnp.stack(list(reversed(rets)), 1)

    expectile = self.config.expectile
    actor = self.actor(outs)
    #baseline = self.critic({**outs, **sg(actor.mode())}).mean() * vscale + voffset
    baseline = value.mean() * vscale + voffset
    roffset, rscale = self.retnorm(ret, update) 
    if ret.shape[-1] == 1:
        Aret = ret.mean(axis=-1)
        Atarval = tarval.mean(axis=-1)
    else:
        beta = 4.0
        Aret = ret.mean(axis=-1) - beta * ret.std(axis=-1)
        Atarval = tarval.mean(axis=-1) - beta * tarval.std(axis=-1)
    adv = (Aret - baseline) / rscale

    #### LEQ ####
    adv_weight = jnp.where(Aret > Atarval, expectile, 1 - expectile)
    adv = adv_weight * adv
    #############
    logpi = sum(logpis_sg.values())
    ents = {k: -logpis[k] if self.actor_dists[k] == 'tanh_normal' else v.entropy() for k, v in actor.items()}
    actor_loss = sg(weight) * -(
        logpi * sg(adv)
        #+ 0.2 * sum(ents.values())
        + self.config.actent * sum(ents.values())
    )
    losses['actor'] = actor_loss

    # Critic
    voffset, vscale = self.valnorm(ret, update)
    ret_normed = (ret - voffset) / vscale
    adv_weight = jnp.where(tarval < ret, expectile, 1-expectile)
    losses['critic'] = sg(weight)[:, :-1, None] * -(
        adv_weight * critic.log_prob(sg(ret_normed)) +
        self.config.slowreg * critic.log_prob(sg(slowcritic.mean()))
    )[:, :-1]

    # Value
    losses['value'] = sg(weight) * -(
        value.log_prob(sg(Atarval))
    )

    if self.config.replay_critic_loss:
      replay_acts = {'action': data['action']}
      replay_acts = jaxutils.onehot_dict(replay_acts, self.act_space)
      replay_critic = self.critic({**replay_outs, **replay_acts})
      replay_slowcritic = self.slowcritic({**replay_outs, **replay_acts})
      replay_value = self.value({**replay_outs})

      replay_next_outs = {k: v[:, 1:] for (k, v) in replay_outs.items()}
      replay_next_actions = sample(self.actor(replay_next_outs))
      replay_next_v = self.critic({**replay_next_outs, **replay_next_actions}).mean()
      replay_con = f32(~data['is_terminal'] * (1 - 1 / self.config.horizon))
      replay_ret = data['reward'][:, 1:, None] + replay_con[:, 1:, None] * replay_next_v
      isvalid = f32(~data['is_last'])[:, :-1, None]

      '''
      replay_rets = [replay_critic[:, -1]]
      replay_con = f32(~data['is_terminal'][:, 1:] * (1 - 1 / self.config.horizon))
      replay_rew = f32(~data['is_terminal'][:, 1:] * (1 - 1 / self.config.horizon))
      interm = replay_rew[:, :, None] + (1 - lam) * replay_con[:, :, None] * replay_next_v[:, 1:]
      for t in reversed(range(disc.shape[1])):
        rets.append(interm[:, t] + disc[:, t, None] * lam * rets[-1])
      replay_ret = jnp.stack(list(reversed(replay_rets)), 1)
      '''

      voffset, vscale = self.valnorm(replay_ret, update)
      ret_normed = (replay_ret - voffset) / vscale
      ret_padded = jnp.concatenate([ret_normed, 0 * ret_normed[:, -1:]], 1)
      losses['replay_critic'] = isvalid * -(
          replay_critic.log_prob(sg(ret_padded)) +
          self.config.slowreg * replay_critic.log_prob(sg(replay_slowcritic.mean()))
      )[:, :-1]

      replay_actor = self.actor(replay_outs)
      metrics.update(jaxutils.tensorstats(replay_critic.mean(), 'replay_val'))

      replay_adv = (replay_critic.mode().mean(axis=-1) - replay_value.mode()) / rscale
      exp_adv = jnp.clip(repaly_adv, 0., 100.)
      #exp_adv = jnp.clip(jnp.exp(jnp.clip(replay_adv, -20., 5.)) - 1., 0., 100.)
      bc_loss = -sum(actor_log_prob(replay_actor, self.actor_dists, replay_acts).values())

      metrics.update(jaxutils.tensorstats(bc_loss, 'bc_loss'))
      awr_loss = sg(exp_adv) * bc_loss
      losses['actor'] = actor_loss.mean() + awr_loss.mean()
      metrics.update(jaxutils.tensorstats(replay_adv, 'replay_adv'))

    ret = ret[:, :-1]
    # Metrics
    metrics.update({f'{k}_loss': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics.update(jaxutils.tensorstats(adv_weight, 'adv_weight'))
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics.update(jaxutils.tensorstats(rew, 'rew'))
    metrics.update(jaxutils.tensorstats(weight, 'weight'))
    metrics.update(jaxutils.tensorstats(val, 'val'))
    metrics.update(jaxutils.tensorstats(ret, 'ret'))
    metrics.update(jaxutils.tensorstats(ret.std(axis=-1), 'Dret'))
    metrics.update(jaxutils.tensorstats(
        (ret - roffset) / rscale, 'ret_normed'))
    metrics.update({'rscale': rscale})
    if self.config.replay_critic_loss:
      metrics.update(jaxutils.tensorstats(replay_ret, 'replay_ret'))

    metrics.update(jaxutils.tensorstats(f32(data['action']), f'raw_action'))
    metrics.update(jaxutils.tensorstats(dists['reward'].mode() - data['reward'], 'rew_td_error'))
    metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
    metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()
    for k, space in self.act_space.items():
      act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
      #jax.debug.print(k + 'ACT {x} {y}', x=act.min(), y=act.max())
      metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
      if hasattr(actor[k], 'minent'):
        lo, hi = actor[k].minent, actor[k].maxent
        rand = ((ents[k] - lo) / (hi - lo)).mean(
            range(2, len(ents[k].shape)))
        metrics.update(jaxutils.tensorstats(rand, f'rand/{k}'))
      metrics.update(jaxutils.tensorstats(ents[k], f'ent/{k}'))
    metrics['data_rew/min'] = data['reward'].min()
    metrics['pred_rew/min'] = rew.min()
    metrics['data_rew/max'] = data['reward'].max()
    metrics['pred_rew/max'] = rew.max()
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
    outs = {'replay_outs': replay_outs, 'prevacts': prevacts, 'embed': embed}
    outs.update({f'{k}_loss': v for k, v in losses.items()})
    carry = (newlat, newact)

    #jax.debug.print("AFTERREW {x} {y}", x=data['reward'].min(), y=data['reward'].max())
    return loss, {'outs':outs, 'carry':carry, 'metrics':metrics}

  def report(self, data, carry):
    self.config.jax.jit and embodied.print(
        'Tracing report function', color='yellow')
    if not self.config.report:
      return {}, carry
    metrics = {}
    data = self.preprocess(data)

    # Train metrics
    _, outs = self.loss(data, carry, update=False)
    outs, carry_out, mets = outs['outs'], outs['carry'], outs['metrics']
    metrics.update(mets)

    # Open loop predictions
    B, T = data['is_first'].shape
    num_obs = min(self.config.report_openl_context, T // 2)
    # Rerun observe to get the correct intermediate state, because
    # outs_to_carry doesn't work with num_obs<context.
    img_start, rec_outs = self.dyn.observe(
        carry[0],
        {k: v[:, :num_obs] for k, v in outs['prevacts'].items()},
        outs['embed'][:, :num_obs],
        data['is_first'][:, :num_obs])
    img_acts = {k: v[:, num_obs:] for k, v in outs['prevacts'].items()}
    img_outs = self.dyn.imagine(img_start, img_acts)[1]
    rec = dict(
        **self.dec(rec_outs), reward=self.rew(rec_outs),
        cont=self.con(rec_outs))
    img = dict(
        **self.dec(img_outs), reward=self.rew(img_outs),
        cont=self.con(img_outs))

    # Prediction losses
    data_img = {k: v[:, num_obs:] for k, v in data.items()}
    losses = {k: -v.log_prob(data_img[k].astype(f32)) for k, v in img.items()}
    metrics.update({f'openl_{k}_loss': v.mean() for k, v in losses.items()})
    stats = jaxutils.balance_stats(img['reward'], data_img['reward'], 0.1)
    metrics.update({f'openl_reward_{k}': v for k, v in stats.items()})
    stats = jaxutils.balance_stats(img['cont'], data_img['cont'], 0.5)
    metrics.update({f'openl_cont_{k}': v for k, v in stats.items()})

    # Video predictions
    for key in self.dec.imgkeys:
      true = f32(data[key][:6])
      pred = jnp.concatenate([rec[key].mode()[:6], img[key].mode()[:6]], 1)
      error = (pred - true + 1) / 2
      video = jnp.concatenate([true, pred, error], 2)
      metrics[f'openloop/{key}'] = jaxutils.video_grid(video)
    for key in self.dec.veckeys:
      if key != 'vector': continue
      true = f32(data[key][:6])
      pred = jnp.concatenate([rec[key].mode()[:6], img[key].mode()[:6]], 1)
      metrics[f'openloop/{key}'] = jnp.concatenate([true, pred], axis=0)
      jax.debug.print('VECTOR {x}', x=metrics['openloop/vector'].shape)

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

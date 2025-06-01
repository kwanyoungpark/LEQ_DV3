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

def gaussian_shape(sigma, size):
  x = jnp.arange(0, size) - jnp.floor((size-1) / 2)
  exponent = jnp.exp(-(x ** 2) / (2 * sigma ** 2))
  exponent = exponent * exponent[:,np.newaxis]
  return exponent / jnp.sum(exponent)

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
    self.slowactor = nets.MLP(**kwargs, **config.actor, name='slowactor')
    #self.fakeactor = nets.MLP(**kwargs, **config.actor, name='fakeactor', dtype='float32')
    self.retnorm = jaxutils.Moments(**config.retnorm, name='retnorm')
    #self.valnorm = jaxutils.Moments(**config.valnorm, name='valnorm')
    #self.advnorm = jaxutils.Moments(**config.advnorm, name='advnorm')

    # Critic
    self.value = nets.MLP((), name='value', **self.config.value)
    self.slowvalue = nets.MLP(
            (), name='slowvalue', **self.config.value, dtype='float32')

    self.updaterP = jaxutils.SlowUpdater(
        self.actor, self.slowactor,
        self.config.slow_critic_fraction,
        self.config.slow_critic_update,
        name='updaterP')

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
        self.actor, self.value
    ]
    self.ac_modules = [
        self.actor, self.value
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
    #if mode == 'eval':
    if False:
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
    #self.updaterQ()
    self.updaterV()
    self.updaterP()
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
    #for k in data:
    #    jax.debug.print(k + ': {x} {y}', x=data[k].min(), y=data[k].max())

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
    weight = f32(~data['is_first'])
    losses['cont'] = losses['cont']# * weight
    losses['reward'] = losses['reward']# * weight

    dynlosses, mets = self.dyn.loss(outs, **self.config.rssm_loss)
    losses.update(dynlosses)
    metrics.update(mets)
    replay_outs = {k: v for (k, v) in outs.items()}
    
    # Value loss
    value = self.value({**replay_outs}) 
    slowvalue = self.slowvalue({**replay_outs})
    val = value.mean()
    slowval = slowvalue.mean()  
    tarval = slowval if self.config.slowtar else val
    rets = [tarval[:, -1]]
    live = f32(~data['is_terminal'])[:, 1:] * (1 - 1 / self.config.horizon)
    cont = f32(~data['is_last'])[:, 1:] * self.config.return_lambda_replay
    interm = data['reward'][:, 1:] + (1 - cont) * live * val[:, 1:]
    for t in reversed(range(cont.shape[1])):
      rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    ret = jnp.stack(list(reversed(rets)), 1)

    losses['critic'] = sg(f32(~data['is_last']))[:, :-1] * -(
        value.log_prob(sg(ret)) +
        self.config.slowreg * value.log_prob(sg(slowvalue.mean()))
    )[:, :-1]

    # Actor
    B, T = data['is_first'].shape
    startlat = self.dyn.outs_to_carry(treemap(
      lambda x: x.reshape((B * T, 1, *x.shape[2:])), replay_outs)
    )
    startout = treemap(
      lambda x: x.reshape((B * T, *x.shape[2:])),
      replay_outs
    )

    N = 4
    outs, lats = treemap(
      lambda x: jnp.repeat(x[None], N, axis=0).reshape((N*x.shape[0], *x.shape[1:])),
      (startout, startlat)
    )
    acts = sample(self.slowactor(outs, bdims=1))
    #acts = jaxutils.onehot_dict(acts, self.act_space)

    print("LATS", treemap(jnp.shape, lats))
    print("ACTS", treemap(jnp.shape, acts))
    print("OUTS", treemap(jnp.shape, outs))

    _, next_outs = self.dyn.imagine(lats, acts, bdims=1)
    outs, lats, next_outs, acts = treemap(
      lambda x: x.reshape((N, x.shape[0] // N, *x.shape[1:])),
      (outs, lats, next_outs, acts)
    )
    rew = self.rew(next_outs).mean()
    con = self.con(next_outs).mean()
    #val = self.value(outs).mean()
    #next_val = self.value(next_outs).mean()
    val = self.slowvalue(outs).mean()
    next_val = self.slowvalue(next_outs).mean()

    act_dist = self.actor(outs)
    logpis = sum([v.log_prob(acts[k]) for k, v in act_dist.items()])
    adv = rew + con * next_val - val; exp_adv = jnp.exp(jnp.clip(adv, -5., 5.))
    norm = ((1 + exp_adv).sum(axis=0, keepdims=True) - exp_adv) / N

    #jax.debug.print("WEIGHT {x} {y}", x=sg(exp_adv/norm).min(), y=sg(exp_adv / norm).max())

    ents = {k: -logpis[k] if self.actor_dists[k] == 'tanh_normal' else v.entropy() for k, v in act_dist.items()}
    weight = weight.reshape((B * T,))
    actor_loss = sg(weight) * -(
        jnp.sum(sg(exp_adv / norm) * logpis, axis=0)
        #+ 0.2 * sum(ents.values())
        #+ self.config.actent * sum(ents.values())
    )
    losses['actor'] = actor_loss
   
    # BC loss
    startact = {k: data[k] for k in self.act_space}
    startact = jaxutils.onehot_dict(startact, self.act_space)
    bc_actor = self.actor(replay_outs)
    val_bc = self.value(replay_outs).mean()
    next_val_bc = val_bc[:, 1:]

    #bc_loss = bc_actor.log_prob(startact)
    live = f32(~data['is_terminal'])[:, 1:] * (1 - 1 / self.config.horizon)
    cont = f32(~data['is_last'])[:, 1:] * self.config.return_lambda_replay
    adv_bc = data['reward'][:, 1:] + (1 - cont) * live * val_bc[:, 1:] - val_bc[:, :-1]

    logpis_bc = sum([bc_actor[k].log_prob(startact[k]) for k in bc_actor])
    bc_loss = -logpis_bc[:, :-1] * sg(jnp.where(adv_bc > 0, 1.0, 0.0))

    losses['actor_bc'] = bc_loss

    ret = ret[:, :-1]
    # Metrics
    metrics.update({f'{k}_loss': v.mean() for k, v in losses.items()})
    metrics.update({f'{k}_loss_std': v.std() for k, v in losses.items()})
    metrics.update(jaxutils.tensorstats(adv, 'adv'))
    metrics.update(jaxutils.tensorstats(adv, 'adv_bc'))
    metrics.update(jaxutils.tensorstats(rew, 'rew'))
    metrics.update(jaxutils.tensorstats(weight, 'weight'))
    metrics.update(jaxutils.tensorstats(val, 'val'))
    metrics.update(jaxutils.tensorstats(ret, 'ret'))
    metrics.update(jaxutils.tensorstats(ret.std(axis=-1), 'Dret'))
    #metrics.update(jaxutils.tensorstats((ret - roffset) / rscale, 'ret_normed'))
    #metrics.update({'rscale': rscale})

    metrics.update(jaxutils.tensorstats(f32(data['action']), f'raw_action'))
    metrics.update(jaxutils.tensorstats(dists['reward'].mode() - data['reward'], 'rew_td_error'))
    #metrics['td_error'] = jnp.abs(ret - val[:, :-1]).mean()
    #metrics['ret_rate'] = (jnp.abs(ret) > 1.0).mean()
    for k, space in self.act_space.items():
      act = f32(jnp.argmax(acts[k], -1) if space.discrete else acts[k])
      #jax.debug.print(k + 'ACT {x} {y}', x=act.min(), y=act.max())
      metrics.update(jaxutils.tensorstats(f32(act), f'act/{k}'))
      if hasattr(act_dist[k], 'minent'):
        lo, hi = act_dist[k].minent, act_dist[k].maxent
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
      pred = (pred >= 0.05).astype(pred.dtype)
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
      if len(space.shape) >= 3:
        if space.dtype == jnp.uint8:
          value = cast(value) / 255.0
        '''
        BB = len(value.shape)
        if BB == 4: value = value[None]
        print(value.shape)
        B, T, H, W, C = value.shape 
        value = value.transpose((0, 1, 4, 2, 3))
        value = value.reshape((B*T*C, 1, H, W))
        blur = gaussian_shape(2, 5).astype(value.dtype)[None, None]
        print(value.shape, blur.shape)
        value = jax.lax.conv(value, blur, (1, 1), 'same')
        value = value.reshape((B, T, C, H, W))
        value = value.transpose((0, 1, 3, 4, 2))
        #value = (value >= 0.05).astype(value.dtype)
        print(value.shape)
        if BB == 4: value = value[0]
        '''
      result[key] = value
    result['cont'] = 1.0 - f32(result['is_terminal'])
    return result

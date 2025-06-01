import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np
import jax
import jax.numpy as jnp
import time

def train_eval(
    make_agent, make_train_replay, make_eval_replay,
    make_train_env, make_eval_env, make_logger, make_score, make_video, args):

  agent = make_agent()
  train_replay = make_train_replay()
  logger = make_logger()
  get_score = make_score()
  get_video = make_video()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  train_episodes = defaultdict(embodied.Agg)
  train_epstats = embodied.Agg()
  policy_fps = embodied.FPS()
  train_fps = embodied.FPS()

  #batch_steps = args.batch_size * (args.batch_length - args.replay_context)
  should_expl = embodied.when.Until(args.expl_until)
  should_log = embodied.when.Clock(args.log_every)
  should_save = embodied.when.Clock(args.save_every)
  should_eval = embodied.when.Clock(args.eval_every)

  @embodied.timer.section('log_step')
  def log_step(tran, worker, mode):
    episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
    epstats = dict(train=train_epstats, eval=eval_epstats)[mode]

    if videos[worker] is not None:
        return

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    rewards[worker].append(tran['reward'])

    if tran['is_first']:
      episode.reset()

    #if worker < args.log_video_streams:
    if True:
      for key in args.log_keys_video:
        if key in tran:
          image = get_video(tran[key])
          episode.add(f'policy_{key}', image, agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()
      score = get_score(result['score'])
      length = result['length']
      logger.add({
          'score': score,
          'length': length,
      }, prefix='episode')

      rew = result.pop('rewards')
      #if len(rew) > 1:
      #  result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      prev_cnt = cnts[0]; cnts[0] += 1
      print("TERMINAL", worker, prev_cnt)
      if prev_cnt < args.num_envs_eval:
          scores.append(score)
          for key in args.log_keys_video:
            if key in tran:
              videos[worker] = result[f'policy_{key}']
          #epstats.add(result)

  dataset_train = agent.dataset(
      bind(train_replay.dataset, args.batch_size, args.batch_length))
  dataset_report = agent.dataset(
      bind(train_replay.dataset, args.batch_size, args.batch_length_eval))
  fns = [bind(make_eval_env, i) for i in range(args.num_envs_eval)]
  carry = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def train_step():
      with embodied.timer.section('dataset_next'):
        batch = next(dataset_train)
      #print(jax.tree_util.tree_map(jnp.shape, batch))
      outs, carry[0], mets = agent.train(batch, carry[0])
      train_fps.step(1.0)
      if 'replay' in outs:
        train_replay.update(outs['replay'])
      agg.add(mets, prefix='train')

  checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
  checkpoint.step = step
  checkpoint.agent = agent
  checkpoint.train_replay = train_replay
  #checkpoint.eval_replay = eval_replay
  if args.from_checkpoint:
    checkpoint.load(args.from_checkpoint)
  if args.model_checkpoint:
    checkpoint.load_model(args.model_checkpoint)
  checkpoint.load_or_save()
  should_save(step)  # Register that we just saved.

  print('Start training loop')
  train_policy = lambda *args: agent.policy(*args, mode='train')
  eval_policy = lambda *args: agent.policy(*args, mode='eval')

  def vector2img(x):
    pris = np.split(x, [6], axis=0)
    true, pred = pris
    print(true.shape, pred.shape)
    B, T, _ = true.shape
    true_img = []
    pred_img = []
    for i in range(B):
        true_img.append(np.stack([get_video(_true) for _true in true[i]], axis=0))
        pred_img.append(np.stack([get_video(_pred) for _pred in pred[i]], axis=0))
    true_img = np.concatenate(true_img, axis=2)
    pred_img = np.concatenate(pred_img, axis=2)
    error_img = (pred_img - true_img + 1) // 2
    print(true_img.shape)
    full_img = np.concatenate([true_img, pred_img, error_img], axis=1).astype(np.uint8)
    return full_img

  #for _ in range(10):
      #step.increment()
      #train_step()
      #policy_fps.step()

  init_value = step.value
  while step < args.steps:
    #if should_eval(step):
    if (step.value - init_value) % args.eval_every == 0:
      eval_episodes = defaultdict(embodied.Agg)
      eval_epstats = embodied.Agg()
      eval_replay = make_eval_replay()
      eval_driver = embodied.Driver(fns, args.driver_parallel, mode='eval')
      eval_driver.on_step(eval_replay.add)
      eval_driver.on_step(bind(log_step, mode='eval'))
      eval_driver.on_step(lambda tran, _: policy_fps.step())
      dataset_eval = agent.dataset(bind(eval_replay.dataset, args.batch_size, args.batch_length_eval))

      print('Start evaluation')
      #scores = [[] for _ in range(args.num_envs_eval)]
      scores = []
      rewards = [[] for _ in range(args.num_envs_eval)]
      videos = [None for _ in range(args.num_envs_eval)]
      cnts = [0]
      eval_driver.reset(agent.init_policy)
      eval_driver(eval_policy, episodes=args.eval_eps)
      idx = np.argmax([0 if videos[i] is None else videos[i].shape[0] for i in range(args.num_envs_eval)])
      for i in range(args.num_envs_eval):
          print(i, scores[i], np.sum(rewards[i]))
          if videos[i] is None:
              videos[i] = np.zeros((0, *videos[idx].shape[1:]), dtype=np.uint8)

          print(videos[i].shape)

      D = max([videos[i].shape[0] for i in range(args.num_envs_eval)])
      agg_video = []
      for i in range(args.num_envs_eval):
          print(i, videos[i].shape)
          video = np.concatenate([
              videos[i], 
              np.zeros((D-videos[i].shape[0], *videos[i].shape[1:]), dtype=videos[i].dtype)
          ], axis=0)
          agg_video.append(video)
      
      if args.num_envs_eval == 10:
          agg_video = np.stack(agg_video, axis=0)
          agg_video = agg_video.reshape((2, 5, *agg_video.shape[1:]))
          agg_video = np.concatenate(agg_video, axis=2)
          agg_video = np.concatenate(agg_video, axis=2)
      else:
          agg_video = agg_video[0]
      agg_video = agg_video[:5000] # Maximum 5000 frames
      agg_video = agg_video[::4]  # 4x
      rewards = np.concatenate(rewards, axis=0)
      logger.add({
          'scores_min': np.min(scores),
          'scores_max': np.max(scores),
          'scores': np.mean(scores),
          'scores_std': np.std(scores),
          'rewards': np.mean(rewards),
          'videos': agg_video,
      }, prefix='epstats')
      #logger.add(eval_epstats.result(), prefix='epstats')
      if len(train_replay):
        mets, _ = agent.report(next(dataset_report), carry_report)
        if 'openloop/vector' in mets:
          mets['openloop/vector'] = vector2img(mets['openloop/vector'])
        logger.add(mets, prefix='report')
      if len(eval_replay):
        mets, _ = agent.report(next(dataset_eval), carry_report)  
        if 'openloop/vector' in mets:
          mets['openloop/vector'] = vector2img(mets['openloop/vector'])
        logger.add(mets, prefix='eval')
        eval_driver.close()
      print('End evaluation')

    for _ in range(10):
      step.increment()
      train_step()
      policy_fps.step()

    if should_log(step):
      logger.add(agg.result())
      logger.add(train_epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(train_replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.write()

    if should_save(step):
      checkpoint.save()

  logger.close()

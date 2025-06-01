import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np


def eval_only(make_agent, make_env, make_logger, make_score, make_video, args):
  #assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()
  get_score = make_score()
  get_video = make_video()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  should_log = embodied.when.Clock(args.log_every)
  policy_fps = embodied.FPS()

  @embodied.timer.section('log_step')
  def log_step(tran, worker):

    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    #if worker < args.log_video_streams:
    if True:
      for key in args.log_keys_video:
        if key in tran:
          image = get_video(tran[key])
          episode.add(f'policy_{key}', tran[key], agg='stack')
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

  fns = [bind(make_env, i) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel, mode='eval')
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(log_step)

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  while step < args.steps:
    driver(policy, steps=10)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.write()

  logger.close()

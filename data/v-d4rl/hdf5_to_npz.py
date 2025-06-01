import datetime
import io
import pathlib
import os
import glob

import numpy as np
import h5py
import argparse
import cv2
import dreamerv3_uuid as uuid
from tqdm import tqdm

def save_episode(directory, episode, num, chunk_uuid):
    timestamp = (datetime.datetime.now() + datetime.timedelta(0,num)).strftime('%Y%m%dT%H%M%S')
    succ_uuid = uuid.uuid()
    length = len(episode['action'])
    filename = directory / f'{timestamp}-{chunk_uuid}-{succ_uuid}-{length}.npz'

    step_id = np.array([np.frombuffer(bytes(chunk_uuid) + i.to_bytes(4, 'big'), np.uint8) for i in range(length)])
    episode['stepid'] = step_id
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return succ_uuid

P = 1024
def main(task, diff):
    # Include argument parser
    parser = argparse.ArgumentParser(description='Convert hdf5 files to npz.')
    input_dir = os.path.join(os.getcwd(), task, diff, '84px')
    output_dir = os.path.join(os.getcwd(), task, diff, '64px_dv3')

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in glob.glob(os.path.join(output_dir, '*.npz')):
        os.remove(f)

    filenames = sorted(pathlib.Path(input_dir).glob('*.hdf5'))
    for filename in filenames:
        print("Loaded", filename)
        with h5py.File(filename, "r") as f:
            actions = f['action'][:]
            observations = f['observation'][:]
            rewards = f['reward'][:]
            discounts = f['discount'][:]
            N = observations.shape[0]

            # ARS (DreamerV2) -> RSA (DreamerV3)
            actions = np.concatenate([actions[1:], np.zeros_like(actions[:1])], axis=0)
            for i in range(5):
                print(actions[0 + 501 * i], actions[500 + 501 * i], "(1, 0)")
                print(rewards[0 + 501 * i], rewards[500 + 501 * i], "(0, 1)")
 
            is_last = np.zeros(N, dtype=bool); is_last[500::501] = True
            is_first = np.zeros(N, dtype=bool); is_first[0::501] = True
            is_terminal = np.zeros(N, dtype=bool)

            if task == 'walker_walk':
                ep_base = {
                    'orientations': np.zeros((N, 14), dtype=np.float32),
                    'height': np.zeros((N, 1), dtype=np.float32),
                    'velocity': np.zeros((N, 9), dtype=np.float32),
                }
            if task == 'cheetah_run':
                ep_base = {
                    'position': np.zeros((N, 8), dtype=np.float32),
                    'velocity': np.zeros((N, 9), dtype=np.float32),
                }

            obss = []
            for i in range(N):
                if (i+1) % 10000 == 0:
                    print(i+1, N)
                    print(observations[i].shape)
                obss.append(cv2.resize(np.transpose(observations[i], (1, 2, 0)), (64, 64), interpolation=cv2.INTER_CUBIC))
            observations = np.stack(obss, axis=0)
            chunk_uuid = uuid.uuid(); cnt = 0
            for idx in tqdm(range(0, N, P)):
                ep = {
                    'image': observations[idx:idx+P],
                    'action': actions[idx:idx+P],
                    'reward': rewards[idx:idx+P],
                    'discount': discounts[idx:idx+P],
                    'is_last': is_last[idx:idx+P],
                    'is_first': is_first[idx:idx+P],
                    'is_terminal': is_terminal[idx:idx+P],
                    **{k: v[idx:idx+P] for (k, v) in ep_base.items()}
                }
                chunk_uuid = save_episode(out_dir, ep, cnt, chunk_uuid)
                print({k: v.shape for (k, v) in ep.items()})
                cnt+=1


if __name__ == '__main__':
    task, diff = 'cheetah_run', 'medium_expert'
    for task in ['cheetah_run', 'walker_walk']:
        for diff in ['random', 'medium', 'medium_replay', 'medium_expert']:
            main(task, diff)

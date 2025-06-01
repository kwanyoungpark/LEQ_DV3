# Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning (Pixel-based environments)

This repository contains the official implementation of [Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning](https://kwanyoungpark.github.io/LEQ/) for pixel-based environments by [Kwanyoung Park](https://kwanyoungpark.github.io/) and [Youngwoon Lee](https://youngwoon.github.io/).

If you use this code for your research, please consider citing our paper:
```
@article{park2024tackling,
  title={Tackling Long-Horizon Tasks with Model-based Offline Reinforcement Learning},
  author={Kwanyoung Park and Youngwoon Lee},
  journal={arXiv Preprint arxiv:2407.00699},
  year={2024}
}
```

## How to run the code

### Install dependencies

```bash
conda create -n LEQ_DV3 python=3.9
conda activate LEQ_DV3

pip install -e .
```

### Download V-D4RL dataset

Please download the V-D4RL dataset from the [original repository](https://github.com/conglu1997/v-d4rl) in `data/v-d4rl/` folder.
After downloading the dataset, execute the conversion script:
```bash
cd data/v-d4rl
python hdf5_to_npz.py
```

### Pretrain world model

For training the world model, please use the script as follows (replace the value of {SEED}):

```bash
./eval/vd4rl/test_whole.sh dmc_vision dmc_walker_walk-medium_expert model 0 0 {SEED}
```

The model will be stored in `logdir/model/dmc_walker_walk-medium_expert/0/{SEED}`.

After training the model, please move the checkpoint to `pretrained_model/12m/model/dmc_walker_walk-medium_expert/0/{SEED}/checkpoint.ckpt`

### Run training

#### LEQ

```bash
./eval/vd4rl/test.sh dmc_vision dmc_walker_walk-medium_expert leq 0 0 {SEED}
```

## References

* The implementation is based on [DreamerV3 codebase](https://github.com/danijar/dreamerv3).

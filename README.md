# Revisiting Discrete Soft Actor-Critic
This repository is the implementation of our optimized soft actor-critic algorithm for discrete action spaces, and it is based on the open-source [tianshou](https://github.com/thu-ml/tianshou) codebase.

## Requirements
- python: 3.6+
- gym>=0.23.1
- torch>=1.4.0
- numba>=0.51.0
- tensorboard>=2.5.0
- atari_py
- tqdm

## Doc
```
.
├── README.md
├── requirements.txt
└── src
    ├── examples
    │   └── atari
    │       ├── atari_network.py
    │       ├── atari_sac.py ##main program
    │       ├── atari_wrapper.py
    ├── libs ## modify tianshou code for discrete SAC alternative design 
    │    
    └── tianshou ##tianshou  library code,version 0.4.9

```


## Usage

1. run base discrete SAC for Pong  10m steps
```
cd src
python3 examples/atari/atari_sac.py --task PongNoFrameskip-v4 --epoch 200  --step-per-epoch 50000
```

2. run  discrete SAC with entropy-penalty for Pong  10m steps
```shell
cd src
python3 examples/atari/atari_sac.py --entropy-penalty --task PongNoFrameskip-v4 --epoch 200  --step-per-epoch 50000
```
3.run  discrete SAC with double avg q for Pong  10m steps
```shell
cd src
python3 examples/atari/atari_sac.py --avg-q --clip-q  --task PongNoFrameskip-v4  --epoch 200  --step-per-epoch 50000
```

4. run discrete SAC with both alternative designs for for Pong  10m steps
```shell
cd src
python3 examples/atari/atari_sac.py --avg-q --clip-q --entropy-penalty --task PongNoFrameskip-v4  --epoch 200  --step-per-epoch 50000
```


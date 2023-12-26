# 2Modes-FlappyBird

`2Modes-FlappyBird` is a project that allows people to play FlappyBird by themselves, or by computers through a model based on Deep Q-Learning. You can also train your own model.

## Preface

Please note that the part of code about DQN in this project heavily references [the project from xmfbit](https://github.com/xmfbit/DQN-FlappyBird) currently.

Here are the versions of some of the dependencies I used:

dependencies:
  - numpy=1.26.0
  - python=3.11.5
  - pip:
      - pygame==2.5.2
      - torch==2.1.0+cu121
      - torchaudio==2.1.0+cu121
      - torchvision==0.16.0+cu121

## Example

To play game by yourself, simply go into the root of the project and type:

```shell
python launch.py --mode play --play_mode human
```

To play game with a pretrained model, type:

```shell
python launch.py --mode play --play_mode dqn --model < name of the model you want to play the game with >
```

To train a model from scratch with GPU support, type:

```shell
python launch.py --mode train --cuda
```

There are many optional parameters to help you train a model. To know all the the parameters and what they do, type:
```shell
python launch.py -h
```
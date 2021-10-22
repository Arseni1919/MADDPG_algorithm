# ------------------------------------------- #
# ------------------IMPORTS:----------------- #
# ------------------------------------------- #
import os
import time
import logging
from collections import namedtuple, deque
from termcolor import colored

import gym
import pettingzoo
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import neptune.new as neptune
from neptune.new.types import File

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms
from torch.distributions import Normal
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset

# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import Callback
# from pytorch_lightning import loggers as pl_loggers


# ------------------------------------------- #
# ------------------FOR ENV:----------------- #
# ------------------------------------------- #
from pettingzoo.mpe import simple_spread_v2
MAX_CYCLES = 25
# ENV = simple_spread_v2.env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)
ENV = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=MAX_CYCLES, continuous_actions=True)

NUMBER_OF_GAMES = 10
SAVE_RESULTS = True
# SAVE_RESULTS = False
SAVE_PATH = 'data/actor_net.pt'

# NEPTUNE = True
NEPTUNE = False
# PLOT_LIVE = True
PLOT_LIVE = False
RENDER_WHILE_TRAINING = False

# ------------------------------------------- #
# ------------------FOR ALG:----------------- #
# ------------------------------------------- #

# MAX_LENGTH_OF_A_GAME = 10000
# ENTROPY_BETA = 0.001
# REWARD_STEPS = 4
# CLIP_GRAD = 0.1

MAX_STEPS = MAX_CYCLES * 100  # maximum epoch to execute
M_EPISODES = 20
BATCH_SIZE = 64  # size of the batches
BATCHES_PER_TRAINING_STEP = 3
REPLAY_BUFFER_SIZE = BATCH_SIZE * 10
LR_CRITIC = 3e-4  # learning rate
LR_ACTOR = 3e-3  # learning rate
GAMMA = 0.99  # discount factor
ACT_NOISE = 0.5  # actuator noise
POLYAK = 0.999
VAL_EVERY = 5
TRAIN_EVERY = 5
HIDDEN_SIZE = 256
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done', 'new_state'])


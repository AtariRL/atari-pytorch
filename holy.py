import time

import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from typing import Callable
from cmd_utils import *

def run():
    # There already exists an environment generator that will make and wrap atari environments correctly.
    # We use 16 parallel processes
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=4, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)
    model = A2C('MlpPolicy', env, verbose=0)
    n_timesteps = 25000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")
    
if __name__ == "__main__":
    run()
import time

import gym
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import logger

#Atari
from atari_wrappers import *
from wrappers import *

import monitor as Monitor

import os

from typing import Callable

game = 'BreakoutNoFrameskip-v4'

def custom_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0,
                     allow_early_resets=True):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have
           in subprocesses
    :param seed: (int) the inital seed for RNG
    :param wrapper_kwargs: (dict) the parameters for wrap_deepmind function
    :param start_index: (int) start rank index
    :param allow_early_resets: (bool) allows early reset of the environment
    :return: (Gym Environment) The atari environment
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(
                logger.get_dir(), str(rank)),
                          allow_early_resets=allow_early_resets)
            return wrap_deepmind(env, **wrapper_kwargs)
        
        return _thunk

    return DummyVecEnv([make_env(i + start_index) for i in range(num_env)])


def make_env(seed, rank):
    def _thunk():
        print("Created new env")
        env = gym.make(game)
        env.seed(seed + rank)
        assert "NoFrameskip" in game, "Require environment with no frameskip"
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

        allow_early_resets = False
        log_dir = 'logs'
        assert log_dir is not None, "Log directory required for Monitor! (which is required for episodic return reporting)"
        try:
            os.mkdir(log_dir)
        except:
            pass
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=allow_early_resets)

        env = EpisodicLifeEnv(env)
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        # env = PyTorchFrame(env)
        env = ClipRewardEnv(env)
        # env = FrameStack(env, 4)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)
        #env.render()
        return env
    return _thunk

def run():
    num_cpu = 2  # Number of processes to use
    # Create the vectorized environment
    env = custom_atari_env('BreakoutNoFrameskip-v4', num_env=4, seed=42)
    env = SubprocVecEnv([make_env(42, i) for i in range(num_cpu)])  

    model = A2C('MlpPolicy', env, verbose=0)
    # By default, we use a DummyVecEnv as it is usually faster (cf doc)
    
    #vec_env = make_vec_env(env_id, n_envs=num_cpu)

   # model = A2C('MlpPolicy', vec_env, verbose=0)

    n_timesteps = 25000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

if __name__ == "__main__":
    run()
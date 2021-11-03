import time

import gym
import numpy as np

import torch

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from typing import Callable
from cmd_utils import *

import shutil
import logger

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.true_values = [], [], []
    
    def push(self, state, action, true_value):
        self.states.append(state)
        self.actions.append(action)
        self.true_values.append(true_value)
    
    def pop_all(self):
        states = torch.stack(self.states)
        actions = LongTensor(self.actions)
        true_values = FloatTensor(self.true_values).unsqueeze(1)
        
        self.states, self.actions, self.true_values = [], [], []
        
        return states, actions, true_values

class Model(torch.nn.Module):
    def __init__(self, action_space, env):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        feature_size = self.features(
             torch.zeros(1, *env.observation_space.shape)).cuda().view(1, -1).size(1)
        
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(feature_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, action_space),
            torch.nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        value = self.critic(x)
        actions = self.actor(x)
        return value, actions
    
    def get_critic(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.critic(x)
    
    def evaluate_action(self, state, action):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()
        
        return value, log_probs, entropy
    
    def act(self, state):
        value, actor_features = self.forward(state)
        dist = torch.distributions.Categorical(actor_features)
        
        chosen_action = dist.sample()
        return chosen_action.item()

def compute_true_values(states, rewards, dones, model):
    R = []
    rewards = FloatTensor(rewards)
    dones = FloatTensor(dones)
    states = torch.stack(states)
    
    if dones[-1] == True:
        next_value = rewards[-1]
    else:
        next_value = model.get_critic(states[-1].unsqueeze(0))
        
    R.append(next_value)
    for i in reversed(range(0, len(rewards) - 1)):
        if not dones[i]:
            next_value = rewards[i] + next_value * gamma
        else:
            next_value = rewards[i]
        R.append(next_value)
        
    R.reverse()
    
    return FloatTensor(R)

def get_batch(batch_size, model, state, env):
        states, actions, rewards, dones = [], [], [], []
        for _ in range(batch_size):
            action = model.act(state.unsqueeze(0))
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            if done:
                state = FloatTensor(env.reset())
                logger.logkv("Episode Reward", episode_reward)
                episode_reward = 0
            else:
                state = FloatTensor(next_state)
                logger.logkv("Episode Reward", episode_reward)
                
        values = compute_true_values(states, rewards, dones, model).unsqueeze(1)
        return states, actions, values


def reflect(memory, model, optimizer):
    states, actions, true_values = memory.pop_all()

    values, log_probs, entropy = model.evaluate_action(states, actions)

    advantages =  true_values - values
    critic_loss = advantages.pow(2).mean()

    actor_loss = -(log_probs * advantages.detach()).mean()
    total_loss = (critic_coef * critic_loss) + actor_loss - (entropy_coef * entropy)

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
        
    return values.mean().item()   

def run():
    
    # There already exists an environment generator that will make and wrap atari environments correctly.
    # We use 16 parallel processes
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=5, seed=0)
    # Stack 4 frames
    env = VecFrameStack(env, n_stack=4)

    #torch.Size([5, 84, 84, 4]) where 5 is num_env
    state = FloatTensor(env.reset())

    #Breakout is 4, number of actions 
    num_act = env.action_space.n


    learning_rate = 7e-4
    max_frames = 5000000

    model = Model(num_act, env).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)

    memory = Memory()
    episode_reward = 0
    frame_idx = 0

    while frame_idx < max_frames:
        states, actions, true_values = get_batch(32, model, state, env)
        for i, _ in enumerate(states):
            memory.push(
                states[i],
                actions[i],
                true_values[i]
            )
        frame_idx += batch_size
        
    value = reflect(memory, model, optimizer)
    logger.logkv("frame", frame_idx)
    logger.logkv("value", value)
    logger.dumpkvs()
    
if __name__ == "__main__":
    DEBUG = 10
    # Setup logging for the model
    logger.set_level(DEBUG)
    dir = "logs"
    if os.path.exists(dir):
        shutil.rmtree(dir)
    logger.configure(dir=dir)
    if torch.cuda.is_available():
        FloatTensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
    else:
        FloatTensor = torch.FloatTensor
        LongTensor = torch.LongTensor

    gamma = 0.99
    batch_size = 32
    entropy_coef = 0.01
    critic_coef = 0.5
    
    run()
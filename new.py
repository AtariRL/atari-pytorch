import time

import gym
import numpy as np

import torch
from torchsummary import summary    
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from typing import Callable
from cmd_utils import *

import shutil
import logger

def ortho_weights(shape, scale=1.):
    """ PyTorch port of ortho_init from baselines.a2c.utils """
    shape = tuple(shape)

    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError

    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))


def atari_initializer(module):
    """ Parameter initializer for Atari models
    Initializes Linear, Conv2d, and LSTM weights.
    """
    classname = module.__class__.__name__

    if classname == 'Linear':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'Conv2d':
        module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
        module.bias.data.zero_()

    elif classname == 'LSTM':
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'weight_hh' in name:
                param.data = ortho_weights(param.data.size(), scale=1.)
            if 'bias' in name:
                param.data.zero_()


class AtariCNN(nn.Module):
    def __init__(self, num_actions):
        """ Basic convolutional actor-critic network for Atari 2600 games
        Equivalent to the network in the original DQN paper.
        Args:
            num_actions (int): the number of available discrete actions
        """
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, 4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, stride=1),
                                  nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512),
                                nn.ReLU())

        self.pi = nn.Linear(512, num_actions)
        self.v = nn.Linear(512, 1)

        self.num_actions = num_actions

        # parameter initialization
        self.apply(atari_initializer)
        self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
        self.v.weight.data = ortho_weights(self.v.weight.size())

    def forward(self, conv_in):
        """ Module forward pass
        Args:
            conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]
        Returns:
            pi (Variable): action probability logits, shaped [N x self.num_actions]
            v (Variable): value predictions, shaped [N x 1]
        """
        N = conv_in.size()[0]

        conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

        fc_out = self.fc(conv_out)

        pi_out = self.pi(fc_out)
        v_out = self.v(fc_out)

        return v_out,pi_out

    def get_critic(self, state):
        value, _ = self.forward(state)
        return value
    
    def evaluate_action(self, state, action):
        action = torch.FloatTensor(action).cuda()
        value, actor_features = self.forward(state)
        m = nn.Softmax(dim=1)
        actor_features = m(actor_features)
        dist = torch.distributions.Categorical(actor_features)
        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()
        
        return value, log_probs, entropy
    
    def act(self, state):
        _, actor_features = self.forward(state)
        m = nn.Softmax(dim=1)
        actor_features = m(actor_features)
        dist = torch.distributions.Categorical(actor_features)
        chosen_action = dist.sample()
        return chosen_action

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
        true_values = FloatTensor(self.true_values)
        
        self.states, self.actions, self.true_values = [], [], []
        
        return states, actions, true_values

def compute_true_values(states, rewards, dones, model):
    true_values = []
    rewards = FloatTensor(rewards)
    dones = FloatTensor(dones)
    states = torch.stack(states)

    for i in range(len(states)):
        R = []
        
        next_value = model.get_critic(states[i][-1].unsqueeze(0))
            
        R.append(next_value)
        for e in reversed(range(0, len(rewards[i]) - 1)):
            if not dones[i,e]:
                next_value = rewards[i,e] + next_value * gamma
            else:
                next_value = rewards[i,e]
            R.append(next_value)
            
        R.reverse()
        true_values.append(FloatTensor(R).unsqueeze(1))
    
    return true_values

def get_batch(batch_size, model, state, env, episode_reward):
        states, action_list, rewards, dones, values = [], [], [], [], []
        for _ in range(batch_size):

            # Get action probability for each of the envs
            actions = model.act(state)
            actions = actions.cpu().numpy()
            actions = list(actions)
            next_state, reward, done, _ = env.step(actions)
            episode_reward += reward

            value = model.get_critic(state)

            # Appends env x everything lol
            states.append(state)
            action_list.append(actions)
            rewards.append(reward)
            dones.append(done)
            values.append(value)
            
            if True in done:
                state = FloatTensor(env.reset())
                #episode_reward = 0
            else:
                state = FloatTensor(next_state)
                
        values = compute_true_values(states, rewards, dones, model)
        return states, action_list, values, episode_reward


def reflect(states, actions, true_values, model, optimizer):
    for i in range(len(states)):

        values, log_probs, entropy = model.evaluate_action(states[i], actions[i])
        advantages =  true_values[i] - values
        critic_loss = advantages.pow(2).mean()

        actor_loss = -(log_probs * advantages.detach()).mean()
        total_loss = (critic_coef * critic_loss) + actor_loss - (entropy_coef * entropy)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
    return values.mean().item()   


def run():
    n_envs = 20
    # There already exists an environment generator that will make and wrap atari environments correctly.
    # We use 16 parallel processes
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=n_envs, seed=0)

    #torch.Size([5, 84, 84, 4]) where 5 is num_env
    # [5, 84, 84, 4] -> [5, 4, 84, 84]
    state = FloatTensor(env.reset())

    #Breakout is 4, number of actions 
    num_act = env.action_space.n

    learning_rate = 7e-4
    max_frames = 5000000

    model = AtariCNN(num_act).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)

    episode_reward = 0
    frame_idx = 0
    while frame_idx < max_frames:
        states, actions, true_values, episode_reward = get_batch(batch_size, model, state, env, episode_reward)
        frame_idx += batch_size * n_envs
    
        value = reflect(states, actions, true_values, model, optimizer)
        logger.logkv("frame", frame_idx)
        logger.logkv("episode_reward", torch.mean(true_values[0]))
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
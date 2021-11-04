import time

import gym
import numpy as np

import torch
from torchsummary import summary    
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import A2C
#from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

from a2c.multiprocessing_env import SubprocVecEnv, VecPyTorch, VecPyTorchFrameStack
from a2c.wrappers import *

from collections import deque

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
        entropy = dist.entropy()
        
        return value, log_probs, entropy
    
    def act(self, state):
        _, actor_features = self.forward(state)
        m = nn.Softmax(dim=1)
        actor_features = m(actor_features)
        dist = torch.distributions.Categorical(actor_features)
        chosen_action = dist.sample()
        return chosen_action, dist.entropy()

def compute_returns(next_value, rewards, masks, gamma):
    #rewards = torch.from_numpy(rewards)
    #rewards = [torch.from_numpy(item).float().cuda() for item in rewards]
    r = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        r = rewards[step] + gamma * r * masks[step]
        returns.insert(0, r)
    return returns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def make_env(seed, rank):
    def _thunk():
        env = gym.make(game)
        env.seed(seed + rank)
        assert "NoFrameskip" in game, "Require environment with no frameskip"
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)

        allow_early_resets = False
        log_dir = m_log
        assert m_log is not None, "Log directory required for Monitor! (which is required for episodic return reporting)"
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

        return env
    return _thunk
def make_envs():
    envs = [make_env(seed, i) for i in range(n_envs)]
    envs = SubprocVecEnv(envs)
    envs = VecPyTorch(envs, device)
    envs = VecPyTorchFrameStack(envs, 4, device)
    return envs

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    DEBUG = 10
    n_envs = 5
    seed = 42
    game = 'BreakoutNoFrameskip-v4'
    m_log = 'qwe'
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

    n_steps = 5 # forward steps
    n_frames = int(10e6)

    num_updates = n_frames // n_steps // n_envs

    #envs = make_atari_env('BreakoutNoFrameskip-v4', n_envs=n_envs, seed=0)
    envs = make_envs()

    num_act = 4
    model = AtariCNN(num_act).cuda()
    learning_rate = 7e-4
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)

    observation = FloatTensor(envs.reset())
    start = time.time()

    episode_rewards = deque(maxlen=10)
    for update in range(num_updates):

        log_probs = []
        values = []
        rewards = []
        actions = []
        masks = []
        entropies = []

        for step in range(n_steps):
            actions, entropy = model.act(observation)
            value = model.get_critic(observation)
            actions = actions.cpu().numpy()
            actions = list(actions)
            next_observation, reward, done, infos = envs.step(actions)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            _, log_prob, entropy = model.evaluate_action(observation, actions)

            mask = torch.from_numpy(1.0 - done).to(device).float()

            entropies.append(entropy)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            rewards.append(reward.to(device).squeeze())
            masks.append(mask)

            observation = observation = FloatTensor(next_observation)

        next_observation = FloatTensor(next_observation)
        with torch.no_grad():
            next_values = model.get_critic(next_observation)
            returns = compute_returns(next_values.squeeze(), rewards, masks, 0.99)
            returns = torch.cat(returns)

        log_probs = torch.cat(log_probs)
        values = torch.cat(values)
        entropies = torch.cat(entropies)

        advantages = returns - values

        print("LOG PROB")
        print(log_probs)
        print("VALUES")
        print(values)
        print("entropies")
        print(entropies)
        print("advantages")
        print(advantages)
        exit()


        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = 1.0 * actor_loss + \
               0.5 * critic_loss - \
               0.01 * entropy_loss

        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        loss.backward()
        optimizer.step()

        if len(episode_rewards) > 1 and update % 100 == 0:
            end = time.time()
            total_num_steps = (update + 1) * n_envs * n_steps
            print("********************************************************")
            print("update: {0}, total steps: {1}, FPS: {2}".format(update, total_num_steps, int(total_num_steps / (end - start))))
            print("mean/median reward: {:.1f}/{:.1f}".format(np.mean(episode_rewards), np.median(episode_rewards)))
            print("min/max reward: {:.1f}/{:.1f}".format(np.min(episode_rewards), np.max(episode_rewards)))
            print("actor loss: {:.5f}, critic loss: {:.5f}, entropy: {:.5f}".format(actor_loss.item(), critic_loss.item(), entropy_loss.item()))
            print("********************************************************")
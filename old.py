import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from IPython.display import clear_output
from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
import atari_wrappers as wrappers
import logger
import shutil
import os
from torchsummary import summary

max_frames = 5000000
batch_size = 16
learning_rate = 7e-4
gamma = 0.99
entropy_coef = 0.01
critic_coef = 0.5
env_name = 'BreakoutNoFrameskip-v4'
no_of_workers = 16
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


env = wrappers.make_atari(env_name)
env = wrappers.wrap_deepmind(env, scale=True)
env = wrappers.wrap_pytorch(env)


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
        #print("CRITIC")
        value, _ = self.forward(state)
        #m = nn.Softmax(dim=1)
        #actor_features = m(value)
        return value
    
    def evaluate_action(self, state, action):
        #print("EVAL")
        value, actor_features = self.forward(state)
        m = nn.Softmax(dim=1)
        actor_features = m(actor_features)

        dist = torch.distributions.Categorical(actor_features)
        
        log_probs = dist.log_prob(action).view(-1, 1)
        entropy = dist.entropy().mean()
        
        return value, log_probs, entropy
    
    def act(self, state):
        #print("ACTOR")
        _, actor_features = self.forward(state)
        m = nn.Softmax(dim=1)
        actor_features = m(actor_features)
        dist = torch.distributions.Categorical(actor_features)
        #print(actor_features)
        chosen_action = dist.sample()
        return chosen_action.item()

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


def compute_true_values(states, rewards, dones):
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


def reflect(memory):
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


class Worker(object):
    def __init__(self, env_name):
        self.env = wrappers.make_atari(env_name)
        self.env = wrappers.wrap_deepmind(self.env, frame_stack=True, scale=True)

        self.env = wrappers.wrap_pytorch(self.env)
        self.episode_reward = 0
        self.state = FloatTensor(self.env.reset())
        
    def get_batch(self):
        states, actions, rewards, dones = [], [], [], []
        for _ in range(batch_size):
            #print("IN GET BATCH")
            #print(self.state.shape)

            # UPDATE BOTH U FACKING BLISTERING IDIOT
            #print(self.state.unsqueeze(0).shape)

            print("ACTION PROB")
            print(self.state.shape)
            print("ACTION PROB2")
            print(self.state.unsqueeze(0).shape)
            action = model.act(self.state.unsqueeze(0))
            next_state, reward, done, _ = self.env.step(action)
            self.episode_reward += reward

            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            if done:
                self.state = FloatTensor(self.env.reset())
                data['episode_rewards'].append(self.episode_reward)
                logger.logkv("Episode Reward", self.episode_reward)
                self.episode_reward = 0
            else:
                self.state = FloatTensor(next_state)
                logger.logkv("Episode Reward", self.episode_reward)
                
        values = compute_true_values(states, rewards, dones).unsqueeze(1)
        return states, actions, values



model = AtariCNN(env.action_space.n).cuda()

#print(model.parameters())
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)
memory = Memory()
workers = []
for _ in range(no_of_workers):
    workers.append(Worker(env_name))
frame_idx = 0
data = {
    'episode_rewards': [],
    'values': []
}
state = FloatTensor(env.reset())
episode_reward = 0

while frame_idx < max_frames:
    for worker in workers:
        states, actions, true_values = worker.get_batch()
        for i, _ in enumerate(states):
            memory.push(
                states[i],
                actions[i],
                true_values[i]
            )
        frame_idx += batch_size
        
    value = reflect(memory)
    logger.logkv("frame", frame_idx)
    logger.dumpkvs()
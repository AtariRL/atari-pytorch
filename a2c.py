import torch
import gym
import numpy as np
from IPython.display import clear_output
from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt
import atari_wrappers as wrappers

max_frames = 5000000
batch_size = 5
learning_rate = 7e-4
gamma = 0.99
entropy_coef = 0.01
critic_coef = 0.5
env_name = 'PongNoFrameskip-v4'
no_of_workers = 16

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


env = wrappers.make_atari(env_name)
env = wrappers.wrap_deepmind(env, scale=True)
env = wrappers.wrap_pytorch(env)

class Model(torch.nn.Module):
    def __init__(self, action_space):
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
        self.env = wrappers.wrap_deepmind(self.env, scale=True)
        self.env = wrappers.wrap_pytorch(self.env)
        self.episode_reward = 0
        self.state = FloatTensor(self.env.reset())
        
    def get_batch(self):
        states, actions, rewards, dones = [], [], [], []
        for _ in range(batch_size):
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
                self.episode_reward = 0
            else:
                self.state = FloatTensor(next_state)
                
        values = compute_true_values(states, rewards, dones).unsqueeze(1)
        return states, actions, values

model = Model(env.action_space.n).cuda()
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
    if frame_idx % 1000 == 0:
        print(value)

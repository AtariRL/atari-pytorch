import torch
import gym
import numpy as np
from IPython.display import clear_output
from IPython.core.debugger import set_trace
import wrappers.atari_wrappers as wrappers
import wrappers.logger as logger
import shutil
import os

from model import Model
from worker import Worker
from memory import Memory

max_frames = 5000000
batch_size = 5
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

model = Model(env.action_space.n).cuda()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, eps=1e-5)
memory = Memory()
workers = []
for _ in range(no_of_workers):
    workers.append(Worker(env_name))
frame_idx = 0
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

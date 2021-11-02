import wrappers.atari_wrappers as wrappers
from a2c import FloatTensor, batch_size, model, logger, data, compute_true_values

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
                logger.logkv("Episode Reward", self.episode_reward)
                self.episode_reward = 0
            else:
                self.state = FloatTensor(next_state)
                logger.logkv("Episode Reward", self.episode_reward)
                
        values = compute_true_values(states, rewards, dones).unsqueeze(1)
        return states, actions, values
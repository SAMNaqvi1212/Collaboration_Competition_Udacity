import numpy as np
import random
import copy
from collections import namedtuple, deque
from itertools import count
import time

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

_batch_size = 256       # minibatch size
_buffer_size = int(1e5) # replay buffer size
_gamma = 0.99           # discount factor
_lr_actor = 1e-4        # learning rate of the actor 
_lr_critic = 1e-4       # learning rate of the critic
_tau = 3e-1             # soft update interpolation
_noise_decay = 0.999    # OU Noise decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import time

# Define hyperparameters
_buffer_size = int(1e6)  # Replay buffer size
_batch_size = 128         # Mini-batch size for training
_gamma = 0.99             # Discount factor
_tau = 1e-3               # Soft update of target parameters
_noise_decay = 0.995      # Noise decay factor
_lr_actor = 1e-3          # Actor learning rate
_lr_critic = 1e-3         # Critic learning rate

class Agent():
    def __init__(self, state_size, action_size, random_seed, num_agents):
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=_lr_actor)

        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=_lr_critic, weight_decay=0)

        self.noise_decay = _noise_decay
        self.noise = OUNoise(action_size, random_seed)
        self.memory = ReplayBuffer(action_size, _buffer_size, _batch_size, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        if len(self.memory) > _batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, _gamma)

    def act(self, states, add_noise=True, apply_noise_decay=False):
        states = torch.from_numpy(states).float().to(device)
        actions = []
        self.actor_local.eval()
        with torch.no_grad():
            for state in states:
                action = self.actor_local(state).cpu().data.numpy()
                actions.append(action)
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            if apply_noise_decay:
                actions += self.noise_decay * noise
                self.noise_decay *= self.noise_decay
            else:
                actions += noise
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, _tau)
        self.soft_update(self.actor_local, self.actor_target, _tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def train(self, env, n_episodes=3000, checkpoint_file='checkpoint.pt', print_every=10, apply_noise_decay=False):
        scores_deque = deque(maxlen=100)
        scores_all = []
        moving_average = []

        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]

        for i_episode in range(1, n_episodes + 1):
            timestep = time.time()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            scores = np.zeros(self.num_agents)
            self.reset()
            score_average = 0

            for t in count():
                actions = self.act(states, apply_noise_decay=apply_noise_decay)
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                self.step(states, actions, rewards, next_states, dones)

                states = next_states
                scores += rewards

                if np.any(dones):
                    break

            score = np.max(scores)
            scores_deque.append(score)
            score_average = np.mean(scores_deque)
            scores_all.append(score)
            moving_average.append(score_average)

            if i_episode % print_every == 0:
                print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Avg: {:.2f}, Time: {:.2f}' \
                      .format(i_episode, score_average,
                              np.max(scores_all[-print_every:]),
                              np.min(scores_all[-print_every:]),
                              np.mean(scores_all[-print_every:]),
                              time.time() - timestep), end="\n")

            if score_average >= 0.5:
                print('\n\nEnvironment solved in {:d} episodes!\t' \
                      'Moving Average Score: {:.3f}'
                      .format(i_episode, moving_average[-1]))
                self.save(checkpoint_file)
                break

        return scores_all, moving_average

    def save(self, file='checkpoint.pt'):
        checkpoint = {
            'actor_dict': self.actor_local.state_dict(),
            'critic_dict': self.critic_local.state_dict()
        }
        print('\nSaving model ...', end=' ')
        torch.save(checkpoint, file)
        print('done.')
     def load(self, file='checkpoint.pt', map_location='cpu'):
        """Load the trained model."""
        checkpoint = torch.load(file, map_location=map_location)
        self.actor_local.load_state_dict(checkpoint['actor_dict'])
        self.critic_local.load_state_dict(checkpoint['critic_dict'])

    def test(self, env, n_episodes=5):
        """Test the agent."""
        for i in range(n_episodes):
            brain_name = env.brain_names[0]
            brain = env.brains[brain_name]
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            scores = np.zeros(self.num_agents)

            while True:
                actions = self.act(states, add_noise=False)
                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                scores += env_info.rewards
                states = next_states

                if np.any(dones):
                    break

            print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

class OUNoise():
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
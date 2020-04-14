from collections import deque, namedtuple

import numpy as np
import torch


class ExperienceBuffer:
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.actions_memory = deque(maxlen=buffer_size)
        self.rewards_memory = deque(maxlen=buffer_size)
        self.log_probs_memory = deque(maxlen=buffer_size)
        self.dones_memory = deque(maxlen=buffer_size)
        self.states_memory = deque(maxlen=buffer_size)

    def append(self, actions, rewards, log_probs, dones, states):
        self.actions_memory.append(actions)
        self.rewards_memory.append(rewards)
        self.log_probs_memory.append(log_probs)
        self.dones_memory.append(dones)
        self.states_memory.append(states)

    def sample(self):
        if self.buffer_size > len(self.actions_memory):
            return None

        return self.actions_memory, self.rewards_memory, self.log_probs_memory, self.dones_memory, self.states_memory

    def dump_all(self):
        self.actions_memory.clear()
        self.rewards_memory.clear()
        self.log_probs_memory.clear()
        self.dones_memory.clear()
        self.states_memory.clear()

    def processed_sample(self, experiences, gamma=.99, device="cpu"):
        if self.buffer_size > len(self.actions_memory):
            return None
        actions, rewards, log_probs, dones, state_values = experiences

        rewards = torch.Tensor(rewards).transpose(0, 1).contiguous()
        processed_experience = [None] * (self.buffer_size - 1)
        return_ = state_values[-1].detach()
        for i in reversed(range(len(actions) - 1)):
            done_ = torch.FloatTensor(dones[i + 1]).to(device).unsqueeze(1)
            reward_ = torch.FloatTensor(rewards[:, i]).to(device).unsqueeze(1)
            return_ = reward_ + gamma * (1 - done_) * return_
            next_value_ = state_values[i + 1]
            advantage_ = reward_ + gamma * (1 - done_) * next_value_.detach() - state_values[i].detach()
            processed_experience[i] = [log_probs[i], advantage_, state_values[i], return_]
        log_probs, advantages, values, returns = map(
            lambda x: torch.cat(x, dim=0), zip(*processed_experience))

        return log_probs, advantages, values, returns


class ReplayBuffer:
    memory = None

    def __init__(self, action_size, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.action_size = action_size

        self.memory = deque(maxlen=int(buffer_size))
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "prob", "done"])

    def append(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, device="cpu"):
        """Randomly sample a batch of experiences from memory."""
        if self.batch_size > len(self.memory):
            return None

        starting_point = int(np.random.uniform(0, len(self.memory) - self.batch_size - 1))

        experiences = [self.memory[i] for i in range(starting_point, starting_point + self.batch_size)]

        # experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.stack([e.state for e in experiences if e is not None])
        actions = torch.stack([e.action for e in experiences if e is not None])
        # [e.reward for e in experiences if e is not None]
        rewards = torch.Tensor([e.reward for e in experiences if e is not None])
        probs = torch.stack([e.prob for e in experiences if e is not None])
        dones = torch.Tensor([e.done for e in experiences if e is not None])

        return states, actions, rewards, probs, dones


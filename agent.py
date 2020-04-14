import numpy as np
import torch
from torch.optim import Adam

# from layers import NoiseLayer
from model import TwoHeadModel
from rl_utils import ExperienceBuffer


class Agent:
    def __init__(self, state_size, action_size, args, number_of_agents=20, is_master=True,
                 device="cpu", eval_mode=False):
        # self.eval_mode = eval_mode
        self.number_of_agents = number_of_agents
        self.action_size = action_size
        self.state_size = state_size
        self.is_master = is_master
        self.gamma = args["gamma"]
#         self.tau = args["tau"]
        self.device = device

        self.TwoHeadModel = TwoHeadModel(state_size, action_size, eval_mode).to(self.device)
        self.optimizer = Adam(self.TwoHeadModel.parameters(), lr=args["lr"])
        self.memory = ExperienceBuffer(args["buffer_size"])

    def step(self, actions, rewards, probs, dones, states):
        self.memory.append(actions, rewards, probs, dones, states)
        samples = self.memory.sample()

        if samples:
            actions, rewards, log_probs, dones, state_values = samples
            rewards = torch.Tensor(rewards).transpose(0, 1).contiguous()
            processed_experience = [None] * (len(samples[0]) - 1)
            return_ = state_values[-1].detach()
            for i in reversed(range(len(samples[0]) - 1)):
                not_done_ = (1 - torch.Tensor(dones[i + 1])).to(self.device).unsqueeze(1)
                reward_ = torch.Tensor(rewards[:, i]).to(self.device).unsqueeze(1)
                return_ = reward_ + self.gamma * not_done_ * return_
                next_value_ = state_values[i + 1]
                advantage_ = reward_ + self.gamma * not_done_ * next_value_.detach() - state_values[i].detach()
                processed_experience[i] = [log_probs[i], advantage_, state_values[i], return_]
            log_probs, advantages, values, returns = map(
                lambda x: torch.cat(x, dim=0), zip(*processed_experience))
            policy_loss = -log_probs * advantages
            value_loss = 0.5 * (returns - values).pow(2)
            self.optimizer.zero_grad()
            loss = (policy_loss + value_loss).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.TwoHeadModel.parameters(), 5)
            self.optimizer.step()
            self.memory.dump_all()

    def act(self, states):
        if isinstance(states, np.ndarray):
            states = torch.Tensor(states).to(self.device).float()
        action, prob, q_value = self.TwoHeadModel(states)

        return action, prob, q_value

    def sync_models(self, global_model):
        self.TwoHeadModel.load_state_dict(global_model.TwoHeadModel.state_dict())

    def copy_gradients(self, agent):
        for param, param_global in zip(agent.TwoHeadModel.parameters(), self.TwoHeadModel.parameters()):
            if param_global.grad is not None:
                return
            param_global._grad = param.grad


import numpy as np

import torch
import torch.nn as nn


def hidden_init(layer):
    """
    Used for parameter initialization
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class TwoHeadModel(nn.Module):
    def __init__(self, n_inputs, n_actions, fc_sizes=[128, 128], eval_mode=False):
        super(TwoHeadModel, self).__init__()
        self.eval_mode = eval_mode
        self.backbone = nn.Sequential(
            nn.Linear(n_inputs, fc_sizes[0]),
            nn.ReLU(),
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(),
        )

        self.actor = nn.Linear(fc_sizes[1], n_actions)
        self.critic = nn.Linear(fc_sizes[1], 1)


    def forward(self, inputs):
        x = self.backbone(inputs)
        policy_out = self.actor(x)

        dist = torch.distributions.Normal(policy_out, 1.0)
        if not self.eval_mode:
            policy_out = dist.sample()
        prob = dist.log_prob(policy_out)

        return torch.clamp(policy_out, -1, 1), prob, self.critic(x)

    def reset_parameters(self):
        """
        Reset parameters to the initial states
        """
        self.backbone.weight.data.uniform_(*hidden_init(self.backbone))
        self.actor.weight.data.uniform_(-3e-3, 3e-3)
        self.critic.weight.data.uniform_(-3e-3, 3e-3)

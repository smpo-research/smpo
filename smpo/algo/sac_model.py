# smpo/algo/sac_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# 也可以直接引用 hiro_model 里的 weights_init，或者在这里重新定义
def weights_init(m)-> None:
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight, gain=2**0.5)  # type: ignore
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class DiscreteSACActor(nn.Module):
    def __init__(
        self,
        state_shape: tuple[int,int],
        goal_shape: tuple[int,int],
        action_dim: int,
        hidden_dim: int = 256
    ) -> None:
        super(DiscreteSACActor, self).__init__()

        self.flat_input_dim: int = state_shape[0] * (state_shape[1] + goal_shape[1])

        self.l1: nn.Linear = nn.Linear(in_features=self.flat_input_dim, out_features=hidden_dim)
        self.l2: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.l3: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=action_dim)

        # --- 初始化逻辑 ---
        # 1. 基础初始化
        self.apply(weights_init)

        # 2. [关键] 输出层 (Logits) 使用极小权重 (Gain=0.01)
        # 这使得初始 Logits 非常小，Softmax 后的概率分布接近均匀 (1/N)。
        # 否则随机初始化可能导致某些动作初始概率很大，Agent 瞬间陷入确定性策略。
        nn.init.orthogonal_(self.l3.weight, gain=0.01)  # type: ignore
        nn.init.constant_(self.l3.bias, 0.0)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x: torch.Tensor = torch.cat([state, goal], dim=-1)
        x = x.view(x.size(0), -1)

        x = F.relu(input=self.l1(x))
        x = F.relu(input=self.l2(x))
        logits: torch.Tensor = self.l3(x)

        probs: torch.Tensor = F.softmax(input=logits, dim=-1)
        # 数值稳定性处理
        z: torch.Tensor = probs == 0.0
        z = z.float() * 1e-8
        log_probs: torch.Tensor = torch.log(input = probs + z)

        return probs, log_probs

    def sample(self, state: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, ...]:
        probs, log_probs = self.forward(state, goal)
        dist: Categorical = Categorical(probs)
        action: torch.Tensor = dist.sample()
        return action, probs, log_probs


class DiscreteSACCritic(nn.Module):
    def __init__(
        self,
        state_shape: tuple[int,int],
        goal_shape: tuple[int,int],
        action_dim: int,
        hidden_dim: int = 256
    ) -> None:
        super(DiscreteSACCritic, self).__init__()

        self.flat_input_dim: int = state_shape[0] * (state_shape[1] + goal_shape[1])

        # Q1
        self.q1_l1: nn.Linear = nn.Linear(in_features=self.flat_input_dim, out_features=hidden_dim)
        self.q1_l2: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.q1_l3: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=action_dim)

        # Q2
        self.q2_l1: nn.Linear = nn.Linear(in_features=self.flat_input_dim, out_features=hidden_dim)
        self.q2_l2: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.q2_l3: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=action_dim)

        # --- 初始化逻辑 ---
        # Critic 正常初始化即可
        self.apply(weights_init)

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x: torch.Tensor = torch.cat([state, goal], dim=-1)
        x = x.view(x.size(0), -1)

        q1 = F.relu(input=self.q1_l1(x))
        q1 = F.relu(input=self.q1_l2(q1))
        q1_out = self.q1_l3(q1)

        q2 = F.relu(input=self.q2_l1(x))
        q2 = F.relu(input=self.q2_l2(q2))
        q2_out = self.q2_l3(q2)

        return q1_out, q2_out

# smpo/algo/sac_agent.py

import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from base.config_para import Parameter as para
from env.chain_env import ChainEnv

from .buffer import PerLowReplayBuffer
from .sac_model import DiscreteSACActor, DiscreteSACCritic
from .utils import to_tensor

# 配置设备


class DiscreteSACAgent:
    def __init__(
        self,
        state_shape: tuple[int,int],
        goal_shape: tuple[int,int],
        action_dim: int,
        device,
        start_training_steps: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        grad_clip: float = 5.0,
        model_path=para.agent_config["model_path"],
        model_save_freq=para.agent_config["model_save_freq"],
        buffer_size: int = para.agent_config["buffer_size"],
        batch_size: int = para.agent_config["batch_size"],
        start_entropy_ratio: float = 0.98,  # 初始目标熵比率
        end_entropy_ratio: float = 0.6,      # 终止目标熵比率
        phase_1_step: int = 50  * 2000,
        phase_2_step: int = 200 * 2000
    ) -> None:
        self.name: str = 'sac_discrete'
        self.model_path: Path = Path(model_path) / "sac"
        self.model_save_freq: int = model_save_freq
        self.start_training_steps: int = start_training_steps
        self.batch_size: int = batch_size
        self.gamma: float = gamma
        self.tau: float = tau
        self.grad_clip: float = grad_clip

        self.state_shape: tuple[int,int] = state_shape
        self.goal_shape: tuple[int,int] = goal_shape
        self.action_dim: int = action_dim
        self.device=device

        # --- 初始化网络 ---
        self.actor: DiscreteSACActor = DiscreteSACActor(state_shape=state_shape, goal_shape=goal_shape, action_dim=action_dim).to(self.device)
        self.critic: DiscreteSACCritic = DiscreteSACCritic(state_shape=state_shape, goal_shape=goal_shape, action_dim=action_dim).to(self.device)
        self.critic_target: DiscreteSACCritic = copy.deepcopy(x=self.critic)

        # 优化器
        self.actor_optimizer: optim.Adam = optim.Adam(params=self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer: optim.Adam = optim.Adam(params=self.critic.parameters(), lr=critic_lr)


        # --- 自动 Alpha 与 Alpha 课程学习 ---
        # 1. 基础熵参数
        self.start_entropy_ratio: float = start_entropy_ratio # 0.98
        self.end_entropy_ratio: float = end_entropy_ratio     # 最终降低到 0.6

        # 2. 阶段步数定义 (假设 MAX_STEP=2000)
        self.phase_1_step: int = phase_1_step   # 0~50保持0.98
        self.phase_2_step: int = phase_2_step   # 51~200从0.98衰减至0.6

        # 3. 初始化 Target Entropy
        # 离散动作空间的目标熵通常设为: -log(1/|A|) * ratio
        self.target_entropy: float = -np.log(1.0 / action_dim) * start_entropy_ratio
        self.log_alpha: torch.Tensor = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer: optim.Adam = optim.Adam(params=[self.log_alpha], lr=alpha_lr)
        self.alpha: torch.Tensor = self.log_alpha.exp()

        # Replay Buffer
        self.replay_buffer: PerLowReplayBuffer = PerLowReplayBuffer(
            state_shape=state_shape,
            goal_shape=goal_shape,
            action_shape=(1,), # 离散动作存 int
            device=self.device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            alpha=0.6,
            beta=0.4
        )

    def reset_entropy(self)-> None:
        with torch.no_grad():
            self.log_alpha.fill_(0.0)  # 重置回初始探索水平 (alpha ≈ 1.0)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.state.clear()

    def select_action(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        evaluate=False
    ) -> int:
        """
        选择动作
        evaluate=True:  Argmax (选择概率最大的节点)
        evaluate=False: Categorical Sample (按概率采样)
        """
        state_t: torch.Tensor = to_tensor(arr=state, device=self.device).unsqueeze(dim=0) # [1, N, M]
        goal_t: torch.Tensor = to_tensor(arr=goal, device=self.device).unsqueeze(dim=0)   # [1, N, 1]

        with torch.no_grad():
            probs, _ = self.actor(state=state_t, goal=goal_t)

            if evaluate:
                # 确定性策略：选择概率最大的
                action: torch.Tensor = torch.argmax(input=probs, dim=1)
            else:
                # 随机策略：按分布采样
                dist: torch.distributions.Categorical = torch.distributions.Categorical(probs)
                action = dist.sample()

        return int(action.item())

    def step(
        self,
        step_num: int,
        env: ChainEnv,
        s: np.ndarray,
        g: np.ndarray|None = None,
        explore: bool = True # 在 SAC 中，explore=True 意味着从分布采样，False 意味着 argmax
    ) -> tuple[np.ndarray, float, np.ndarray, bool]:

        # 处理 Goal
        if g is None:
            g = np.zeros(self.goal_shape, dtype=np.float32)

        # 随机预热
        if step_num < self.start_training_steps and explore:
            a_int: int = env.action_space.sample()
        else:
            a_int = self.select_action(state=s, goal=g, evaluate=not explore)

        # Env step 接收 numpy array 或 int
        n_s, r, done, _, _ = env.step(action=np.array(a_int))

        # 这里的 action 存入 buffer 时保持 (1,) 维度
        return np.array([a_int]), r, n_s, done

    def _train(self)-> dict[str,float]:
        sample_result: tuple[Tensor,...] = self.replay_buffer.sample()

        # =======================================================
        # SAC 的解包与 PER 支持
        # =======================================================
        weights = None
        if len(sample_result) == 8:
            states, goals, actions, n_states, n_goals, rewards, not_done, weights = sample_result
        else:
            states, goals, actions, n_states, n_goals, rewards, not_done = sample_result


        # 确保 action 是 long 类型用于 gather
        actions: Tensor = actions.long()

        # --- 1. 更新 Critic ---
        with torch.no_grad():
            # 计算下一个状态的所有动作概率: pi(a'|s')
            n_probs, n_log_probs = self.actor(n_states, n_goals) # 这里的 goals 假设是定值或由上层给出，此处复用
            n_probs: Tensor = torch.clamp(n_probs, 1e-8, 1.0)
            n_log_probs: Tensor = torch.log(n_probs)

            # 计算目标 Q 值: Q_target(s', all_actions)
            n_q1_target, n_q2_target = self.critic_target(n_states, n_goals)
            n_min_q_target: Tensor = torch.min(n_q1_target, n_q2_target) # [batch, action_dim]

            # 离散 SAC 的核心：直接计算 V(s') 的期望
            # V(s') = sum_a' [ pi(a'|s') * ( Q(s',a') - alpha * log_pi(a'|s') ) ]
            #       = sum_a' [ pi(a'|s') * Q(s',a') ] - alpha * sum_a' [ pi(a'|s') * log_pi(a'|s') ]
            #       = E[Q] + alpha * Entropy
            target_v: Tensor = (n_probs * (n_min_q_target - self.alpha * n_log_probs)).sum(dim=1, keepdim=True)
            # Bellman Equation
            target_q: Tensor = rewards + not_done * self.gamma * target_v

        # 当前 Q 值
        q1, q2 = self.critic(states, goals)
        # 根据实际执行的 action 提取对应的 Q 值: Q(s, a)
        # gather: 沿 dim=1 (action维度) 提取索引为 actions 的值
        curr_q1: Tensor = q1.gather(1, actions)
        curr_q2: Tensor = q2.gather(1, actions)
        #critic_loss: torch.Tensor = F.mse_loss(input=curr_q1, target=target_q) + F.mse_loss(input=curr_q2, target=target_q)
        # 新代码: 使用 SmoothL1Loss 防止梯度爆炸
        # critic_loss: Tensor = F.smooth_l1_loss(input=curr_q1, target=target_q) + F.smooth_l1_loss(input=curr_q2, target=target_q)

        # [关键修改] 计算加权 Loss
        td_loss1: Tensor = F.smooth_l1_loss(input=curr_q1, target=target_q, reduction='none')
        td_loss2: Tensor = F.smooth_l1_loss(input=curr_q2, target=target_q, reduction='none')

        if weights is not None:
            critic_loss: Tensor = (td_loss1 * weights).mean() + (td_loss2 * weights).mean()
        else:
            critic_loss = td_loss1.mean() + td_loss2.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # [新增] Critic 梯度裁剪 (阈值通常设为 1.0 到 10.0 之间)
        torch.nn.utils.clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()

        # [新增] 更新优先级
        # 对于 Double Critic，通常取两者误差的平均值或最大值，或者只用 Q1
        if weights is not None and hasattr(self.replay_buffer, 'update_priorities'):
            with torch.no_grad():
                # 这里使用 Q1 的误差作为优先级的依据
                td_error: Tensor = torch.abs(target_q - curr_q1)
            self.replay_buffer.update_priorities(td_errors=td_error)

        # --- 2. 更新 Actor ---
        # 重新计算当前状态的 pi(a|s) 和 Q(s, :)，构建计算图
        probs, log_probs = self.actor(states, goals)
        # 这里的 Q 网络不需要梯度更新，detach 掉（但这在 PyTorch 中通过 q1.detach() 或只对 actor opt step 均可，
        # 标准做法是使用刚才算的 q1, q2，但视作常数）
        with torch.no_grad():
            q1, q2 = self.critic(states, goals)
            min_q: torch.Tensor = torch.min(q1, q2)
        # Actor Loss: 最小化 KL 散度 -> 最大化 E[Q - alpha * log_pi]
        # Loss = sum_a [ pi(a|s) * (alpha * log_pi(a|s) - Q(s,a)) ]
        actor_loss: torch.Tensor = (probs * (self.alpha.detach() * log_probs - min_q)).sum(dim=1).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # [新增] Actor 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=self.grad_clip)
        self.actor_optimizer.step()

        # --- 3. 更新 Alpha ---
        # Loss = -alpha * (H(pi) - target_entropy) = alpha * sum_a [pi * log_pi] + alpha * target
        # 但为了利用 log_alpha 的梯度，公式重写为:
        # Loss = -log_alpha * (log_pi + target_entropy).mean() (如果是采样的话)
        # 对于离散全期望形式：
        # entropy = - sum (pi * log_pi)
        # alpha_loss = -log_alpha * (entropy - target_entropy)

        with torch.no_grad():
            entropy: torch.Tensor = - (probs * log_probs).sum(dim=1).mean()

        alpha_loss: torch.Tensor = -self.log_alpha * (self.target_entropy - entropy).detach() # detach entropy 否则会反向传播到 actor

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        # [新增] Alpha 梯度裁剪 (可选，但推荐)
        torch.nn.utils.clip_grad_norm_(parameters=[self.log_alpha], max_norm=self.grad_clip)
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(-2.3, 2.5) # 限制 alpha 在 exp(-3) 到 exp(3) 之间

        self.alpha = self.log_alpha.exp()

        # --- 4. 软更新 Target Networks ---
        self._update_target_network(target=self.critic_target, origin=self.critic, tau=self.tau)

        return {
            "sac_critic_loss": critic_loss.item(),
            "sac_actor_loss": actor_loss.item(),
            "sac_alpha_loss": alpha_loss.item(),
            "sac_alpha": self.alpha.item(),
            "sac_entropy": entropy.item()
        }


    def train(self, global_step)-> dict[str,float]:
        train_info: dict[str,float] = {}
        if global_step >= self.start_training_steps:
            self._update_entropy_schedule(global_step)
            return self._train()
        return train_info

    def _update_entropy_schedule(self, global_step: int) -> None:
        """根据当前 global_step 动态调整 target_entropy"""
        if global_step <= self.phase_1_step:
            # [Phase 1] Warm-up / High Exploration: 保持 0.98
            current_ratio: float = self.start_entropy_ratio
        elif global_step <= self.phase_2_step:
            # [Phase 2] Linear Decay: 0.98 -> 0.6
            progress: float = (global_step - self.phase_1_step) / (self.phase_2_step - self.phase_1_step)
            current_ratio = self.start_entropy_ratio - (self.start_entropy_ratio - self.end_entropy_ratio) * progress
        else:
            # [Phase 3] Stable: 保持 0.6
            current_ratio = self.end_entropy_ratio
        # 更新 Target Entropy
        self.target_entropy = -np.log(1.0 / self.action_dim) * current_ratio

    def _update_target_network(self, target, origin, tau)-> None:
        for target_param, origin_param in zip(target.parameters(), origin.parameters()):
            target_param.data.copy_(tau * origin_param.data + (1.0 - tau) * target_param.data)

    def append(self, s, g, a, n_s, n_g, r, d) -> None:
        self.replay_buffer.append(state=s, goal=g, action=a, n_state=n_s, n_goal=n_g, reward=r, done=d)


    def save(self, episode: int, model_path = None) -> None:
        if model_path is None:
            model_path = self.model_path
        path: str = os.path.join(model_path, str(episode))
        if not os.path.exists(path=path):
            os.makedirs(name=path)
        torch.save(obj=self.actor.state_dict(), f=os.path.join(path, self.name + "_actor.pth"))
        torch.save(obj=self.critic.state_dict(), f=os.path.join(path, self.name + "_critic.pth"))
        # 保存 log_alpha
        torch.save(obj=self.log_alpha, f=os.path.join(path, self.name + "_alpha.pth"))

    def load(self, episode: int, model_path = None) -> None:
        """加载指定回合的模型，episode=-1时加载最新模型"""
        if model_path is None:
            model_path = self.model_path
        if episode < 0:
            # 只获取能转换为整数的子目录名（过滤掉 .DS_Store 等非数字文件）
            episode_list: list[int] = []
            for item in os.listdir(model_path):
                try:
                    ep_num = int(item)  # 尝试转换为整数
                    episode_list.append(ep_num)
                except ValueError:
                    continue  # 跳过无法转换的文件（如 .DS_Store）
            if not episode_list:
                return
            episode = max(episode_list)
        path = os.path.join(model_path, str(episode))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.load_state_dict(
            state_dict=torch.load(
                f=os.path.join(path, self.name + "_actor.pth"),
                weights_only=True,
                map_location=device
            )
        )
        self.critic.load_state_dict(
            state_dict=torch.load(
                f=os.path.join(path, self.name + "_critic.pth"),
                weights_only=True,
                map_location=device
            )
        )
        self.critic_target = copy.deepcopy(x=self.critic)
        # 加载 log_alpha
        alpha_path = os.path.join(path, self.name + "_alpha.pth")
        if os.path.exists(alpha_path):
            self.log_alpha.data = torch.load(alpha_path, map_location=self.device)
            self.alpha = self.log_alpha.exp()

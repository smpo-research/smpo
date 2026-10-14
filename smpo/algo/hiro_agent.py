import copy
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.typing import NDArray
from torch import Tensor

from base.config_para import Parameter as para
from env.mmec_env import MmecEnv

from .buffer import PerHighReplayBuffer, PerLowReplayBuffer
from .hiro_model import ManagerActor, ManagerCritic, WorkerActor, WorkerCritic
from .utils import to_tensor

# 配置设备


class BaseController:
    """Base Controller for TD3-style training"""
    def __init__(
        self,
        actor_class,   # 第1个参数：Actor类
        critic_class,  # 第2个参数：Critic类
        state_shape,   # 第3个参数：State维度
        goal_shape,    # 第4个参数：Goal维度 (Manager是Global Goal, Worker是Subgoal)
        action_shape,  # 第5个参数：Action维度 (Manager是Subgoal, Worker是Env Action)
        device,
        scale=1.0,
        actor_lr=1e-4,
        critic_lr=3e-4,
        expl_noise=0.1,
        policy_noise=0.2,
        grad_clip=5.0,      # <--- [新增] 梯度裁剪阈值，默认 1.0 或 5.0
        noise_clip=0.5,
        gamma=0.99,
        tau=0.005,
        policy_freq=2,
        model_path="./result/model"
    )-> None:
        self.name: str = 'hiro'
        self.expl_noise: float = expl_noise
        self.policy_noise: float = policy_noise
        self.noise_clip: float = noise_clip
        self.grad_clip: float=grad_clip
        self.gamma: float = gamma
        self.tau: float = tau
        self.policy_freq: int = policy_freq
        self.model_path: str = model_path
        self.scale: float = scale
        self.total_it: int = 0
        self.device=device

        # 初始化网络
        self.actor: ManagerActor|WorkerActor = actor_class(state_shape, goal_shape, action_shape, action_scale=scale).to(self.device)
        self.actor_target: ManagerActor|WorkerActor = copy.deepcopy(x=self.actor)
        self.actor_optimizer: optim.Adam = optim.Adam(params=self.actor.parameters(), lr=actor_lr)

        self.critic: ManagerCritic|WorkerCritic = critic_class(state_shape, goal_shape, action_shape).to(self.device)
        self.critic_target: ManagerCritic|WorkerCritic = copy.deepcopy(x=self.critic)
        self.critic_optimizer: optim.Adam = optim.Adam(params=self.critic.parameters(), lr=critic_lr)

    def select_action(
        self,
        state:np.ndarray,
        goal:np.ndarray,
        noise:bool=False
    )-> np.ndarray:
        with torch.no_grad():
            state_t: Tensor = to_tensor(arr=state, device=self.device).unsqueeze(dim=0)
            goal_t: Tensor = to_tensor(arr=goal, device=self.device).unsqueeze(dim=0)
            action: Tensor = self.actor(state_t, goal_t)
            if noise:
                noise_t: Tensor = (torch.randn_like(input=action) * self.expl_noise).clamp(-self.noise_clip, self.noise_clip)
                action = (action + noise_t).clamp(-self.scale, self.scale)
            return action.cpu().numpy().squeeze(axis=0)

    def select_action_tensor(self, state_t: Tensor, goal_t: Tensor)-> Tensor:
        return self.actor(state_t, goal_t)

    def train(self, buffer, batch_size=None)-> dict[str,Any]:
        self.total_it += 1
        sample_result:tuple[Tensor, ...] = buffer.sample()

        # 兼容两种Buffer的解包逻辑
        # if isinstance(buffer, LowReplayBuffer):
        #     state, goal, action, n_state, n_goal, reward, not_done = sample_result
        # else:
        #     state, goal, action, n_state, reward, not_done, _, _ = sample_result
        #     n_goal = goal # HighBuffer goal通常是不变的，或者由外部处理

        # 兼容 PER Buffer 的解包和 Loss 计算逻辑
        weights = None

        # 判断返回长度来确定是否包含 weights
        # PrioritizedLowReplayBuffer 返回 8 个元素 (最后是 weights)
        if len(sample_result) == 8:
            state, goal, action, n_state, n_goal, reward, not_done, weights = sample_result
        # 普通 LowReplayBuffer 返回 7 个元素
        elif len(sample_result) == 7:
            state, goal, action, n_state, n_goal, reward, not_done = sample_result
        # Fallback (针对 HighBuffer 传入 BaseController 的防御性编程)
        else:
            state, goal, action, n_state, reward, not_done, _, _, _ = sample_result
            n_goal = goal

        # --- 1. Critic Update (Double Q) ---
        with torch.no_grad():
            noise: Tensor = (torch.randn_like(input=action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action: Tensor = (self.actor_target(n_state, n_goal) + noise).clamp(-self.scale, self.scale)

            # [关键] 获取两个 Target Q，取最小值
            target_q1, target_q2 = self.critic_target(n_state, n_goal, next_action)
            target_q_min: Tensor = torch.min(target_q1, target_q2)

            target_q = reward + (not_done * self.gamma * target_q_min).detach()

        # [关键] 获取两个 Current Q
        current_q1, current_q2 = self.critic(state, goal, action)

        # [关键] 分别计算 Loss 并求和
        td_loss1: Tensor = F.smooth_l1_loss(input=current_q1, target=target_q, reduction='none')
        td_loss2: Tensor = F.smooth_l1_loss(input=current_q2, target=target_q, reduction='none')



        if weights is not None:
            critic_loss: Tensor = (td_loss1 * weights).mean() + (td_loss2 * weights).mean()
        else:
            critic_loss = td_loss1.mean() + td_loss2.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Critic 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()

        # 更新优先级 (使用 Q1 的误差)
        if weights is not None and hasattr(buffer, 'update_priorities'):
            with torch.no_grad():
                td_error: Tensor = torch.abs(target_q - current_q1)
            buffer.update_priorities(td_error)

        # --- 2. Actor Update ---
        actor_loss_val: float = 0.0
        if self.total_it % self.policy_freq == 0:
            pi: Tensor = self.actor(state, goal)
            # 只使用 Q1 来更新 Actor
            actor_loss: Tensor = -self.critic.q1(state, goal, pi).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            # Actor 梯度裁剪
            torch.nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=self.grad_clip)
            self.actor_optimizer.step()

            self._soft_update(target=self.critic_target, source=self.critic)
            self._soft_update(target=self.actor_target, source=self.actor)
            actor_loss_val = actor_loss.item()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}

    def _soft_update(self, target, source)-> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tp.data * (1.0 - self.tau) + sp.data * self.tau)

    def save(self, name:str)-> None:
        torch.save(obj=self.actor.state_dict(), f=name+"_actor.pth")
        torch.save(obj=self.critic.state_dict(), f=name+"_critic.pth")

    def load(self, name:str)-> None:
        if os.path.exists(path = self.name+"_actor.pth"):
            self.actor.load_state_dict(state_dict=torch.load(f=name+"_actor.pth"))
            self.critic.load_state_dict(state_dict=torch.load(f=name+"_critic.pth"))
            self.actor_target = copy.deepcopy(x=self.actor)
            self.critic_target = copy.deepcopy(x=self.critic)

class WorkerController(BaseController):
    def __init__(self, state_shape, goal_shape, action_shape, device, scale=1.0, **kwargs)-> None:
        """
        Worker:
        - input goal: Subgoal (from Manager)
        - output action: Environment Action
        """
        super().__init__(
            actor_class=WorkerActor,  # 显式传递 WorkerActor 类
            critic_class=WorkerCritic, # 显式传递 WorkerCritic 类
            state_shape=state_shape,
            goal_shape=goal_shape,
            action_shape=action_shape,
            device=device,
            scale=scale,
            **kwargs
        )

class ManagerController(BaseController):
    def __init__(self, state_shape, goal_shape, action_shape,device, scale=1.0, **kwargs):
        """
        Manager:
        - input goal: Global Goal
        - output action: Subgoal (for Worker)
        """
        super().__init__(
            actor_class=ManagerActor,  # 显式传递 ManagerActor 类
            critic_class=ManagerCritic, # 显式传递 ManagerCritic 类
            state_shape=state_shape,
            goal_shape=goal_shape,
            action_shape=action_shape,
            device=device,
            scale=scale,
            **kwargs
        )

    def off_policy_corrections(
        self,
        worker_con:WorkerController,
        batch_size:int,
        actions:np.ndarray,
        states_seq:np.ndarray,
        actions_seq:np.ndarray,
        seq_lens:np.ndarray,
        candidate_num: int = 10
    )-> np.ndarray:
        first_s:np.ndarray = states_seq[:, 0, ...]
        last_s:np.ndarray = states_seq[:, -1, ...]
        diff_goal:np.ndarray = (last_s[..., -1:] - first_s[..., -1:])
        original_goal:np.ndarray = actions
        # 注意：这里假设 subgoal 的 scale 和 manager 的 scale 一致
        # 使用 [:, None, ...] 在 batch 之后插入 candidate_num 维度
        random_goals:np.ndarray = np.random.normal(
            loc=diff_goal[:, None, ...],
            scale=0.5 * self.scale,
            size=(batch_size, candidate_num, *original_goal.shape[1:])
        )
        random_goals = np.clip(random_goals, -self.scale, self.scale)
        candidates:np.ndarray = np.concatenate(
            [original_goal[:, None, ...], diff_goal[:, None, ...], random_goals], axis=1
        )
        # 必须确保 diff_goal (以及所有候选) 都在 Manager 的 Scale 范围内
        candidates = np.clip(candidates, -self.scale, self.scale)
        best_goals: np.ndarray = copy.deepcopy(x=original_goal)
        max_log_prob: np.ndarray = -np.inf * np.ones(batch_size)
        states_seq_t: Tensor = to_tensor(arr=states_seq, device=self.device)
        actions_seq_t: Tensor = to_tensor(arr=actions_seq, device=self.device)
        # [新增] 准备 Mask 比较用的长度
        seq_lens_t: Tensor = to_tensor(arr=seq_lens, device=self.device).reshape(-1)
        for c in range(candidates.shape[1]):
            cand_g: np.ndarray = candidates[:, c, ...]
            cand_g_t: Tensor = to_tensor(arr=cand_g, device=self.device)
            log_prob_sum: Tensor = torch.zeros(batch_size, device=self.device)
            for t in range(states_seq.shape[1]):
                s_t: Tensor = states_seq_t[:, t, ...]
                a_t_real: Tensor = actions_seq_t[:, t, ...]
                s_0_feat: Tensor = states_seq_t[:, 0, ..., -1:]
                s_t_feat: Tensor = s_t[..., -1:]
                # HIRO 转换公式: g_t = s_0 + g - s_t
                # Worker 的输入 goal 是 g_step
                g_step: Tensor = s_0_feat + cand_g_t - s_t_feat
                pred_a: Tensor = worker_con.select_action_tensor(state_t=s_t, goal_t=g_step)
                # 使用负 MSE 代表 log probability (假设高斯分布)
                mse: Tensor = -F.mse_loss(input=pred_a, target=a_t_real, reduction='none')
                step_log_prob: Tensor = mse.sum(dim=(1, 2, 3))
                mask: Tensor = (t < seq_lens_t).float()
                log_prob_sum += step_log_prob * mask
            log_prob_sum_np: np.ndarray = log_prob_sum.detach().cpu().numpy()
            better_idx: np.ndarray = log_prob_sum_np > max_log_prob
            max_log_prob[better_idx] = log_prob_sum_np[better_idx]
            best_goals[better_idx] = cand_g[better_idx]

        return best_goals

    def train(self, buffer, batch_size=None, worker_con:WorkerController|None=None)-> dict[str,Any]:
        if worker_con is None:
            raise ValueError("Manager training requires worker_con for off-policy correction.")

        sample_result: tuple[Tensor, ...] = buffer.sample()
        # state, goal, action, n_state, reward, not_done, state_arr, action_arr, seq_lens = sample_result

        # =======================================================
        # Manager 的解包与 PER 支持
        # =======================================================
        weights = None
        # PrioritizedHighReplayBuffer 返回 10 个元素
        if len(sample_result) == 10:
            state, goal, action, n_state, reward, not_done, state_arr, action_arr, seq_lens, weights = sample_result
        # 普通 HighReplayBuffer 返回 9 个元素
        else:
            state, goal, action, n_state, reward, not_done, state_arr, action_arr, seq_lens = sample_result

        corrected_actions: np.ndarray = self.off_policy_corrections(
            worker_con=worker_con,
            batch_size=state.shape[0],
            actions=action.cpu().numpy(),
            states_seq=state_arr.cpu().numpy(),
            actions_seq=action_arr.cpu().numpy(),
            seq_lens=seq_lens.cpu().numpy(),
        )

        corrected_actions_t: Tensor = to_tensor(arr=corrected_actions, device=self.device)
        self.total_it += 1

        # --- 1. Critic Update (Double Q) ---
        with torch.no_grad():
            noise: Tensor = (torch.randn_like(input=corrected_actions_t) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action: Tensor = (self.actor_target(n_state, goal) + noise).clamp(-self.scale, self.scale)

            # [关键] Double Q Min
            target_q1, target_q2 = self.critic_target(n_state, goal, next_action)
            target_q_min: Tensor = torch.min(target_q1, target_q2)

            target_q = reward + (not_done * self.gamma * target_q_min).detach()

        # [关键] 获取两个 Current Q (使用修正后的动作)
        current_q1, current_q2 = self.critic(state, goal, corrected_actions_t)

        # [关键] 计算 Double Q Loss
        td_loss1: Tensor = F.smooth_l1_loss(input=current_q1, target=target_q, reduction='none')
        td_loss2: Tensor = F.smooth_l1_loss(input=current_q2, target=target_q, reduction='none')

        if weights is not None:
            critic_loss: Tensor = (td_loss1 * weights).mean() + (td_loss2 * weights).mean()
        else:
            critic_loss = td_loss1.mean() + td_loss2.mean()


        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # [新增] Manager Critic 梯度裁剪
        torch.nn.utils.clip_grad_norm_(parameters=self.critic.parameters(), max_norm=self.grad_clip)
        self.critic_optimizer.step()

        # 更新优先级
        if weights is not None and hasattr(buffer, 'update_priorities'):
            with torch.no_grad():
                td_error: Tensor = torch.abs(target_q - current_q1)
            buffer.update_priorities(td_error)

        # --- 2. Actor Update ---
        actor_loss_val: float = 0.0
        if self.total_it % self.policy_freq == 0:
            pi = self.actor(state, goal)
            # [关键] 只用 Q1 更新 Actor
            actor_loss = -self.critic.q1(state, goal, pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.actor.parameters(), max_norm=self.grad_clip)
            self.actor_optimizer.step()

            self._soft_update(target=self.critic_target, source=self.critic)
            self._soft_update(target=self.actor_target, source=self.actor)
            actor_loss_val = actor_loss.item()

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_val}


class HiroAgent:
    def __init__(
        self,
        esn_num,
        image_num,
        feature_dim,
        device,
        start_training_steps: int,
        inst_num = para.ms_config["instance_candidate_num"],
        model_path=para.agent_config["model_path"],
        model_save_freq=para.agent_config["model_save_freq"],
        buffer_size=para.agent_config["buffer_size"],
        batch_size=para.agent_config["batch_size"],
        buffer_freq=para.agent_config["hiro_buffer_freq"],  # manager多久决策一次
        train_freq=para.agent_config["hiro_train_freq"],    #多少个step训练一次
        grad_clip=5.0,  # <---可以在这里配置梯度裁剪
        #manager_scale = np.log10(1+esn_num),
        worker_scale = 3.0,
        worker_reward_alpha: float = 0.2 # 环境reward辅助，不宜过高, 0.2刚好能提供 0.1左右的辅助矫正
    )-> None:
        self.state_shape: tuple[int,...] = (esn_num, image_num, feature_dim)
        # Global Goal shape
        self.goal_shape: tuple[int,...] = (esn_num, image_num, 1)
        # Action shape (同时也是 Subgoal shape)
        self.action_shape: tuple[int,...] = (esn_num, image_num, 1)
        # # manager的全局目标（这里使用微服务的图谱聚类）
        # # 暂时不初始化
        # self.fg =None
        self.device=device

        self.start_training_steps = start_training_steps
        self.model_path: Path = Path(model_path) / "hiro"
        self.manager_buffer_freq: int = buffer_freq
        self.train_freq: int = train_freq
        # 保存 alpha 值, 用来把一部分reward分给worker
        self.worker_reward_alpha: float = worker_reward_alpha


        # Manager: 输入 Global Goal, 输出 Subgoal (Action)
        manager_scale = np.log10(1+inst_num).astype(float)
        self.manager: ManagerController = ManagerController(
            state_shape=self.state_shape,
            goal_shape=self.goal_shape,
            action_shape=self.action_shape,
            scale=manager_scale,
            grad_clip=grad_clip,
            device = self.device

        )

        # Worker: 输入 Subgoal (作为Goal), 输出 Env Action
        self.worker: WorkerController = WorkerController(
            state_shape=self.state_shape,
            goal_shape=self.action_shape,
            action_shape=self.action_shape,
            scale=worker_scale,
            grad_clip=grad_clip,
            device = self.device
        )

        self.low_buffer: PerLowReplayBuffer = PerLowReplayBuffer(
            state_shape=self.state_shape,
            goal_shape=self.action_shape,
            action_shape=self.action_shape,
            device = self.device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            alpha=0.6,
            beta=0.4
        )

        self.high_buffer: PerHighReplayBuffer = PerHighReplayBuffer(
            state_shape=self.state_shape,
            goal_shape=self.goal_shape,
            subgoal_shape=self.action_shape,
            low_action_shape=self.action_shape,
            device = self.device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            freq=buffer_freq,
            alpha=0.6,
            beta=0.4
        )

        self.current_subgoal: np.ndarray = np.zeros(self.action_shape, dtype=np.float32)
        self.reset_temp_buffer()

    def reset_temp_buffer(self)-> None:
        self.temp_states: list[np.ndarray] = []
        self.temp_actions: list[np.ndarray] = []
        self.temp_start_state: np.ndarray = np.zeros(self.state_shape, dtype=np.float32)
        self.temp_subgoal: np.ndarray = np.zeros(self.action_shape, dtype=np.float32)
        self.temp_reward_sum: float = 0.0

    def reset_status(self)-> None:
        """在Episode开始或从评估模式切换回来时调用"""
        self.reset_temp_buffer()
        # 可以选择将 current_subgoal 置零，或者在 step 中强制重新生成
        self.current_subgoal = np.zeros(self.action_shape, dtype=np.float32)

    def step(
        self,
        state: np.ndarray,
        env: MmecEnv,
        step_num: int,
        explore=True,
        start_of_episode=False  # <--- 新增参数
    )-> tuple[np.ndarray, float, np.ndarray, bool]:


        is_new_subgoal: bool = (step_num % self.manager_buffer_freq == 0) or start_of_episode

        if is_new_subgoal:
            # 如果是 start_of_episode 触发的，我们需要先清空之前的残余 buffer
            if start_of_episode:
                self.reset_temp_buffer()
            if explore and step_num < self.start_training_steps:
                # 使用 self.manager.scale
                limit: float = self.manager.scale
                self.current_subgoal = np.random.uniform(-limit, limit, self.action_shape).astype(np.float32)
            else:
                self.current_subgoal = self.manager.select_action(state, goal=env.final_g, noise=explore)

            self.temp_start_state = state.copy()
            self.temp_subgoal = self.current_subgoal.copy()
            self.temp_reward_sum = 0.0

        if explore and step_num < self.start_training_steps:
            # 使用 self.worker.scale
            limit = self.worker.scale
            action: NDArray[np.float32] = np.random.uniform(-limit, limit, self.action_shape).astype(np.float32)
        else:
            # Worker 使用 current_subgoal 作为输入
            action = self.worker.select_action(state, self.current_subgoal, noise=explore)

        obs, reward, done, _, info = env.step(action=action.squeeze(-1))
        next_state: np.ndarray = obs.astype(np.float32)

        # === [关键修改] 获取真实执行的动作 ===
        if "masked_action" in info:
            # 这一步至关重要！我们要把 Env 真正执行的动作拿回来
            action = info["masked_action"]
            # 确保维度匹配 (N, M) -> (N, M, 1)
            if action.ndim == 2:
                action = action[..., None]



        # === [修改] 只有在训练模式下才记录 temp buffer ===
        if explore:
            self.temp_states.append(state)
            self.temp_actions.append(action)
            self.temp_reward_sum += float(reward)

        s_feat, ns_feat = state[..., -1:], next_state[..., -1:]
        # 计算下一时刻的目标 g_{t+1}
        next_subgoal: np.ndarray = s_feat + self.current_subgoal - ns_feat
        # HIRO 内在奖励计算
        # ======基于距离的惩罚=========
        dist_error: float = float(np.linalg.norm(s_feat + self.current_subgoal - ns_feat))
        # 防止 dist_error 过大导致 reward 极负
        max_dist_penalty: float = 10.0
        dist_term: float = min(dist_error, max_dist_penalty)
        # ======基于余弦相似度的奖励=======
        # 1. 计算 Worker 实际的移动向量
        actual_movement: np.ndarray = ns_feat - s_feat
        # 2. 计算 Manager 期望的移动向量 (即 subgoal)
        desired_movement: np.ndarray = self.current_subgoal
        # 3. 展平向量以计算整体相似度
        act_vec: np.ndarray = actual_movement.flatten()
        des_vec: np.ndarray = desired_movement.flatten()
        # 4. 计算余弦相似度 (值域 -1 到 1) 加上 epsilon 防止除零
        epsilon: float = 1e-8
        cos_sim = np.dot(act_vec, des_vec) / (np.linalg.norm(act_vec) * np.linalg.norm(des_vec) + epsilon)

        # 5. 映射奖励
        w_cos: float = 0.95 * 2
        w_dist: float = 0.05
        intrinsic_reward: float = w_cos * float(cos_sim) - w_dist * float(dist_term) # 推荐尝试正向奖励，或者做线性映射

        # 稍微放大一点环境奖励
        scaled_env_reward: float = float(reward) * 5
        # 最终worker得分
        worker_reward: float = (1.0 - self.worker_reward_alpha) * intrinsic_reward + self.worker_reward_alpha * scaled_env_reward

        if explore:
            # 只有在 explore=True (训练) 时，才把数据存入 Buffer
            self.low_buffer.append(
                state=state,
                goal=self.current_subgoal,
                action=action,
                n_state=next_state,
                n_goal=next_subgoal,
                reward=worker_reward,
                done=done
            )
            # 只有在 explore=True (训练) 时，才把数据存入 Buffer
            if (step_num + 1) % self.manager_buffer_freq == 0 or done:
                if len(self.temp_states) > 0:  # <--- 确保序列长度完整
                    self.high_buffer.append(
                        state=self.temp_start_state,
                        goal=env.final_g,
                        action=self.temp_subgoal,
                        n_state=next_state,
                        reward=self.temp_reward_sum,
                        done=done,
                        state_arr=np.array(self.temp_states),
                        action_arr=np.array(self.temp_actions)
                    )
                    self.reset_temp_buffer()

        # Subgoal Transition function
        self.current_subgoal = next_subgoal

        return action.squeeze(axis=-1), reward, next_state, done

    def train(self, global_step)-> dict[str,float]:
        losses:dict[str,float] = {}
        if global_step >= self.start_training_steps:
            low_loss:dict[str,float] = self.worker.train(buffer=self.low_buffer)
            losses.update({"hiro_low_" + k: v for k, v in low_loss.items()})

            if global_step % self.train_freq == 0:
                high_loss:dict[str,float] = self.manager.train(buffer=self.high_buffer, worker_con=self.worker)
                losses.update({"hiro_high_" + k: v for k, v in high_loss.items()})

        return losses

    def save(self,  episode, model_path = None)-> None:
        if model_path is None:
            model_path = self.model_path
        path: str = os.path.join(model_path, str(episode))
        if not os.path.exists(path):
            os.makedirs(name=path)
        self.manager.save(name=os.path.join(path, "manager"))
        self.worker.save(name=os.path.join(path, "worker"))

    def load(self, model_path, episode)-> None:
        path: str = os.path.join(model_path, str(episode))
        self.manager.load(name=os.path.join(path, "manager"))
        self.worker.load(name=os.path.join(path, "worker"))

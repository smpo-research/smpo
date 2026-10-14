# smpo/algo/buffer.py

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from base.config_para import Parameter as para


class LowReplayBuffer:
    def __init__(
        self,
        state_shape: tuple,     # (N,M,F)或(N,F)
        goal_shape: tuple,
        action_shape: tuple,
        device: torch.device,
        buffer_size: int = para.agent_config["buffer_size"],
        batch_size: int = para.agent_config["batch_size"]
    ) -> None:
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.ptr: int = 0
        self.size: int = 0
        self.device: torch.device = device
        # 初始化存储空间
        self.state: np.ndarray = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.goal: np.ndarray = np.zeros((buffer_size, *goal_shape), dtype=np.float32)
        self.action: np.ndarray = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.n_state: np.ndarray = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.n_goal: np.ndarray = np.zeros((buffer_size, *goal_shape), dtype=np.float32)
        self.reward: np.ndarray = np.zeros((buffer_size, 1), dtype=np.float32)
        self.not_done: np.ndarray = np.zeros((buffer_size, 1), dtype=np.float32)

    def append(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        action: np.ndarray,
        n_state: np.ndarray,
        n_goal: np.ndarray,  # 显式要求传入 n_goal
        reward: float,
        done: bool
    ) -> None:
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.n_goal[self.ptr] = n_goal
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self) -> tuple[Tensor, ...]:
        ind: np.ndarray = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.tensor(data=self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.goal[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.action[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.n_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.n_goal[ind], dtype=torch.float32).to(self.device), # 返回 n_goal
            torch.tensor(data=self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.not_done[ind], dtype=torch.float32).to(self.device),
        )


class HighReplayBuffer:
    def __init__(
        self,
        state_shape: tuple,     # (N,M,F)
        goal_shape: tuple,
        subgoal_shape: tuple,    # High Action
        low_action_shape: tuple, # 底层动作维度，用于存储序列
        device: torch.device,
        buffer_size: int = para.agent_config["buffer_size"],
        batch_size: int = para.agent_config["batch_size"],
        freq: int = para.agent_config["hiro_buffer_freq"]           # 动作序列长度    10
    ) -> None:
        self.buffer_size: int = buffer_size
        self.batch_size: int = batch_size
        self.ptr: int = 0
        self.size: int = 0
        self.device: torch.device = device
        self.freq: int = freq

        # 初始化存储空间
        self.state: np.ndarray = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.goal: np.ndarray = np.zeros((buffer_size, *goal_shape), dtype=np.float32)
        self.action: np.ndarray = np.zeros((buffer_size, *subgoal_shape), dtype=np.float32)
        self.n_state: np.ndarray = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.reward: np.ndarray = np.zeros((buffer_size, 1), dtype=np.float32)
        self.not_done: np.ndarray = np.zeros((buffer_size, 1), dtype=np.float32)
        # High Level 特有: 存储底层动作序列和状态序列 (用于HIRO等算法的Off-policy correction)
        self.state_arr: np.ndarray = np.zeros((buffer_size, self.freq, *state_shape), dtype=np.float32)
        self.action_arr: np.ndarray = np.zeros((buffer_size, self.freq, *low_action_shape), dtype=np.float32)
        # 记录state_arr和action_arr为0的index
        self.seq_lens: np.ndarray = np.zeros((buffer_size, 1), dtype=np.int32)

    def append(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        action: np.ndarray,
        n_state: np.ndarray,
        reward: float,
        done: bool,
        state_arr: np.ndarray,  # 显式要求传入序列
        action_arr: np.ndarray
    ) -> None:
        self.state[self.ptr] = state
        self.goal[self.ptr] = goal
        self.action[self.ptr] = action
        self.n_state[self.ptr] = n_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done


        # 获取输入数据的实际长度
        actual_freq_len = state_arr.shape[0]
        store_len: int = min(actual_freq_len, self.freq)
        # 先清零当前槽位 (Padding 准备)
        self.state_arr[self.ptr] = 0.0
        self.action_arr[self.ptr] = 0.0
        # 填入数据 (使用 store_len 进行切片)
        # state_arr[:store_len] 自动截断输入中多余的部分
        self.state_arr[self.ptr, :store_len] = state_arr[:store_len]
        self.action_arr[self.ptr, :store_len] = action_arr[:store_len]

        self.seq_lens[self.ptr] = store_len
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self) -> tuple[Tensor, ...]:
        ind: np.ndarray = np.random.randint(0, self.size, size=self.batch_size)

        return (
            torch.tensor(data=self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.goal[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.action[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.n_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.not_done[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.state_arr[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.action_arr[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.seq_lens[ind], dtype=torch.int32).to(self.device)
        )


# ==========================================
# 新增: SumTree 数据结构 (用于 O(log N) 采样)
# ==========================================
class SumTree:
    """
    一种二叉树结构，叶子节点存储优先级，父节点存储子节点的和。
    用于在 O(log N) 时间内根据优先级进行采样。
    """
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.tree: NDArray[np.float64] = np.zeros(2 * capacity - 1)
        self.write: int = 0

    def add(self, priority: float)-> None:
        idx: int = self.write + self.capacity - 1
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_total_priority(self) -> float:
        return self.tree[0]

    def sample(self, value: float) -> tuple[int, float, int]:
        parent_idx: int = 0
        while True:
            left_child: int = 2 * parent_idx + 1
            right_child: int = left_child + 1
            if left_child >= len(self.tree):
                leaf_idx: int = parent_idx
                break
            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

        data_idx: int = leaf_idx - (self.capacity - 1)
        return leaf_idx, self.tree[leaf_idx], data_idx


# ==========================================
# PerLowReplayBuffer (PER 实现 - 修复版)
# ==========================================
class PerLowReplayBuffer(LowReplayBuffer):
    def __init__(
        self,
        state_shape,
        goal_shape,
        action_shape,
        device,
        buffer_size=para.agent_config["buffer_size"],
        batch_size=para.agent_config["batch_size"],
        total_steps: int = 100*2000,
        alpha=0.6,
        beta=0.4
    )-> None:
        super().__init__(state_shape, goal_shape, action_shape, device, buffer_size, batch_size)
        self.alpha: float = alpha
        self.beta: float = beta
        self.beta_increment: float = (1.0 - beta) / total_steps
        self.max_priority: float = 1.0
        self.tree: SumTree = SumTree(capacity=buffer_size)
        self.epsilon: float = 1e-6  # [新增] 防止除以零的极小值

        self.last_tree_indices: list[int]|None = None

    def append(self, state, goal, action, n_state, n_goal, reward, done) -> None:
        super().append(state, goal, action, n_state, n_goal, reward, done)
        # 存入时确保优先级有效
        priority = self.max_priority if self.max_priority > 0 else 1.0
        self.tree.add(priority=priority ** self.alpha)

    def sample(self) -> tuple[Tensor, ...]:
        total_p = self.tree.get_total_priority()

        # [修复1] 防御 SumTree 损坏 (NaN 或 0)
        if np.isnan(total_p) or total_p <= 0:
            # 这种情况极少发生，除非网络已经输出了 NaN
            print(f"[Warning] SumTree corrupted (total_p={total_p}). Resetting segment.")
            total_p = float(self.batch_size) # 强制给一个有效值，避免除以零

        segment: float = total_p / self.batch_size
        batch_indices: list[int] = []
        tree_indices: list[int] = []
        is_weights: NDArray[np.float64] = np.empty((self.batch_size, 1))

        for i in range(self.batch_size):
            a: float = segment * i
            b: float = segment * (i + 1)

            # [修复2] 防止 uniform 报错 (OverflowError / bounds error)
            # b 可能会因为浮点误差略微超过 total_p
            if b > total_p:
                b = total_p
            # 确保 a < b，防止崩溃
            if a >= b:
                s = a
            else:
                s = np.random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.sample(value=s)

            data_idx: int = min(data_idx, self.size - 1)
            tree_indices.append(tree_idx)
            batch_indices.append(data_idx)

            # [修复3] 安全计算概率
            prob: float = priority / total_p
            # 防止 prob 为 0 导致 power 运算结果为 Inf
            safe_prob = max(prob, self.epsilon)

            # 计算权重: (N * P)^(-beta)
            is_weights[i, 0] = np.power(self.size * safe_prob, -self.beta)

        # [修复4] 安全归一化权重
        max_weight = is_weights.max()
        if max_weight == 0 or np.isnan(max_weight) or np.isinf(max_weight):
            is_weights[:] = 1.0 # 异常情况回退到 1.0
        else:
            is_weights /= max_weight

        self.beta = np.min([1.0, self.beta + self.beta_increment])
        self.last_tree_indices = tree_indices

        ind: NDArray[np.int64] = np.array(batch_indices)

        return (
            torch.tensor(data=self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.goal[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.action[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.n_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.n_goal[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.not_done[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=is_weights, dtype=torch.float32).to(self.device)
        )

    def update_priorities(self, td_errors: Tensor)-> None:
        if self.last_tree_indices is None:
            raise RuntimeError("Must call sample() before update_priorities()!")

        td_errors_np: NDArray[np.float64] = td_errors.detach().cpu().numpy().flatten()

        # [修复5] 彻底清洗 NaN 和 Inf，防止污染 SumTree
        # 如果网络梯度爆炸输出了 NaN，这里必须拦截，否则下次 sample 必崩
        td_errors_np = np.nan_to_num(td_errors_np, nan=1e-5, posinf=1.0, neginf=1e-5)
        td_errors_np = np.abs(td_errors_np) # 确保绝对值

        for i, idx in enumerate(self.last_tree_indices):
            # 加上 epsilon 确保优先级不为 0
            error = td_errors_np[i] + self.epsilon
            self.tree.update(idx, priority=float(error ** self.alpha))

        # 更新 max_priority
        current_max: float = float(np.max(td_errors_np))
        if not np.isnan(current_max) and not np.isinf(current_max):
            self.max_priority = max(self.max_priority, current_max)

        self.last_tree_indices = None


# ==========================================
# PerHighReplayBuffer (PER 实现 - 修复版)
# ==========================================
class PerHighReplayBuffer(HighReplayBuffer):
    def __init__(
        self,
        state_shape,
        goal_shape,
        subgoal_shape,
        low_action_shape,
        device,
        buffer_size=para.agent_config["buffer_size"],
        batch_size=para.agent_config["batch_size"],
        freq=para.agent_config["hiro_buffer_freq"],
        total_steps: int = 100*2000,
        alpha=0.6,
        beta=0.4
    )-> None:
        super().__init__(state_shape, goal_shape, subgoal_shape, low_action_shape, device, buffer_size, batch_size, freq)
        self.alpha: float = alpha
        self.beta: float = beta
        self.beta_increment: float = (1.0 - beta) / total_steps
        self.max_priority: float = 1.0
        self.tree: SumTree = SumTree(capacity=buffer_size)
        self.epsilon: float = 1e-6 # [新增]

        self.last_tree_indices: list[int]|None = None

    def append(self, state, goal, action, n_state, reward, done, state_arr, action_arr) -> None:
        super().append(state, goal, action, n_state, reward, done, state_arr, action_arr)
        # 存入时确保优先级有效
        priority = self.max_priority if self.max_priority > 0 else 1.0
        self.tree.add(priority=priority ** self.alpha)

    def sample(self) -> tuple[Tensor, ...]:
        total_p = self.tree.get_total_priority()

        # [修复1] 防御 SumTree 损坏
        if np.isnan(total_p) or total_p <= 0:
            total_p = self.batch_size

        segment: float = total_p / self.batch_size
        batch_indices: list[int] = []
        tree_indices: list[int] = []
        is_weights: NDArray[np.float64] = np.empty((self.batch_size, 1))

        for i in range(self.batch_size):
            a: float = segment * i
            b: float = segment * (i + 1)

            # [修复2] 防止 uniform 报错
            if b > total_p:
                b = total_p
            if a >= b:
                s = a
            else:
                s = np.random.uniform(a, b)

            tree_idx, priority, data_idx = self.tree.sample(value=s)

            data_idx: int = min(data_idx, self.size - 1)
            tree_indices.append(tree_idx)
            batch_indices.append(data_idx)

            # [修复3] 安全计算概率和权重
            prob: float = priority / total_p
            safe_prob = max(prob, self.epsilon)
            is_weights[i, 0] = np.power(self.size * safe_prob, -self.beta)

        # [修复4] 安全归一化
        max_weight = is_weights.max()
        if max_weight == 0 or np.isnan(max_weight) or np.isinf(max_weight):
            is_weights[:] = 1.0
        else:
            is_weights /= max_weight

        self.beta = np.min([1.0, self.beta + self.beta_increment])
        self.last_tree_indices = tree_indices

        ind: NDArray[np.int64] = np.array(batch_indices)
        return (
            torch.tensor(data=self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.goal[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.action[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.n_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.not_done[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.state_arr[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.action_arr[ind], dtype=torch.float32).to(self.device),
            torch.tensor(data=self.seq_lens[ind], dtype=torch.int32).to(self.device),
            torch.tensor(data=is_weights, dtype=torch.float32).to(self.device)
        )

    def update_priorities(self, td_errors: Tensor):
        if self.last_tree_indices is None:
            raise RuntimeError("Must call sample() before update_priorities()!")

        td_errors_np: NDArray[np.float64] = td_errors.detach().cpu().numpy().flatten()

        # [修复5] 清洗 NaN 和 Inf
        td_errors_np = np.nan_to_num(td_errors_np, nan=1e-5, posinf=1.0, neginf=1e-5)
        td_errors_np = np.abs(td_errors_np)

        for i, idx in enumerate(self.last_tree_indices):
            # 加上 epsilon
            error = td_errors_np[i] + self.epsilon
            self.tree.update(idx, priority=float(error ** self.alpha))

        current_max: float = float(np.max(td_errors_np))
        if not np.isnan(current_max) and not np.isinf(current_max):
            self.max_priority = max(self.max_priority, current_max)

        self.last_tree_indices = None

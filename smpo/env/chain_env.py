# smpo/env/chain_env.py
from collections import deque
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from base.config_para import Parameter as para
from base.esn import EdgeServerNode
from base.esns import MmecBdScheduler
from base.ms_control import InstanceControl
from base.task_chain import MsChain
from base.task_compute import MsComputeTask
from base.task_transmit import MsDownlinkTask, MsUplinkTask
from base.user_request import AllRequestInfo, UserRequest, UserRequestTask


class ChainEnv(gym.Env):
    """
    任务管理器 (Sequential & Immediate Update Version)

    特性：
    1. 细粒度决策：在一个时隙内，依次对每个就绪任务进行 Step。
    2. 即时更新：Step 收到 Action 后，立即调用该 Chain 的 update()，确保下一个 Chain 做决策时，能看到上一个 Chain 造成的资源变化。
    3. 防重复机制：使用 updated_in_current_slot 集合防止任务在同一时隙被更新两次。
    """
    metadata: dict[str,Any] = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        arrival_rate: float|None = None,
        esn_num: int = para.esnc_config["esn_num"],
        image_num: int = para.ms_config["microservice_num"],
        instance_num: int = para.ms_config["instance_candidate_num"],
        user_num: int = para.user_config["user_num"],
        max_exec_tolerance: int = para.chain_config["max_execute_tolerable_slot"],
        total_simu_slot: int = para.simu_config["total_time_slots"],
        feature_dim: int = 7,  # 显式传入特征维度，与 _get_obs 保持一致
        reward_scale_factor: float = 1.0,
        evaluate=False,
        cool_down_slot: int = 300  # 冷却时间，默认 200 时隙
    )-> None:
        super().__init__()
        self.evaluate: bool = evaluate

        # --- 1. Gym ---
        self.esn_num: int = esn_num
        self.image_num = image_num
        self.instance_num: int = instance_num
        self.feature_dim: int = feature_dim
        # action space
        self.action_space = spaces.Discrete(n=self.esn_num) # 选择哪一个 ESN (0 到 esn_num-1)
        # observation:
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.esn_num, self.feature_dim),
            dtype=np.float32
        )

        # --- 2. 基础信息 ---
        self.user_num: int = user_num
        self.reward_scale_factor: float = reward_scale_factor

        # esn
        self.esn_nodes: list[EdgeServerNode] = [EdgeServerNode(node_id=i) for i in range(self.esn_num)]
        self.instance_control: InstanceControl = InstanceControl()


        self.request_info: AllRequestInfo = AllRequestInfo(evaluate=evaluate,arrival_rate=arrival_rate)
        self.user_requests: list[UserRequest] = self.request_info.user_requests
        self.total_chains_info: list[list[UserRequestTask]] = self.request_info.total_slot_chains_info
        # self.total_chain_num: int = self.request_info.total_chain_num

        # 仿真总时长 1000 -> 在 800 时隙停止生成，剩下 200 时隙处理尾部任务
        self.generation_end_slot: int = max(0, total_simu_slot - cool_down_slot)
        # 只统计 generation_end_slot 之前生成的任务，避免把注定跑不完的任务算进分母
        self.total_chain_num: int = 0
        # 防止越界，取 min
        if self.evaluate:
            limit_idx = min(len(self.total_chains_info), self.generation_end_slot)
            for s in range(limit_idx):
                self.total_chain_num += len(self.total_chains_info[s])
        else:
            self.total_chain_num = self.request_info.total_chain_num

        # 任务队列
        self.executing_chains: list[MsChain] = []  # 执行中队列
        self.completed_chains: list[MsChain] = []  # 正常完成队列，包括轻微超时
        self.timeout_chains: list[MsChain] = []  # 系统超时队列

        # 任务队列决策控制
        self.pending_decisions: deque[MsChain] = deque() # 当前时隙等待决策的任务队列
        self.updated_in_current_slot: set[int] = set()   # 防止重复更新的集合 {chain_id}
        self.chain_last_obs: dict[int, np.ndarray] = {}    # 管理 Chain 上一次决策时的 Observation

        # 环境控制
        self.current_slot: int = 0
        self.total_slots: int = total_simu_slot
        self.max_exec_slot: int = max_exec_tolerance  # 单位为slot
        self.done: bool = False

        # 统计信息
        self.chain_finished_num: int = 0
        self.chain_completed_num: int = 0
        self.chain_on_time_num: int = 0
        self.chain_timeout_num: int = 0
        self.chain_success_rate_slist: list[float] = [0.0 for _ in range(self.total_slots)]
        self.task_migrate_num_slist: list[int] = [0 for _ in range(self.total_slots)]
        self.chain_schedule_rate_slist: list[float] = [0.0 for _ in range(self.total_slots)]
        self.chain_timeout_ratio_slist: list[float] = [0.0 for _ in range(self.total_slots)]

        self.task_demand_count_slist: list[list[int]] = [[0 for _ in range(self.image_num)] for _ in range(self.total_slots)]
        self.task_finished_num_slist: list[int] = [0 for _ in range(self.total_slots)]
        self.task_on_time_num_slist: list[int] = [0 for _ in range(self.total_slots)]
        self.task_success_rate_slist: list[float] = [0.0 for _ in range(self.total_slots)]



    def reset(
        self,
        *,
        seed: int|None = None,
        options: dict|None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置并快进到第一个决策点"""
        super().reset(seed=seed)

        # 重置信息

        self.request_info.reset()
        self.user_requests = self.request_info.user_requests
        self.total_chains_info = self.request_info.total_slot_chains_info
        # === 分母计算逻辑 ===
        self.total_chain_num = 0 # 先清零
        if self.evaluate:
            # 重新计算截断后的分母
            limit_idx = min(len(self.total_chains_info), self.generation_end_slot)
            for s in range(limit_idx):
                self.total_chain_num += len(self.total_chains_info[s])
        else:
            self.total_chain_num = self.request_info.total_chain_num

        self.executing_chains.clear()
        self.completed_chains.clear()
        self.timeout_chains.clear()
        self.pending_decisions.clear()
        self.updated_in_current_slot.clear()
        self.chain_last_obs.clear()
        self.current_slot = 0
        self.done = False
        self.chain_finished_num = 0
        self.chain_completed_num = 0
        self.chain_on_time_num = 0
        self.chain_timeout_num = 0
        self.task_migrate_num_slist = [0 for _ in range(self.total_slots)]
        self.chain_success_rate_slist = [0.0 for _ in range(self.total_slots)]
        self.chain_schedule_rate_slist = [0.0 for _ in range(self.total_slots)]
        self.chain_timeout_ratio_slist = [0.0 for _ in range(self.total_slots)]
        self.task_demand_count_slist = [[0 for _ in range(self.image_num)] for _ in range(self.total_slots)]
        self.task_finished_num_slist = [0 for _ in range(self.total_slots)]
        self.task_on_time_num_slist = [0 for _ in range(self.total_slots)]
        self.task_success_rate_slist = [0.0 for _ in range(self.total_slots)]
        MmecBdScheduler().reset()
        # 初始加载与快进
        self._process_slot_arrivals()   # 收集第0时隙到达的任务
        self._scan_for_pending_decisions()  # 检查是否有任务一出生就需要计算（通常不会，因为有传输，但为了健壮性）
        if not self.pending_decisions:
            self._fast_forward()
        # 如果快进直接导致结束，返回全0观测
        obs: np.ndarray = np.zeros((self.esn_num, self.feature_dim), dtype=np.float32) if self.done else self._get_obs()

        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Gymnasium step:
        Args:   针对 current pending task 的 esn_id
        Return: (obs, reward, terminated, truncated, info)
        """
        # 类型转换，确保 action 是 int (某些算法可能传进来 numpy int)
        if action.size != 1:
               raise ValueError(f"Expected action to be a single-value array, got shape {action.shape}")
        action_int: int = int(action.item())  # 安全转换为 P

        if self.done:
            info = {
                "error": "Env already done",
                "waiting": 0  # 关键：显式告诉外部没有在等待的任务
            }
            return np.zeros((self.esn_num, self.feature_dim), dtype=np.float32), 0.0, True, False, info
        if not self.pending_decisions:
            # 理论上 _fast_forward 保证了只要不 done 就有 pending
            raise RuntimeError("No task waiting for decision but env is not done.")
        # =========================================
        # 1. 立即执行：应用决策并更新状态
        # =========================================
        current_chain: MsChain = self.pending_decisions.popleft()
        target_task: MsComputeTask = cast(MsComputeTask,current_chain.current_task)
        # 记录self.task_demand_count_slist
        self.task_demand_count_slist[self.current_slot][target_task.task_id] += 1
        # 必须先生成 Observation 再更新，因为更新后资源状态就变了
        current_obs: np.ndarray = cast(np.ndarray, self._get_obs(target_chain=current_chain))
        self.chain_last_obs[current_chain.chain_id] = current_obs
        # 计算决策前的fairness
        # before_fairness: float = self._get_fairness_reward()
        # 更新决策
        current_chain.update(esn_id=action_int)
        # 标记该链在本时隙已运行过，稍后统一推进时跳过它
        self.updated_in_current_slot.add(current_chain.chain_id)

        # 检查 Target Task 是否完成
        if target_task.is_completed:
            self._check_and_update_task_stats(task=target_task)
        # 检查该任务是否完成
        if current_chain.is_completed:
            self.completed_chains.append(current_chain)
            if current_chain in self.executing_chains:
                self.executing_chains.remove(current_chain)
            # --- 统计更新 ---
            self.chain_completed_num += 1          # <--- Add: 完成总数+1
            self.chain_finished_num += 1
            if current_chain.complete_on_time:
                self.chain_on_time_num += 1        # <--- Add: 准时完成+1

        # after_fairness: float = self._get_fairness_reward()
        # fairness_reward: float = max(0, min(after_fairness - before_fairness, 1))

        # 临时 metrics，通常只在时隙切换时结算
        reward: float = self._get_task_reward(chain=current_chain, target_task=target_task)
        # reward += fairness_reward
        # 根据每个step获得的r数值对reward进行缩放
        reward = reward / self.reward_scale_factor
        # =========================================
        # 2. 决策循环逻辑
        # =========================================
        info: dict[str, Any] = {}
        # 当前时隙还有其他任务排队等待决策
        if self.pending_decisions:
            # 直接返回下一个任务的 Observation
            # 注意：此时时间 current_slot 没变，环境资源状态变了
            info["waiting"] = len(self.pending_decisions)

        else:
            # 当前时隙所有决策已处理完，可以推进时间了
            # 这里计算reward(fairness_reward+slot_reward)容易产生分布问题，现在只交给task_reward
            self._finish_slot_and_advance()
            # 如果推进后没有 pending decision，继续快进直到有决策或结束
            if not self.pending_decisions and not self.done:
                self._fast_forward()
        obs: np.ndarray = np.zeros((self.esn_num, self.feature_dim), dtype=np.float32) if self.done else self._get_obs()
        return obs, reward, self.done, False, info



    def _get_obs(
        self,
        target_chain: MsChain|None = None,
        instance_num: int = para.ms_config["instance_candidate_num"],
        delta: int = para.simu_config["time_slot"]
    )-> np.ndarray:
        """
        针对特定任务，观察所有ESN的状态。
        Returns:
            Shape: (N,7)
            Features: [
                (0)上次决策是否在该ESN,
                (1)该ESN可用Instance数(针对该任务类型),
                (2)该ESN剩余CPU能力(百分比),
                (3)chain的进度（比例）
                (4)预计计算延迟/task_deadline
                (5)该task剩余deadline/task_deadline，
                (6)该chain已经超时时间/chain_deadline
                #(7)goal（该esn增加或减少的instance数，除以instance数归一化）
            ]
        """
        if target_chain is None:
            if self.done or not self.pending_decisions:
                raise ValueError("ChainEnv is done or not pending_decisions, but still get_obs!")
            target_chain = self.pending_decisions[0]
        # 获取下一个要决策的chain的pending_task
        current_pending_task: MsComputeTask = cast(MsComputeTask,target_chain.current_task)
        # 初始化obs
        obs: np.ndarray = np.zeros((self.esn_num, self.feature_dim), dtype=np.float32)


        # === 1. 批量获取 ESN 级特征 (避免循环中多次调用 .core_use_rate) ===
        # 使用预缓存的 esn_nodes
        esn_core_rates: np.ndarray = np.array([node.core_use_rate for node in self.esn_nodes], dtype=np.float32)
        # 获取所有 ESN 的最大可用核心 (用于计算延迟)
        # 假设 available_core 是列表，取 max，如果没有则为 0
        max_compute_capacities: np.ndarray = np.array(
            [max(node.available_core) if node.available_core else 0.0 for node in self.esn_nodes], dtype=np.float32
        )

        # === 2. 向量化计算 Instance 特征 (避免双重循环) ===
        task_id = current_pending_task.task_id
        task_size = current_pending_task.task_size

        # 切片获取该 Image 在所有 ESN 和所有 Instance 上的状态
        # Shape: (esn_num, instance_num)
        view_deployed = self.instance_control.state_deployed[:, task_id, :]
        view_occupied = self.instance_control.state_occupied[:, task_id, :]
        # 假设 state_memory 存储的是实例容量 (如果是动态使用量，需确认逻辑)
        view_memory = self.instance_control.state_memory[:, task_id, :]
        # 条件：已部署 AND 未被占用 AND 内存容量 > 任务需求
        valid_mask = view_deployed & (~view_occupied) & (view_memory > task_size)
        # 按行求和，得到每个 ESN 的可用实例数
        available_inst_counts = np.sum(valid_mask, axis=1).astype(np.float32)

        # === 3. 填充 Obs ===
        # (0) 上次决策位置
        last_decided_esn_id = current_pending_task.decided_esn_id
        if last_decided_esn_id is not None:
            obs[last_decided_esn_id, 0] = 1.0
        # (1) 可用 Instance 数 (向量化结果)
        obs[:, 1] = available_inst_counts / self.instance_num
        # (2) ESN CPU 使用率
        obs[:, 2] = esn_core_rates
        # (3) Chain 进度 (标量广播)
        obs[:, 3] = target_chain.schedule_rate
        # (4) 预计计算延迟 (向量化)
        remain_cycle = current_pending_task.remain_compute_cycle
        task_ddl = current_pending_task.task_deadline
        # 避免除以 0
        safe_caps = max_compute_capacities.copy()
        safe_caps[safe_caps == 0] = 1e-6 # 防止除零
        if task_ddl > 0:
            latencies = np.ceil(remain_cycle / safe_caps / delta / task_ddl)
            # 对于 capacity 为 0 的节点，延迟设为 1.0 (最大风险)
            latencies[max_compute_capacities == 0] = 1.0
        else:
            latencies = np.ones(self.esn_num, dtype=np.float32)
        obs[:, 4] = np.clip(latencies, 0.0, 1.0)
        # (5) 任务剩余 DDL
        feature_5 = current_pending_task.remain_deadline / task_ddl if task_ddl > 0 else 0.0
        obs[:, 5] = feature_5
        # (6) Chain 超时情况
        if target_chain.chain_deadline > 0:
            feature_6 = min(target_chain.timeout_status / target_chain.chain_deadline, 1.0)
        else:
            feature_6 = 1.0 if target_chain.timeout_status > 0 else 0.0
        obs[:, 6] = feature_6

        return obs


    def _finish_slot_and_advance(self) -> None:
        """
        结束当前时隙：
        1. 更新那些【本时隙还没更新过】的链
        2. 推进 current_slot
        3. 收集新任务。
        4. 扫描新的待决策任务。
        5.获得完成的reward
        """

        # === [新增] 统计累加器 ===
        total_schedule_rate: float = 0.0
        total_timeout_ratio: float = 0.0
        active_chain_count: int = 0

        current_slot_finished: list[MsChain] = []
        # 遍历所有活跃链
        for chain in list(self.executing_chains):
            # 如果该链已被step()里的action更新update，则跳过
            if chain.chain_id in self.updated_in_current_slot:
                continue

            # 更新前保存当前的计算任务引用, 更新前保存“任意类型”的任务引用，而不仅仅是 MsComputeTask
            # task_to_check: None|MsComputeTask = None
            # if chain.is_current_task_compute:
            #     task_to_check = cast(MsComputeTask,chain.current_task)
            task_to_check: MsComputeTask|MsUplinkTask|MsDownlinkTask|None = chain.current_task
            # 执行常规更新
            if chain.is_current_task_compute:
                # 记录
                target_task: MsComputeTask = cast(MsComputeTask, chain.current_task)
                self.task_demand_count_slist[self.current_slot][target_task.task_id] += 1

                # 只有正在进行的计算任务才需要 ID，且必须已经有 ID
                decided_esn_id: int|None = chain.current_compute_esn_id
                if decided_esn_id is not None:
                    # 更新计算任务
                    chain.update(esn_id=decided_esn_id)
                else:
                    # 异常：处于计算状态但无ID，且不在 pending 列表里？
                    raise ValueError("In computing state but without decided_esn_id, and not in the pending list!")
            else:
                # 更新传输任务
                chain.update()

            # 统计task信息
            if task_to_check is not None:
                if task_to_check.is_completed:
                    self._check_and_update_task_stats(task=task_to_check)

            # 统计chain完成情况
            if chain.is_completed:
                self.completed_chains.append(chain)
                current_slot_finished.append(chain)
                self.executing_chains.remove(chain)
                # --- 统计更新 ---
                self.chain_completed_num += 1       # <--- Add: 完成总数+1
                self.chain_finished_num += 1
                if chain.complete_on_time:
                    self.chain_on_time_num += 1     # <--- Add: 准时完成+1
                # 任务完成，移除观测记录
                if chain.chain_id in self.chain_last_obs:
                    del self.chain_last_obs[chain.chain_id]

            # 统计chain调度与超时信息
            total_schedule_rate += chain.schedule_rate
            deadline: int = chain.chain_deadline if chain.chain_deadline > 0 else 20
            timeout_ratio: float = chain.timeout_status / deadline
            total_timeout_ratio += min(1, timeout_ratio) # 截断到 1.0
            active_chain_count += 1

        # 结算统计chain调度与超时信息
        if active_chain_count > 0:
            self.chain_schedule_rate_slist[self.current_slot] = total_schedule_rate / active_chain_count
            self.chain_timeout_ratio_slist[self.current_slot] = total_timeout_ratio / active_chain_count
        else:
            self.chain_schedule_rate_slist[self.current_slot] = 0.0
            self.chain_timeout_ratio_slist[self.current_slot] = 0.0

        # 清理step内update过的标记，准备下一帧
        self.updated_in_current_slot.clear()

        # 统计并记录当前slot内的success rate
        slot_success_rate: float = self._success_rate_in_slot(finished_chain=current_slot_finished)
        self.chain_success_rate_slist[self.current_slot] = slot_success_rate
        task_on_time_num: int = self.task_on_time_num_slist[self.current_slot]
        task_finished_num: int = self.task_finished_num_slist[self.current_slot]
        self.task_success_rate_slist[self.current_slot] = task_on_time_num/task_finished_num if task_finished_num>0 else 0.0
        # 时隙最后时刻band allocate
        MmecBdScheduler().alloc_band_pslot()
        # 更新时间
        self.current_slot += 1
        # 处理新时隙到达
        if self.current_slot < self.total_slots:
            self._process_slot_arrivals()
            self._scan_for_pending_decisions()
        else:
            self.done = True


    def _fast_forward(self) -> None:
        """如果没有待决策任务，持续推进时隙直到有任务或结束"""
        while not self.pending_decisions and not self.done:
            # 执行一步完整的时隙更新（因为没有 pending，所以 active 中的所有链都需要 update）
            self._finish_slot_and_advance()

    def _scan_for_pending_decisions(self) -> None:
        """
        1、扫描活跃链，找出需要立即决策的任务
        2、添加需要再决策的 chain（基于状态漂移、负载均衡、DDL风险）
        """
        # 预先获取参数
        drift_threshold: float = para.hyperpara_config.get("obs_drift_threshold", 0.5)
        for chain in list(self.executing_chains):
            # 如果超时被清理了，直接处理下一个，不再进行后续决策逻辑
            if self._process_timeout_chain(chain):
                continue
            # 必须是计算任务
            if not chain.is_current_task_compute:
                continue
            # --- 情况1: 初始决策 (没有 ESN ID) ---
            if chain.current_compute_esn_id is None:
                if chain not in self.pending_decisions:
                    self.pending_decisions.append(chain)
                continue
            # --- 情况2: 再决策/迁移 (已有 ESN ID) ---
            # 防重复检查
            if chain in self.pending_decisions or chain.chain_id in self.updated_in_current_slot:
                continue
            current_task: MsComputeTask = cast(MsComputeTask, chain.current_task)
            # [优化] 逻辑顺序调整：先算便宜的，后算贵的
            # 1. 检查 DDL 风险 (最便宜)
            is_deadline_risk: bool = not current_task.is_on_schedule
            # 如果没有 DDL 风险，我们再看是否有优化的必要
            should_redecide = False
            if is_deadline_risk:
                # 2. 判断负载均衡
                if self._esn_jain_fairness() < 0.95:
                    # 3. 检查 状态漂移 (最贵，涉及 _get_obs)
                    # 只有前两个条件都满足时，才去计算 Observation 和 Drift
                    last_obs: np.ndarray | None = self.chain_last_obs.get(chain.chain_id)
                    is_drift_large = False
                    if last_obs is not None:
                        current_obs: np.ndarray = cast(np.ndarray, self._get_obs(target_chain=chain))
                        drift: float = float(np.linalg.norm(current_obs - last_obs))
                        if drift > drift_threshold:
                            is_drift_large = True
                    if is_drift_large:
                        should_redecide = True
            if should_redecide:
                self.pending_decisions.append(chain)

    def _process_slot_arrivals(self) -> None:
        """读取 UserRequest 在当前时隙生成的新任务"""
        if self.current_slot >= self.total_slots:
            return
        if self.evaluate:
            if self.current_slot >= self.generation_end_slot:
                return
        raw_tasks: list[UserRequestTask] = self.total_chains_info[self.current_slot]
        if not raw_tasks:
            return
        for raw_task in raw_tasks:
            new_chain: MsChain = MsChain(user_request_task=raw_task)
            new_chain.start(start_slot=self.current_slot)   # 启动任务
            self.executing_chains.append(new_chain)

    def _process_timeout_chain(self, chain: MsChain) -> bool:
        """
        检查链是否超时。如果超时，执行清理操作（移除队列、标记失败）。
        Returns: True if timeout and dropped, False otherwise.
        """
        # 计算已存活时间
        alive_duration: int = self.current_slot - chain.start_slot
        if alive_duration > self.max_exec_slot:
            # 移入完成队列（尽管是失败的）
            self.timeout_chains.append(chain)
            self.chain_timeout_num += 1     # <--- Add: 超时总数+1
            self.chain_finished_num += 1
            # 从执行队列移除
            if chain in self.executing_chains:
                self.executing_chains.remove(chain)
            # 从待决策队列移除 (防止它还在队列里)
            if chain in self.pending_decisions:
                self.pending_decisions.remove(chain)
            # 清理观测缓存
            if chain.chain_id in self.chain_last_obs:
                del self.chain_last_obs[chain.chain_id]
            return True
        return False

    def _check_and_update_task_stats(self, task: MsComputeTask|MsUplinkTask|MsDownlinkTask) -> None:
        """检查任务是否刚完成，如果是，则更新统计数据"""
        # 只有当任务真正完成时才统计
        if not task.is_completed:
            return
        # 避免重复统计：你需要确保这个任务没有被统计过。
        self.task_finished_num_slist[self.current_slot] += 1
        # 判断是否准时
        duration: int = task.complete_slot - task.start_slot
        limit: int = task.task_deadline if task.task_deadline > 0 else 0
        if duration <= limit:
            self.task_on_time_num_slist[self.current_slot] += 1

    def _get_task_reward(
        self,
        chain: MsChain,
        target_task: Any = None,    # [Fix] 接收具体的任务对象
        migrate_alpha: float = para.hyperpara_config["migrate_alpha"],
        exec_beta: float = para.hyperpara_config["exec_beta"],
        # timeout_penalty: float = para.hyperpara_config["timeout_penalty"],
        # exec_lambda: float = para.hyperpara_config["task_exec_lambda"],
        migrate_penalty_postitive: float = para.hyperpara_config["migrate_penalty_postitive"],
        migrate_penalty_negative: float = para.hyperpara_config["migrate_penalty_negative"]
    )-> float:
        """计算每个task被决策后的reward"""
        # 优先使用 target_task，解决 chain.current_task 指针偏移问题
        task: MsComputeTask = cast(MsComputeTask,target_task if target_task is not None else chain.current_task)

        # 迁移reward
        migrate_reduced_slot: int = task.migrate_reduced_slot[-1] if task.migrate_reduced_slot else 0
        if migrate_reduced_slot !=0:
            self.task_migrate_num_slist[self.current_slot] += 1
        # migrate_penalty_postitive=0.5, migrate_penalty_negative=0.8

        migrate_penalty = migrate_penalty_postitive if migrate_reduced_slot >= 0 else migrate_penalty_negative

        migrate_reduced_ratio: float =  max(-1, min(1, migrate_reduced_slot/chain.chain_deadline))
        migrate_reward: float =  max(-1, min(1,migrate_penalty * migrate_reduced_ratio))
        # 计算reward
        if task.is_completed:
            # 实际消耗时隙 / 截止时间
            exec_deadline_ratio: float = (task.complete_slot - task.start_slot) / task.task_deadline if task.task_deadline > 0 else 2.0
        else:
            exec_deadline_ratio = task.expect_exec_deadline_ratio

        if 0 <= exec_deadline_ratio <= 1:
            # execute_reward: float = np.exp(-exec_lambda*(1-exec_deadline_ratio)) * max(0, min(1,chain.schedule_rate))
            # speed_bonus 范围: (0, 1]
            speed_bonus = 1 * (1.0 - exec_deadline_ratio)
            # 最终得分范围: [1.0, 1.0 + exec_lambda]
            safe_schedule_rate: float = max(0.3, min(1.0, chain.schedule_rate))
            execute_reward: float = (1.0 + speed_bonus) * safe_schedule_rate
        else:
            # [超时区间] 连续衰减函数
            # 逻辑：从 1.0 开始，随时间指数衰减，最终收敛到 -max_overdue_penalty
            # 公式： R = S * [ (1+P)*exp(-k*(r-1)) - P ]
            # r0 = [1+In(1+1/P)/K]
            # K=0.8 P=2 | K=1.2 P=1
            overdue_decay_k: float = 2  # 2的时候大概1.4就是0
            max_overdue_penalty: float = 1.0
            decay_factor = np.exp(-overdue_decay_k * (exec_deadline_ratio - 1))
            base_reward_factor = (1.0 + max_overdue_penalty) * decay_factor - max_overdue_penalty
            safe_schedule_rate = max(0.3, min(1.0, chain.schedule_rate))
            execute_reward = base_reward_factor * safe_schedule_rate

            # 下限保护 (虽然公式理论上有下限 -P*S，但在极端输入下加一层保险总是好的)
            execute_reward = max(-5.0, execute_reward)

        return migrate_alpha * migrate_reward + exec_beta * execute_reward


    def _success_rate_in_slot(self, finished_chain: list[MsChain]) -> float:
        if not finished_chain:
            return 0.0  # 直接返回
        success_num: int = sum(1 for c in finished_chain if c.complete_on_time)
        return success_num / len(finished_chain)


    def _is_in_jupyter(self) -> bool:
        """检测是否在 Jupyter Notebook 或 JupyterLab 中运行"""
        try:
            from IPython import get_ipython  # type: ignore
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter Notebook / Lab
            elif shell == 'TerminalInteractiveShell':
                return False  # IPython in terminal
            else:
                return False  # Other cases (e.g., None)
        except (NameError, ImportError, AttributeError):
            return False      # Not IPython

    def render(self)-> None:
        """简单的打印渲染"""
        output_str = (
            f"Slot: {self.current_slot} / {self.total_slots} | "
            f"Total: {self.total_chain_num} | "
            f"Complete: {self.chain_completed_num} "
            f"(OK: {self.chain_on_time_num}, TimeOut: {self.chain_timeout_num}) | "
            f"Pending: {len(self.pending_decisions)} / Executing: {len(self.executing_chains)}"
        )
        if self._is_in_jupyter():
            from IPython.display import clear_output
            clear_output(wait=True)  # wait=True 防止闪烁
            print(output_str)
        else:
            print(output_str)



    def _get_fairness_reward(
        self,
        fair_gama: float =para.hyperpara_config["fair_gama"]
    )-> float:
        jain_fairness: float = self._esn_jain_fairness()
        return fair_gama * jain_fairness

    def _esn_jain_fairness(self)-> float:
        """J=(∑xi​)^2​/(n⋅∑​xi^2​)"""
        compute_use_ratios: list[float] = [
            EdgeServerNode(node_id=i).core_use_rate for i in range(self.esn_num)
        ]
        sum_x: float = sum(compute_use_ratios)
        sum_x_sq: float = sum(x ** 2 for x in compute_use_ratios)
        jain_fairness: float = (sum_x ** 2) / (self.esn_num * sum_x_sq) if sum_x_sq != 0 else 1.0
        return jain_fairness

    @property
    def chain_complete_ratio(self)-> float:
        return round(self.chain_completed_num/self.chain_finished_num,2) if self.chain_finished_num else 0.0
    @property
    def chain_on_time_ratio(self)-> float:
        return round(self.chain_on_time_num/self.chain_finished_num,2) if self.chain_finished_num else 0.0

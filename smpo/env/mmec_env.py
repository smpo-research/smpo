# smpo/env/mmec_env.py
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from algo.goal_gen import GoalGenerator
from algo.sac_agent import DiscreteSACAgent
from base.config_para import HyperparaConfig
from base.config_para import Parameter as para

# 导入你提供的控制模块
from base.esn import EdgeServerNode
from base.ms_control import ImageControl, InstanceControl
from base.user_request import AllRequestInfo
from env.chain_env import ChainEnv


class MmecEnv(gym.Env):
    """
    基于 Gymnasium 的 HIRO 分层调度环境
    """

    def __init__(
        self,
        chain_env: ChainEnv,
        sac_agent: DiscreteSACAgent,
        esn_num: int = para.esnc_config["esn_num"],
        image_num: int = para.ms_config["microservice_num"],
        instance_num: int = para.ms_config["instance_candidate_num"],
        image_decision_interval = para.simu_config["image_decision_interval"],
        total_simu_slot: int = para.simu_config["total_time_slots"],
        feature_dim: int = 9,
        reward_scale_factor: float = 1.0,
        evaluate=False
    )-> None:
        super().__init__()
        self.evaluate: bool = evaluate

        if self.evaluate != chain_env.evaluate:
            raise ValueError("the evaluate model of ChainEnv and MmecEnv is different!")

        # --- 1. Gym ---
        self.esn_num: int = esn_num
        self.image_num: int = image_num
        self.instance_num: int = instance_num
        self.feature_dim: int = feature_dim
        # 定义 Action Space (Worker)
        self.action_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(self.esn_num, self.image_num),
            dtype=np.float32
        )
        # 定义 Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.esn_num, self.image_num, self.feature_dim),
            dtype=np.float32
        )

        # --- 2. 基础信息 ---
        self.chain_env: ChainEnv = chain_env
        self.chain_obs: np.ndarray|None = None
        self.sac_agent: DiscreteSACAgent = sac_agent
        all_generate_chain: list[list[int]] = self.chain_env.request_info.all_generate_chain
        self.final_g: np.ndarray = GoalGenerator().get_goal(chain_list=all_generate_chain)


        # 所有请求信息
        self.request_info: AllRequestInfo = self.chain_env.request_info
        # 获取控制器实例 (单例模式)
        self.image_control: ImageControl = ImageControl()
        self.instance_control: InstanceControl = InstanceControl()

        # [Optimization] 预先缓存 ESN 对象，避免在 get_obs 循环中反复实例化
        self.esn_nodes: list[EdgeServerNode] = [EdgeServerNode(node_id=i) for i in range(self.esn_num)]
        # [Vectorization] 预计算 ESN 容量矩阵，用于快速除法 (Shape: esn_num)
        self.esn_memory_capacities = np.array([node.memory_capacity for node in self.esn_nodes], dtype=np.float32)
        # 避免除以0
        self.esn_memory_capacities[self.esn_memory_capacities == 0] = 1.0

        # 时隙信息
        self.total_slots: int = total_simu_slot
        self.image_decision_interval: int = image_decision_interval
        # rewad缩放
        self.reward_scale_factor: float = reward_scale_factor

        # 缓存当前的镜像决策状态 (用于在非更新slot时维持输入)
        self.last_image_decision: np.ndarray = np.zeros((self.esn_num, self.image_num), dtype=np.bool_)
        self.current_image_decision: np.ndarray = np.zeros((self.esn_num, self.image_num), dtype=np.bool_)
        self.current_instance_activation: np.ndarray = np.zeros((self.esn_num, self.image_num), dtype=np.float32)

        # 时序控制
        self.sac_global_step: int = 0
        self.current_slot: int = 0
        self.done: bool = False
        # 统计信息
        self.chain_reward: list[float] = [0.0 for _ in range(self.total_slots)]
        self.mmec_reward: list[float] = [0.0 for _ in range(self.total_slots)]

        # Pre-Training
        self.pretrain_mode: bool = False
        self.resource_noise_mean: float = 0.0
        self.resource_noise_std: float = 0.0
        # self.sac_train_enabled: bool = True

    def set_pretrain_mode(self, mode: bool, mean: float=0.0, std: float=0.0) -> None:
        self.pretrain_mode = mode
        self.resource_noise_mean = mean
        self.resource_noise_std = std

    # def set_enable_sac(self, enabled: bool) -> None:
    #     self.sac_train_enabled = enabled

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """重置环境状态"""
        super().reset(seed=seed)

        self.chain_obs, _ = self.chain_env.reset()
        if not self.evaluate:
            all_generate_chain: list[list[int]] = self.chain_env.request_info.all_generate_chain
            self.final_g = GoalGenerator().get_goal(chain_list=all_generate_chain)

        self.image_control.reset()
        self.instance_control.reset()

        self.last_image_decision = np.zeros((self.esn_num, self.image_num), dtype=np.bool_)
        self.current_image_decision = np.zeros((self.esn_num, self.image_num), dtype=np.bool_)
        self.current_instance_activation = np.zeros((self.esn_num, self.image_num), dtype=np.float32)
        self.current_slot = 0
        self.done = False

        self.chain_reward = [0.0 for _ in range(self.total_slots)]
        self.mmec_reward = [0.0 for _ in range(self.total_slots)]

        obs: np.ndarray = np.zeros((self.esn_num, self.image_num, self.feature_dim), dtype=np.float32) if self.done else self._get_obs()
        return obs, {}


    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            info: dict[str, Any] = {"error": "Env already done"}
            return np.zeros((self.esn_num, self.image_num, self.feature_dim), dtype=np.float32), 0.0, True, False, info

        # ==========================================================
        # 强稀疏策略：动态 Top-K Masking
        # ==========================================================

        # --- A. 全局视角：Global Top-K (保证新热点能进入) ---
        # 假设我们只允许 Agent 关注流量最大的 4 种服务
        k_global = 3
        lookback_window = 5  # 从 50 减小到 10，快速响应变化
        start_idx: int = max(0, self.current_slot - lookback_window)
        recent_demands: NDArray[np.int64] = np.array(self.request_info.task_count_matrix[start_idx : self.current_slot + 1])

        # (image_num,) 统计每个镜像的总请求量
        demand_sum = np.sum(recent_demands, axis=0)

        # 2. [关键] 计算 Top-K 阈值
        # 如果有流量且镜像数足够
        if np.sum(demand_sum) > 0 and self.image_num > k_global:
            # 找到第 K 大的数值作为阈值
            threshold: np.float64 = np.partition(demand_sum, -k_global)[-k_global]
            # 必须 > 0 且 >= 阈值
            is_global_hot: NDArray[np.bool_] = ((demand_sum >= threshold) & (demand_sum > 1e-5))
        else:
            # 流量极低或初期，允许全局探索 (设为全1比较稳妥)
            is_global_hot = np.ones((self.image_num,), dtype=bool)

        # 广播到所有节点
        global_mask: NDArray[np.bool_] = np.tile(is_global_hot, (self.esn_num, 1)) # Shape: (N, M)

        # --- B. 本地视角：基于 InstanceControl 向量化数据的概率采样 ---
        k_local = 4  # 按权重选取的“大户” (保护核心资产)
        j_local = 3  # 随机选取的“散户” (清理僵尸进程)
        local_mask: NDArray[np.bool_] = np.zeros((self.esn_num, self.image_num), dtype=bool)

        # [Optimization] 直接从 InstanceControl 获取计数矩阵，无需遍历对象
        # state_decided shape: (esn_num, image_num, instance_num) -> sum axis 2 -> (esn_num, image_num)
        local_inst_counts: NDArray[np.float64] = np.sum(self.instance_control.state_decided, axis=2)

        # 针对每个 ESN 进行采样
        for i in range(self.esn_num):
            counts = local_inst_counts[i] # (image_num,)
            # 找出该节点已部署的服务索引 (count > 0)
            deployed_indices: NDArray[np.int64] = np.where(counts > 0)[0]
            num_deployed: int = len(deployed_indices)

            if num_deployed == 0:
                continue

            # 1. 计算概率分布 (Instance 越多，被选入"可控列表"的概率越大)
            active_counts = counts[deployed_indices].astype(np.float32)

            total = np.sum(active_counts)

            if total > 0:
                probs: NDArray[np.float64] = active_counts / total
            else:
                # 均等概率
                n_counts: int = len(active_counts)
                probs = np.ones(n_counts) / n_counts

            # 2. 阶段一：Weighted Top-K (按比例保护)
            k_actual: int = min(num_deployed, k_local)
            # replace=False 不放回抽样
            selected_k: NDArray[np.int64] = np.random.choice(deployed_indices, size=k_actual, replace=False, p=probs)

            # 3. 阶段二：Random J (均匀采样剩余部分)
            # 确保长尾小户有机会被选中，从而有机会被 Agent 输出负动作杀掉
            remaining_indices: NDArray[np.int64] = np.setdiff1d(deployed_indices, selected_k)
            j_actual: int = min(len(remaining_indices), j_local)

            if j_actual > 0:
                selected_j: NDArray[np.int64] = np.random.choice(remaining_indices, size=j_actual, replace=False)
                final_indices: NDArray[np.int64] = np.concatenate([selected_k, selected_j])
            else:
                final_indices = selected_k

            # 设置本地掩码
            local_mask[i, final_indices] = True

        # --- C. 融合掩码 ---
        # 设定一个探索概率，例如 5% 的概率允许 mask 之外的动作通过
        random_injection_prob: float = 0.05
        # 生成随机掩码矩阵
        random_mask: NDArray[np.bool_] = np.random.random((self.esn_num, self.image_num)) < random_injection_prob
        # 计算减少资源的mask
        close_mask: NDArray[np.bool_] = action < 0
        # 最终掩码 = (全局很火) OR (本地重点关注) OR (随机)
        final_mask_bool: NDArray[np.bool_] = global_mask | local_mask | random_mask | close_mask
        # 转换为 float 用于计算
        active_mask: NDArray[np.float32] = final_mask_bool.astype(np.float32)
        if not self.evaluate:
            action *= active_mask

        # ====================================================
        # ... (Pre-Training Logic same as before) ...
        if self.pretrain_mode:
            base_activation: float = self.resource_noise_mean * self.instance_num
            noise: np.ndarray = np.random.normal(loc=0.0, scale=self.resource_noise_std * self.instance_num, size=(self.esn_num, self.image_num))
            raw_target: np.ndarray = np.round(base_activation + noise)
            # [关键修复] 删除这行强制保留1个instance的代码！
            #deployed_mask: np.ndarray = self.current_image_decision
            #raw_target[deployed_mask] = np.maximum(raw_target[deployed_mask], 1.0)
            target_activation: np.ndarray = np.clip(raw_target, 0, self.instance_num).astype(np.float32)
            current_int: np.ndarray = np.round(self.current_instance_activation).astype(int)
            target_int: np.ndarray = target_activation.astype(int)
            instance_decisions: np.ndarray = (target_int - current_int).astype(int)
            self.current_instance_activation = target_activation
        else:
            # 使用概率性取整代替 np.round
            # action 是浮点数，例如 0.7
            # floor = 0, ceil = 1
            # 70% 概率取 1, 30% 概率取 0
            floor_val = np.floor(action)
            prob_ceil = action - floor_val
            rand_val = np.random.uniform(size=action.shape)

            # 只有在训练时使用概率取整，评估时使用 round
            if not self.evaluate:
                instance_decisions = floor_val + (rand_val < prob_ceil).astype(np.float32)
            else:
                instance_decisions = np.round(action)
            instance_decisions = instance_decisions.astype(int)
            np.clip(self.current_instance_activation + instance_decisions, 0, self.instance_num, out=self.current_instance_activation)

        has_demand: np.ndarray = self.current_instance_activation > 0
        is_interval_check: bool = (self.current_slot % self.image_decision_interval == 0)

        if is_interval_check:
            self.current_image_decision = has_demand
        else:
            need_emergency_deploy: np.ndarray = has_demand & (~self.current_image_decision)
            self.current_image_decision = self.current_image_decision | need_emergency_deploy

        self.image_control.update(decisions=self.current_image_decision.tolist())
        self.instance_control.update(decisions=instance_decisions.tolist())


        # ====================================================
        # ==================== [ChainEnv] ====================
        #sac_goal = np.sum(instance_decisions.astype(np.float32), axis=1, keepdims=True)
        # 含义：Manager 告诉 Worker：“这是当前的资源天花板，请基于此进行调度。”
        max_theoretical_capacity: int = self.instance_num * self.image_num
        if max_theoretical_capacity > 0:
            sac_goal: NDArray[np.float64] = np.sum(self.current_instance_activation, axis=1, keepdims=True) / max_theoretical_capacity
        else:
            sac_goal = np.zeros((self.esn_num, 1), dtype=np.float64)


        # 增加最大循环次数保护
        loop_guard = 0
        max_loops = 10000
        while self.chain_env.current_slot <= self.current_slot:
            loop_guard += 1
            if loop_guard > max_loops:
                print(f"[Error] MmecEnv stuck! ChainEnv slot {self.chain_env.current_slot} vs MmecEnv slot {self.current_slot}")
                # 强制打破循环，避免程序挂死
                self.chain_env.current_slot = self.current_slot + 1
                break
            if self.chain_env.done:
                break
            s_sac: np.ndarray = self.chain_env._get_obs() if self.chain_obs is None else self.chain_obs
            a_sac, r_sac, ns_sac, done_sac = self.sac_agent.step(
                step_num=self.sac_global_step,
                env=self.chain_env,
                s=s_sac,
                g=sac_goal,
                explore=not self.chain_env.evaluate
            )
            if not self.chain_env.evaluate:
                self.sac_agent.append(
                    s=s_sac, g=sac_goal, a=a_sac, n_s=ns_sac, n_g=sac_goal, r=r_sac, d=done_sac
                )
            self.chain_obs = ns_sac
            if self.current_slot < self.total_slots:
                self.chain_reward[self.current_slot] += r_sac
            self.sac_global_step += 1
        # ==================== [ChainEnv] ====================
        # ====================================================


        obs: np.ndarray = self._get_obs()
        reward: float = self._get_reward()/self.reward_scale_factor
        self.last_image_decision = self.current_image_decision
        truncated = False
        info = {
            "masked_action": action
        }

        self.mmec_reward[self.current_slot] += reward
        self.current_slot += 1
        if self.current_slot >= self.total_slots:
            self.done = True

        return obs, reward, self.done, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        [Vectorized Implementation]
        针对特定任务，观察所有ESN的状态。
        Returns:
            Shape: (N, M, F) # (esn_num, image_num, feature_dim)
            Features: [
                (0) ESN已用计算比例(%)              - 资源可用性
                (1) ESN已用内存比例(%)              - 资源可用性
                (2) ESN已用存储比例(%)              - 资源可用性
                (3) 所有Instance内存占用比例(%)      - 实际资源占用
                (4) 所有Instance的使用比例(%)       - 实际使用
                (5) image是否is_deployed           - 部署情况
                (6) 平均成功调用率                   - 服务质量
                (7) 过去所有slot历史平均请求率        - 流量趋势
                (8) Instance部署数(ln)              - 供给规模
            ]
        """
        if self.done:
            raise ValueError("MmecEnv is done, but still get_obs!")

        # 1. 初始化 obs (N, M, F)
        obs: np.ndarray = np.zeros((self.esn_num, self.image_num, self.feature_dim), dtype=np.float32)
        # 获取 1D 数组
        esn_core_rates: np.ndarray = np.array([n.core_use_rate for n in self.esn_nodes], dtype=np.float32)
        esn_mem_rates: np.ndarray = np.array([n.memory_use_rate for n in self.esn_nodes], dtype=np.float32)
        esn_sto_rates: np.ndarray = np.array([n.storage_use_rate for n in self.esn_nodes], dtype=np.float32)

        # shape: (esn_num, image_num, instance_num)
        state_mem: np.ndarray = self.instance_control.state_memory
        state_deployed: np.ndarray = self.instance_control.state_deployed # bool
        state_occupied: np.ndarray = self.instance_control.state_occupied
        state_calls: np.ndarray = self.instance_control.state_call_count
        state_success: np.ndarray = self.instance_control.state_success_count
        deployed_num = np.sum(state_deployed, axis=2).astype(np.float32) # (N, M)

        # 填充 Feature 0, 1, 2
        # reshape to (N, 1) for broadcasting to (N, M)
        obs[:, :, 0] = esn_core_rates[:, None]
        obs[:, :, 1] = esn_mem_rates[:, None]
        obs[:, :, 2] = esn_sto_rates[:, None]

        # (3) 所有Instance内存占用比例(%)
        total_mem_used_per_image = np.sum(state_mem, axis=2) # (N, M)
        obs[:, :, 3] = total_mem_used_per_image / self.esn_memory_capacities[:, None]

        # (4) 所有Instance的使用比例(%)
        occupied_num = np.sum(state_occupied, axis=2).astype(np.float32)
        safe_deployed_num = deployed_num.copy()
        safe_deployed_num[safe_deployed_num == 0] = 1.0
        obs[:, :, 4] = occupied_num / safe_deployed_num
        # 如果 deployed 为 0，usage 设为 0
        obs[:, :, 4][deployed_num == 0] = 0.0

        # (5) image是否is_deployed
        # 从 ImageControl 直接获取矩阵
        obs[:, :, 5] = self.image_control.state_deployed.astype(np.float32)

        # (6) 平均成功调用率
        # 原逻辑：average of (success/calls) for deployed instances
        # success_rate_per_inst = success / calls (if calls > 0 else 0.5)
        # a. 准备分母和分子
        inst_calls: np.ndarray = state_calls.astype(np.float32)
        inst_succ: np.ndarray = state_success.astype(np.float32)
        # b. 计算每个实例的成功率
        # 如果 calls > 0, rate = success/calls; else rate = 0.5
        inst_rates: np.ndarray = np.full_like(inst_calls, 0.5)
        has_calls: np.ndarray = inst_calls > 0
        np.divide(inst_succ, inst_calls, out=inst_rates, where=has_calls)
        # c. 只对 deployed 的实例求和
        deployed_rates_sum = np.sum(inst_rates * state_deployed, axis=2) # (N, M)
        # d. 求平均
        obs[:, :, 6] = deployed_rates_sum / safe_deployed_num
        obs[:, :, 6][deployed_num == 0] = 0.0

        # (7) 过去所有slot历史平均请求率
        # request_info.history_arrival_rate 是 list[list[float]] shape (Slots, M)
        # 获取当前 slot 的请求率 (1, M)
        current_request_rate: np.ndarray = np.array(self.request_info.history_arrival_rate[self.current_slot], dtype=np.float32)
        # Broadcast (1, M) to (N, M)
        obs[:, :, 7] = current_request_rate # Numpy 会自动 broadcast

        # (8) Instance部署数(ln)
        obs[:, :, 8] = np.log10(1 + deployed_num)

        return obs


    def _get_reward_impl(self, hyperpara_config: HyperparaConfig = para.hyperpara_config) -> float:
        # ======================= 参数提取 =======================
        alpha: float = hyperpara_config.get("performance_alpha", 0.4)   # Performance (主导)
        beta: float = hyperpara_config.get("fairness_beta", 0.1)        # Fairness (辅助)
        gamma: float = hyperpara_config.get("cost_gamma", 0.4)          # Cost (约束)
        delta: float = hyperpara_config.get("penalty_delta", 0.1)       #迁移惩罚
        #epsilon: float = hyperpara_config.get("penalty_lazy", 0.1)      # lazy惩罚

        #w_theoretical: float = hyperpara_config.get("performance_w_acceptance", 0.5)
        #w_actual: float = hyperpara_config.get("performance_w_sla", 0.5)

        # ======================= 1. 资源感知供给 (Resource-Aware Supply) =======================
        # 获取当前 slot 各个 image 的请求数量 [img_id, req_num]
        current_demand_counts: NDArray[np.float64] = np.array(self.chain_env.task_demand_count_slist[self.current_slot])
        # 找出有请求的 Image 索引
        active_img_indices: NDArray[np.int64] = np.where(current_demand_counts > 0)[0]
        # 记录每个img的得分
        supply_rewards_list: list[float] = []

        for img_idx in active_img_indices:
            demand: float = float(current_demand_counts[img_idx])

            # --- [Step A] 统计每个 ESN 节点上的可用实例数 ---
            # state_deployed shape: (esn_num, image_num, instance_num)
            # 针对特定 img_idx 切片 -> (esn_num, instance_num)
            relevant_deployed: NDArray[np.bool_] = self.instance_control.state_deployed[:, img_idx, :]
            relevant_occupied: NDArray[np.bool_] = self.instance_control.state_occupied[:, img_idx, :]

            # 有效可用 = 已部署 AND 未被占用
            # 按行求和，得到每个 ESN 节点上该 Image 的可用实例总数 -> Shape: (esn_num,)
            avail_per_node: NDArray[np.float64] = np.sum(relevant_deployed & (~relevant_occupied), axis=1)

            # --- [Step B] 核心逻辑：Top-N 集中度计算 ---
            # 1. 对可用数量降序排列 (大节点在前)
            sorted_avail: NDArray[np.float64] = np.sort(avail_per_node)[::-1]
            # 2. 计算理论上最少需要几个节点来承载这些请求
            # self.instance_num 是单个节点对单个 Image 的最大实例容量 (Candidate Num)
            # 例如：需求 25，单节点最大 10 -> 需要 ceil(2.5) = 3 个节点
            max_cap_per_node: int = self.instance_num
            ideal_node_num: int = int(np.ceil(demand / max_cap_per_node))
            # 3. 只取 Top-N 节点的和作为“有效供给”
            # 如果资源是碎片化的 (例如10个节点各部署1个)，sorted 前几项和会很小，effective_supply 也会很小
            effective_supply: float = np.sum(sorted_avail[:ideal_node_num])

            # --- [Step C] 映射到 Reward 函数 ---
            ratio: float = effective_supply / demand - 1.0
            # 这里的 Bias 可以去掉，因为 ratio=1.0 就是完美的物理意义
            r_img = 1.0 / (1.0 + np.exp(-10 * ratio))
            supply_rewards_list.append(r_img)

        # 对所有有请求的 Image 的 Reward 取平均
        r_supply: float = float(np.mean(supply_rewards_list)) if supply_rewards_list else 0.5

        # --- [Step D] 动态置信度修正 (Dynamic Confidence) ---
        total_occupied: float = max(0,np.sum(self.instance_control.state_occupied).astype(float))
        total_deployed: float = np.sum(self.instance_control.state_decided).astype(float)

        if r_supply > 0:

            # === 1. 计算执行健康度 (Execution Health) ===
            avg_timeout_ratio: float = self.chain_env.chain_timeout_ratio_slist[self.current_slot]
            # 进度 * (1 - 超时率)^2
            penalty_coeff: float = max(0.0, 1.0 - avg_timeout_ratio) ** 2
            execution_health_rate: float = penalty_coeff
            # === 2. 融合任务成功率 (Quality) ===
            # 最终的 Quality 既看"结果(Success)"，也看"过程(Health)"
            task_success_rate: float = max(0, min(1, self.chain_env.task_success_rate_slist[self.current_slot]))
            # === 3. 置信度计算 (Confidence calculation) ===
            confidence: float = 0.1 + 0.9 * (0.6 * task_success_rate + 0.4 * execution_health_rate)
            # === 4. 应用置信度 (Confidence Apply) ===
            r_supply *= confidence

        # 平均3个slot的chain_success_rate作为额外奖励
        start_idx: int = max(0, self.current_slot - 2)  # 包含当前 slot，共向前追溯最多 3 个
        past_chain_success_rates: list[float] = self.chain_env.chain_success_rate_slist[start_idx : self.current_slot + 1]
        r_chain: float = float(np.mean(past_chain_success_rates))

        # 最终 Performance 得分
        gain_perf: float = 0.9 * r_supply + 0.1 * r_chain

        # ======================= 2. Jain Fairness (公平性) =======================
        core_loads: NDArray[np.float32] = np.array([n.core_use_rate for n in self.esn_nodes], dtype=np.float32)
        memory_loads: NDArray[np.float32] = np.array([n.memory_use_rate for n in self.esn_nodes], dtype=np.float32)
        storage_loads: NDArray[np.float32]  = np.array([n.storage_use_rate for n in self.esn_nodes], dtype=np.float32)

        w_core, w_mem, w_sto = 0.3, 0.4, 0.3
        combined_loads: np.ndarray = w_core * core_loads + w_mem * memory_loads + w_sto * storage_loads
        gain_fair: float = min(1,max(0, self._jain_fairness(load_list=combined_loads)))


        # ======================= 3. Cost (成本) =======================
        # [改进方案] 复合成本结构：物理负载 + 部署规模 + 闲置浪费

        # A. 物理资源负载 (System Load)
        avg_load: float = float(np.mean(combined_loads))

        # B. 资源闲置浪费 (Resource Waste)
        idle_count: float = max(0, total_deployed - total_occupied)
        # 闲置率 = 闲置数 / (已部署数 + 保护分母)
        # 如果部署了10个，只有1个在跑，waste_ratio = 0.9 (惩罚大)
        if total_deployed > 0:
            raw_waste_ratio: float = (idle_count / total_deployed)
        else:
            raw_waste_ratio = 0.0

        # [关键修改] 使用指数级惩罚
        # 如果闲置率 10%: waste = 0.1^2 * 2.0 = 0.02 (微痛)
        # 如果闲置率 50%: waste = 0.5^2 * 2.0 = 0.5  (很痛)
        # 如果闲置率 90%: waste = 0.9^2 * 2.0 = 1.62 (剧痛)
        waste_ratio: float = (raw_waste_ratio ** 2) * 1.8

        # C. image部署变化 (Image Change)
        if self.last_image_decision.any():
            # 计算发生变化的实例数量 (Deployment Changes)
            # 论文中这叫: Reconfiguration Overhead / Migration Cost
            diff = np.sum(np.abs(self.current_image_decision ^ self.last_image_decision))
            # 归一化：除以总实例数或一个常数，防止数值过大爆炸
            change_cost: float = 5 * diff / (self.esn_num * self.image_num)
        else:
            change_cost = 0.0

        # === [最终 Cost 计算] ===
        w_load, w_change, w_waste = 0.3, 0.4, 0.3
        cost_res: float = w_load * avg_load + w_waste * waste_ratio + w_change * change_cost

        # ======================= 4. Migrate (惩罚) =======================
        migrate_num: int = max(0,self.chain_env.task_migrate_num_slist[self.current_slot])
        penalty_migrate: float = 0.0
        task_arrival_num: int = max(0,sum(self.request_info.task_count_matrix[self.current_slot]))
        if task_arrival_num > 0 and migrate_num > 0:
            penalty_migrate = min(1.0, migrate_num / task_arrival_num) * 2

        reward: float = alpha * gain_perf + beta * gain_fair - gamma * cost_res - delta * penalty_migrate

        return np.clip(reward, -5.0, 5.0)

    def _get_reward(self, hyperpara_config: HyperparaConfig = para.hyperpara_config) -> float:
        return self._get_reward_impl(hyperpara_config)

    def _jain_fairness(self, load_list: np.ndarray) -> float:
        """J = (∑xi)^2 / (n ⋅ ∑xi^2)"""
        sum_x: float = float(np.sum(load_list))
        sum_x_sq: float = float(np.sum(load_list ** 2))
        jain_fairness: float = (sum_x ** 2) / (self.esn_num * sum_x_sq) if sum_x_sq != 0 else 1.0
        return jain_fairness

    def _is_in_jupyter(self) -> bool:
        try:
            from IPython import get_ipython  # type: ignore
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True
            elif shell == 'TerminalInteractiveShell':
                return False
            else:
                return False
        except (NameError, ImportError, AttributeError):
            return False

    def render(self, mode="human")-> None:
        if self._is_in_jupyter():
            from IPython.display import clear_output
            clear_output(wait=True)

        print(f"\n[Slot {self.current_slot}] Node Status Monitor:")
        header = f"{'ID':<5} | {'Load (CPU/Mem/Sto)':<24} | {'Images':<8} | {'Instances':<10}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))

        for esn_id in range(self.esn_num):
            esn_node = self.esn_nodes[esn_id]
            load_str = f"{esn_node.core_use_rate:.2f} / {esn_node.memory_use_rate:.2f} / {esn_node.storage_use_rate:.2f}"
            deployed_image_num = np.sum(self.image_control.state_deployed[esn_id])
            deployed_instance_num = np.sum(self.instance_control.state_deployed[esn_id])

            print(f"ESN-{esn_id:<1} | {load_str:<24} | {deployed_image_num:<8} | {deployed_instance_num:<10}")

        print("-" * len(header))
        print("-" * len(header))
        self.chain_env.render()
        print("-" * len(header))

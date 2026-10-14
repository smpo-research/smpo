# smpo/base/user_request.py
from __future__ import annotations

from typing import Any, TypedDict

import numpy as np

from .config_para import BandConfig, ChainConfig, EsnConfig
from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .ms_type import MsType


class UserRequestTask(TypedDict):
    """上下行链路的数据类型"""
    user_id: int
    task_chain_id: int
    task_chain_list: list[int]
    task_chain_input: float
    task_chain_deadline: int


class AllRequestInfo:

    def __init__(
        self,
        user_num: int = para.user_config["user_num"],
        total_simu_slot=para.simu_config["total_time_slots"],
        seed  = para.user_config["random_seed"],
        arrival_rate: float|None = None,
        evaluate: bool = False
    ) -> None:
        if hasattr(self, "_initialized"):
            return
        self.evaluate: bool = evaluate
        current_seed = seed if evaluate else None
        self.seed = current_seed
        # 1. 初始化基本参数
        self.user_num: int = user_num
        self.image_num: int = para.ms_config["microservice_num"] # 确保获取微服务总类别数
        self.user_requests: list[UserRequest] = [UserRequest(user_id=i,seed=current_seed,arrival_rate=arrival_rate,evaluate=self.evaluate) for i in range(self.user_num)]
        self.all_generate_chain: list[list[int]] = RequestGenerator(seed=current_seed).all_generate_chain
        total_slot_chains, total_num, counts_mtx = self.__class__._process_request_data(
            requests=self.user_requests,
            total_simu_slot=total_simu_slot,
            num_task_types=self.image_num
        )
        self.total_slot_chains_info: list[list[UserRequestTask]] = total_slot_chains
        self.total_chain_num: int = total_num
        self.task_count_matrix: list[list[int]] = counts_mtx.astype(int).tolist()
        # 3. 基于构建好的矩阵计算统计率 (纯向量化计算，极快)
        history_arrival_rate, recent_arrival_rate = self.__class__._compute_rates_from_matrix(counts_matrix=counts_mtx)
        self.history_arrival_rate: list[list[float]] = history_arrival_rate
        self.recent_arrival_rate: list[list[float]] = recent_arrival_rate
        self._initialized: bool = True

    def reset(self)-> None:
        """
        重置请求数据。
        如果是评估模式 (evaluate_or_not=True)，保持数据不变。
        如果是训练模式，重新生成所有任务数据。
        """
        if self.evaluate:
            seed = self.seed
        else:
            seed = None

        # --- 训练模式：重新生成数据 ---

        # 1. 重置 RequestGenerator (清空全局计数器和列表)
        # 注意：训练模式下 seed 为 None
        RequestGenerator(seed=seed).reset()

        # 2. 重置每个用户生成新的任务
        for user in self.user_requests:
            user.reset()

        # 3. 重新获取生成的链列表 (因为 RequestGenerator 被重置并重新填充了)
        self.all_generate_chain = RequestGenerator(seed=seed).all_generate_chain

        # 4. 重新处理统计数据
        total_simu_slot = para.simu_config["total_time_slots"]
        total_slot_chains, total_num, counts_mtx = self.__class__._process_request_data(
            requests=self.user_requests,
            total_simu_slot=total_simu_slot,
            num_task_types=self.image_num
        )

        # 5. 更新属性
        self.total_slot_chains_info = total_slot_chains
        self.total_chain_num = total_num
        self.task_count_matrix = counts_mtx.astype(int).tolist()

        history_arrival_rate, recent_arrival_rate = self.__class__._compute_rates_from_matrix(counts_matrix=counts_mtx)
        self.history_arrival_rate = history_arrival_rate
        self.recent_arrival_rate = recent_arrival_rate



    @staticmethod
    def _process_request_data(
        requests: list[UserRequest],
        total_simu_slot: int,
        num_task_types: int
    ) -> tuple[list[list[UserRequestTask]], int, np.ndarray]:
        """
        [One-Pass 处理]
        遍历所有用户的稀疏字典，同时生成：
        1. 仿真用的列表结构 (total_chains_info)
        2. 统计用的计数矩阵 (counts_matrix)
        """
        # 输出容器初始化
        all_task_chains: list[list[UserRequestTask]] = [[] for _ in range(total_simu_slot)]
        total_chain_num = 0
        # 预分配矩阵 [Slots, Task_Types]
        counts_matrix: np.ndarray = np.zeros((total_simu_slot, num_task_types), dtype=np.float32)

        # 开始遍历所有用户
        for request in requests:
            # request.task_chain_list 是字典 {slot_id: task_info}
            for slot, chain_info in request.task_chains_dict.items():
                # 越界检查 (防止生成的任务超出仿真时长)
                if slot >= total_simu_slot:
                    continue
                # --- 任务 A: 填充仿真列表 ---
                all_task_chains[slot].append(chain_info)
                total_chain_num += 1
                # --- 任务 B: 填充计数矩阵 ---
                tids: list[int] = chain_info['task_chain_list']
                # 过滤并统计
                for tid in tids:
                    if 0 <= tid < num_task_types:
                        counts_matrix[slot, tid] += 1

        return all_task_chains, total_chain_num, counts_matrix

    @staticmethod
    def _compute_rates_from_matrix(
        counts_matrix: np.ndarray,
        window_size: int = 10
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        [纯向量化计算]
        基于已经构建好的矩阵计算历史平均和滑动窗口平均, shape: (slot,image_num)
        """
        total_slots = counts_matrix.shape[0]
        # 1. 历史平均 (Cumulative Average)
        cumulative_counts: np.ndarray = np.cumsum(counts_matrix, axis=0)
        # 构造分母 (避免除以0，虽不应该发生，但从1开始安全)
        time_steps: np.ndarray = np.arange(1, total_slots + 1, dtype=np.float32).reshape(-1, 1)
        history_rate: np.ndarray = cumulative_counts / time_steps
        # 2. 滑动窗口平均 (Sliding Window)
        # 逻辑: Sum[t] - Sum[t-window]
        window_sums: np.ndarray = cumulative_counts.copy()
        if total_slots > window_size:
            window_sums[window_size:] -= cumulative_counts[:-window_size]
        # 构造分母矩阵
        divisors: np.ndarray = np.full((total_slots, 1), window_size, dtype=np.float32)
        # 修正前 window_size 个 slot 的分母
        limit: int = min(window_size, total_slots)
        divisors[:limit, 0] = np.arange(1, limit + 1)
        recent_rate: np.ndarray = window_sums / divisors
        # 返回
        return history_rate.tolist(), recent_rate.tolist()

class UserRequest:
    _instances: dict[str, UserRequest] = {}

    def __new__(
        cls,
        user_id: int,
        seed: int|None,
        *args: Any,
        **kwargs: Any
    ) -> UserRequest:
        key: str = f"{user_id}_{seed}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        user_id: int,
        seed: int|None,
        evaluate: bool,
        arrival_rate: float|None = None,
        arrival_rate_base: float = para.user_config["request_rate_base"],
        arrival_rate_fluc: float = para.user_config["request_rate_fluctuate"],
    )->None:
        """
        初始化任务生成器
        :param arrival_rate: 泊松分布参数，单位时间内平均到达的任务数
        """
        if hasattr(self, 'user_id'):
            return
        self.evaluate: bool = evaluate
        self.random_generator: SharedRandomGenerator = SharedRandomGenerator(seed=seed)
        self.request_generator: RequestGenerator = RequestGenerator(seed=seed)
        self.user_id: int = user_id

        # 指定的到达率
        self.task_arrival_rate: float|None = arrival_rate
        # 到达率：
        # 1.直接指定 2.评估 3.训练
        if self.task_arrival_rate is not None:
            arrival_r: float = self.task_arrival_rate
        else:
            random_generator: SharedRandomGenerator = SharedRandomGenerator(seed = None)
            arrival_r = arrival_rate_base+(not self.evaluate)*random_generator.uniform(low=-arrival_rate_fluc,high=arrival_rate_fluc)
            arrival_r = max(arrival_r, arrival_rate_base-arrival_rate_fluc)

        # 单独设定某些chain的最小的deadline

        self.task_chains_dict: dict[int, UserRequestTask] = self._generate_task(arrival_r)
        self.task_chain_num: int = len(self.task_chains_dict) # 任务ID计数器

    def reset(
        self,
        arrival_rate_base: float = para.user_config["request_rate_base"],
        arrival_rate_fluc: float = para.user_config["request_rate_fluctuate"],
    )-> None:
        """
        重置并重新生成任务。
        注意：这会使用 self.random_generator 的当前状态继续生成新的随机数，
        从而在 seed=None 时产生新的随机序列。
        """
        if self.task_arrival_rate is not None:
            arrival_r: float = self.task_arrival_rate
        else:
            random_generator: SharedRandomGenerator = SharedRandomGenerator(seed = None)
            arrival_r = arrival_rate_base+(not self.evaluate)*random_generator.uniform(low=-arrival_rate_fluc,high=arrival_rate_fluc)
            arrival_r = max(arrival_r, arrival_rate_base-arrival_rate_fluc)
        self.task_chains_dict = self._generate_task(arrival_rate=arrival_r)
        self.task_chain_num = len(self.task_chains_dict)

    def _generate_task(
        self,
        arrival_rate: float,
        total_step: int = para.simu_config["total_time_slots"],
        delta: int = para.simu_config["time_slot"]
    )-> dict[int, UserRequestTask]:
        """
        生成所有时隙的任务List
        :param system_slot: 当前系统时隙序号（整数）
        :param delta: 时隙长度（毫秒）
        :return: 每个时隙的任务列表，每个任务是一个包含task_size和task_type的字典
        """
        # 计算每个时隙内期望到达的任务数
        expected_arrivals: float = arrival_rate * delta
        # 一次性生成所有时隙的到达数 (C语言级速度)
        arrival_counts = self.random_generator.poisson(lam=expected_arrivals, size=total_step)
        # 直接提取非零元素的索引 (即有任务到达的 slot)
        active_slots: np.ndarray = np.flatnonzero(a=arrival_counts)

        # --- 构建字典 ---
        task_dict: dict[int, UserRequestTask] = {}

        # 只遍历"有任务"的那些时隙 (稀疏遍历)
        for slot in active_slots:
            # 注意：slot 是 numpy int 类型
            slot_idx: int = int(slot)
            # 生成任务链
            task_dict[slot_idx] = self.request_generator.generate_task_chain(
                user_id=self.user_id
            )

        return task_dict



class RequestGenerator:
    _instances:dict[str, RequestGenerator] = {}

    def __new__(cls, seed:int|None=None, *args: Any, **kwargs: Any)-> RequestGenerator:
        key: str = f"{seed}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self,
        seed:int|None=None,
        all_ms_task_num: int = para.ms_config["microservice_num"]
    )-> None:
        if hasattr(self, '_initialized'):
            return
        self.random_generator: SharedRandomGenerator = SharedRandomGenerator(seed = seed)
        self.ms_type_list: list[int] = list(range(all_ms_task_num))
        s: float = 0.8 # 齐夫分布参数
        zeta: float = sum(1 / (i ** s) for i in range(1, all_ms_task_num + 1))
        self.zipf_probabilities: list[float] = [(1 / (i ** s)) / zeta for i in range(1, all_ms_task_num + 1)]
        # 仅仅记录信息
        self.task_chain_count: int = 0
        self.all_generate_chain_info: list[UserRequestTask] = []
        self.all_generate_chain: list[list[int]]= []

        self._initialized = True

    def reset(self)-> None:
        """清空生成的记录，准备新一轮生成"""
        self.task_chain_count = 0
        self.all_generate_chain_info.clear()
        self.all_generate_chain.clear()

    def generate_task_chain(
        self,
        user_id: int,
        task_deadline_slack: int = para.user_config["task_deadline_slack"],
        task_config: ChainConfig = para.chain_config,
        esn_config: EsnConfig = para.esn_config,
        band_config: BandConfig = para.band_config,
        delta: int = para.simu_config["time_slot"]
    )-> UserRequestTask:
        """生成微服务任务链类型（基于Zipf分布）"""
        # 解析字典
        task_size_base: float = task_config["task_size_base"]
        task_size_fluctuate: float = task_config["task_size_fluctuate"]
        task_deadline_fluctuate: float = task_config["task_deadline_fluctuate"]
        average_compute_capacity: float = esn_config["compute_capacity_base"]
        uplink_bandwidth = band_config["uplink_bandwidth"]
        downlink_bandwidth = band_config["downlink_bandwidth"]
        # 随机设置
        switch_task_deadline: bool = task_config["switch_task_deadline"]
        switch_task_size: bool = task_config["switch_task_size"]

        # 基于齐夫分布的chain采样
        ms_chain: list[int] = self._sample_from_zipf()
        # 生成任务大小（基于配置的基准值和波动范围）
        task_size: float = task_size_base + switch_task_size * self.random_generator.uniform(-1, 1) * task_size_fluctuate
        task_size = max(0, task_size) # 确保任务大小为正值
        # 估计平均必要计算时间
        accumulate_io_rate = 1
        average_task_deadline = 0
        for ms_task_i in ms_chain:
            average_task_deadline += task_size * accumulate_io_rate * MsType(ms_id=ms_task_i).compute_intensity / average_compute_capacity
            accumulate_io_rate *= MsType(ms_id=ms_task_i).io_ratio
        # 生成ms任务链
        aver_band = (uplink_bandwidth + downlink_bandwidth)/2 # Hz, # 10 * 10**6 带宽，大约10MHz
        base_transmission_buffer = task_size/(aver_band/2) * 1000  # 5s->5000ms
        slack_factor = 1.8 # 给 20% 的余量用于排队和迁移
        total_estimated_time = (average_task_deadline + base_transmission_buffer) * slack_factor
        task_deadline_time: float = total_estimated_time + switch_task_deadline * self.random_generator.uniform(-1, 1) * task_deadline_fluctuate
        base_deadline_slot: int = int(np.ceil(task_deadline_time / delta))
        task_deadline_slot: int = max(base_deadline_slot, task_deadline_slack)
        ms_task_chain: UserRequestTask = {
            "user_id": user_id,
            "task_chain_id": self.task_chain_count,
            "task_chain_list": ms_chain,
            "task_chain_input": task_size,
            "task_chain_deadline":task_deadline_slot
        }
        #更新记录
        self.task_chain_count += 1
        self.all_generate_chain_info.append(ms_task_chain)
        self.all_generate_chain.append(ms_task_chain["task_chain_list"])
        return ms_task_chain


    def _sample_from_zipf_optimized(
        self,
        task_config: ChainConfig = para.chain_config
    )-> list[int]:
        """
        [优化版] 使用预计算的概率列表进行采样
        """
        chain_num_base: int = task_config["chain_num_base"]
        chain_num_fluctuate: int = task_config["chain_num_fluctuate"]
        switch_chain_num: bool = task_config["switch_chain_num"]

        # 当波动为0时，直接偏移量为0
        offset = 0
        cnf_val = int(chain_num_fluctuate)
        if switch_chain_num and cnf_val > 0:
            offset = self.random_generator.randint(-cnf_val, cnf_val)
        # 生成任务链长度
        ms_chain_length: int = chain_num_base + offset
        ms_chain_length = max(1, ms_chain_length)

        # 采样任务链，尽量不重复
        need_replace: bool = ms_chain_length > len(self.ms_type_list)

        # 使用预计算的 self.zipf_probabilities
        # 注意：random_generator.choice 接受 1D array 或 list
        sampled_indices = self.random_generator.choice(
            self.ms_type_list,
            size=ms_chain_length,
            replace=need_replace,
            p=self.zipf_probabilities
        )

        generated_task_chain = [-1] + sampled_indices.tolist() + [-2]
        return generated_task_chain

    # 保留旧接口以防兼容性问题，指向优化版 (或者直接删除如果确认无其他调用)
    def _sample_from_zipf(self, *args, **kwargs)-> list[int]:
        return self._sample_from_zipf_optimized()

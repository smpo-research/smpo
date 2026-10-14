#smpo/base/ms_instance.py
from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from .config_para import ChainConfig, MsConfig
from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .esn import EdgeServerNode
from .ms_image import MsImage
from .ms_type import MsType

# 全局变量
MAX_WAIT_SLOT: int = para.simu_config["total_time_slots"]


class MsInstance:
    _instances: dict[str, MsInstance] = {}

    def __new__(
        cls,
        esn_id: int,
        image_id: int,
        instance_id: int,
        *args: Any,
        **kwargs: Any
    )-> MsInstance:
        key: str = f"{esn_id}_{image_id}_{instance_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        esn_id: int,
        image_id: int,
        instance_id: int,
        memory_scale: float = para.chain_config["memory_scale"],
    )->None:
        if hasattr(self,"instance_id"):
            return
        # 属性相关
        self.esn_id: int = esn_id
        self.image_id: int = image_id
        self.instance_id: int = instance_id
        self.ms_type: MsType = MsType(ms_id=image_id)
        self.memory_scale: float = memory_scale     #3M任务存入内存为30M
        # 所属esn和image
        self.esn: EdgeServerNode = EdgeServerNode(node_id=esn_id)
        self.image: MsImage = MsImage(esn_id=esn_id,image_id=image_id)
        # 部署决策相关
        self.current_decision: bool = False
        self.wait_slot: int = MAX_WAIT_SLOT
        self.allocated_memory: float = 0  # 随机分配
        # 运行相关
        self.core_id: int|None = None
        self.is_in_occupation: bool = False
        self.call_num: int = 0
        self.success_call_num: int = 0
        self.update_slot: int = 0

    # [Vectorization Helper] 获取控制器引用并同步数据
    def _sync_to_control(self, field: str, value: Any) -> None:
        """Helper to sync state to the global InstanceControl arrays"""
        # 延迟导入以避免循环依赖，且利用 Python 模块缓存，开销很小
        from .ms_control import InstanceControl
        ic: InstanceControl = InstanceControl()
        if field == 'memory':
            ic.state_memory[self.esn_id, self.image_id, self.instance_id] = value
        elif field == 'decided':
            ic.state_decided[self.esn_id, self.image_id, self.instance_id] = value
        elif field == 'deployed':
            ic.state_deployed[self.esn_id, self.image_id, self.instance_id] = value
        elif field == 'occupied':
            ic.state_occupied[self.esn_id, self.image_id, self.instance_id] = value
        elif field == 'calls':
            ic.state_call_count[self.esn_id, self.image_id, self.instance_id] = value
        elif field == 'success':
            ic.state_success_count[self.esn_id, self.image_id, self.instance_id] = value

    # 更新实例化决策
    def update(
        self,
        decision: bool,
        instantiation_delay: float = para.ms_config["instantiation_delay"],
        delta: int = para.simu_config["time_slot"]
    )->None:
        was_deployed: bool = self.current_decision
        is_now_deploying: bool = self.image.current_decision and decision
        # 情况1：从“未部署”变为“部署” → 首次实例化
        if not was_deployed and is_now_deploying:
            self.allocated_memory = self._random_alloc_memory() # 分配内存
            if self.allocated_memory > 0:
                # 内存分配成功
                self.esn.allocate_memory(memory=self.allocated_memory)  # 从esn分配内存
                # 记录决策信息
                self.wait_slot=int(np.ceil(instantiation_delay/delta))+self.image.wait_slot # 等待时长 = 实例化延迟 + 镜像部署等待时长
                self.success_call_num = 0   # 重新计数
                self.call_num = 0
                # [Sync]
                self._sync_to_control(field='calls', value=0)
                self._sync_to_control(field='success', value=0)
            else:
                # 内存分配失败
                self.wait_slot = MAX_WAIT_SLOT
                self.allocated_memory = 0
            self.current_decision = self.allocated_memory > 0

        # 情况2：正在部署中 → 持续减少等待计时
        elif was_deployed and is_now_deploying:
            self.wait_slot = max(0, self.wait_slot - 1)

        # 情况3：从“部署”变为“未部署” → 释放资源
        elif was_deployed and not is_now_deploying:
            # 这里不直接释放计算，由task延迟释放
            #self.release()
            # 释放内存
            self.esn.release_memory(memory=self.allocated_memory)
            self.allocated_memory = 0
            self.wait_slot = MAX_WAIT_SLOT
            self.current_decision = False
            self.success_call_num = 0
            self.call_num = 0
            # [Sync]
            self._sync_to_control(field='calls', value=0)
            self._sync_to_control(field='success', value=0)

        # 最终更新时刻
        self.update_slot += 1

        # [Sync] 实时同步关键状态
        self._sync_to_control(field='memory', value=self.allocated_memory)
        self._sync_to_control(field='decided', value=self.current_decision)
        self._sync_to_control(field='deployed', value=self.is_deployed)


    def reset(self)-> None:
        self.update(decision=False)
        self.release()
        self.update_slot = 0


    def _random_alloc_memory(
        self,
        ms_config: MsConfig=para.ms_config,
        task_config: ChainConfig=para.chain_config
    )-> float:
        """随机分配内存大小"""
        # 估计输入输出的上下界
        task_size_min: float = task_config["task_size_base"] - task_config["task_size_fluctuate"]
        task_size_max: float = task_config["task_size_base"] + task_config["task_size_fluctuate"]
        io_ratio_min: float = ms_config["io_ratio_base"] - ms_config["io_ratio_fluctuate"]
        io_ratio_max: float = ms_config["io_ratio_base"] + ms_config["io_ratio_fluctuate"]
        # 链路中的中间节点数量（不包括第一个节点）
        chain_num_max: int = max(0,task_config["chain_num_base"] + task_config["chain_num_fluctuate"] - 1)
        # 计算估计范围
        estimate_low_bound: float = task_size_min * io_ratio_min
        max_expansion = max(1.0, io_ratio_max ** chain_num_max)
        estimate_upper_bound: float = task_size_max * max_expansion
        # 完全随机均匀分配
        alloc_memory: float = SharedRandomGenerator(seed=ms_config["random_seed"]).uniform(low=estimate_low_bound, high=estimate_upper_bound)
        # 内存需求是task_size的memory_scale倍
        alloc_memory *= self.memory_scale
        return alloc_memory if self.esn.available_memory >= alloc_memory else 0

    @staticmethod
    def _alloc_core_id(node_id: int)-> int|None:
        """贪婪分配最大计算能力的core_id"""
        core_capacity_list: list[float] = EdgeServerNode(node_id).available_core
        greed_core_capacity: float  = max(core_capacity_list)
        return core_capacity_list.index(greed_core_capacity) if greed_core_capacity > 0 else None

    def allocate(self)-> None:
        if not self.current_decision or self.wait_slot > 0:
            return
        # 防止重复分配导致的状态覆盖
        if self.is_in_occupation:
            return
        self.core_id = self.__class__._alloc_core_id(node_id=self.esn_id)
        if self.core_id is not None:
            self.esn.allocate_core(core_id=self.core_id)
            self.is_in_occupation = True
            # [Sync]
            self._sync_to_control(field='occupied', value=True)

    def release(self)-> None:
        if self.core_id is not None:
            self.esn.release_core(core_id=self.core_id)
            self.core_id = None
        self.is_in_occupation = False
        # [Sync]
        self._sync_to_control(field='occupied', value=False)

    @property
    def allocated_compute_capacity(self) -> float|None:
        """根据decided_esn_id和core_id获得分配的计算能力"""
        if self.core_id is not None:
            return EdgeServerNode(node_id=self.esn_id).core_capacity_list[self.core_id]

    @property
    def success_call_rate(self) -> float:
        return self.success_call_num/self.call_num if self.call_num != 0 else 0.5

    def record_call(self, success_call:bool)-> None:
        self.call_num += 1
        self._sync_to_control(field='calls', value=self.call_num)
        if success_call:
            self.success_call_num += 1
            self._sync_to_control(field='success', value=self.success_call_num)


    @property
    def is_deployed(self) -> bool:
        """判断当前instance是否已成功部署并生效"""
        return self.current_decision and self.image.current_decision and self.wait_slot <= 0



    @override
    def __repr__(self) -> str:
            return (f"MsInstance(ms_id={self.image.image_id})")

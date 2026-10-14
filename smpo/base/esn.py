# SMPO/base/esn.py
from __future__ import annotations

from typing import Any

from typing_extensions import override

from .config_para import EsnConfig
from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .esnc import EsnCoordinates
from .user_mobility import UserMobility


class EdgeServerNode:
    """
    Edge Server Node (ESN) in the Mobile-Multi-access Edge Computing (MMEC)
    """
    _instances: dict[str, EdgeServerNode] = {}

    def __new__(cls, node_id:int, *args:Any, **kwargs:Any)-> EdgeServerNode:
        key:str = f"{node_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        node_id: int, # node_id索引必须从0开始
        location: tuple[float, float]|None = None,
        esn_config: EsnConfig  = para.esn_config,
        inst_num = para.ms_config["instance_candidate_num"]
    )-> None:
        if hasattr(self, 'node_id'):
            return
        # 解析字典
        core_num_base: int = esn_config["core_num_base"]
        core_num_fluc: int = esn_config["core_num_fluctuate"]
        compute_capacity_base: float = esn_config["compute_capacity_base"]
        compute_capacity_fluc: float = esn_config["compute_capacity_fluctuate"]
        memory_capacity_per_inst: float = esn_config["memory_capacity_per_inst"]
        storage_capacity_per_inst:float = esn_config["storage_capacity_per_inst"]
        random_generator:SharedRandomGenerator = SharedRandomGenerator(seed=esn_config["random_seed"])

        switch_core_num: bool = esn_config["switch_core_num"]
        switch_compute_capacity: bool = esn_config["switch_compute_capacity"]

        # esn的node_id
        self.node_id: int = node_id
        # 地理位置
        self.location: tuple[float,float] = location if location is not None else EsnCoordinates().get_coordinate(node_id)
        # 覆盖的用户列表
        self.cover_user_slist:list[list[int]] = self.__class__._get_cover_user_slist(node_id)
        # 计算核心数
        cnf_val: int = int(core_num_fluc)
        if switch_core_num and cnf_val > 0:
            self.core_num: int = core_num_base + random_generator.randint(low=-cnf_val, high=cnf_val)
        else:
            self.core_num = core_num_base
        self.core_num = max(1, self.core_num)
        # 计算核心能力list
        ccf_val: int = int(compute_capacity_fluc)
        if switch_compute_capacity and ccf_val > 0: # 浮点数判断
            offsets = random_generator.randint(low=-ccf_val, high=ccf_val, size=self.core_num)
            self.core_capacity_list: list[float] = [compute_capacity_base + x for x in offsets]
        else:
            self.core_capacity_list = [compute_capacity_base] * self.core_num
        # 设置最小算力
        self.core_capacity_list = [x if x> 0 else compute_capacity_base for x in self.core_capacity_list]

        # 核心使用情况
        self.core_useage: list[bool] = [False] * self.core_num
        # 内存总大小
        self.memory_capacity: float = inst_num * memory_capacity_per_inst
        # 剩余的内存大小
        self.memory_useage_surplus: float = self.memory_capacity
        # 存储
        self.storage_capacity: float = inst_num * storage_capacity_per_inst
        # 剩余的存储大小
        self.storage_useage_surplus: float = self.storage_capacity


    def reset(self)->None:
        self.core_useage = [False] * self.core_num
        self.memory_useage_surplus = self.memory_capacity
        self.storage_useage_surplus = self.storage_capacity


    @staticmethod
    def _get_cover_user_slist(
        node_id: int,
        total_user_num: int = para.user_config["user_num"],
        total_time_slots: int = para.simu_config["total_time_slots"]
    )-> list[list[int]]:
        """计算所有时隙esn覆盖的用户列表"""
        cover_user_slist: list[list[int]] = [[] for _ in range(total_time_slots)]
        # 按用户遍历，避免重复访问UserMobility
        for i in range(total_user_num):
            user_bs_indices: list[int] = UserMobility(user_id=i).connected_bs_idx
            for t in range(total_time_slots):
                if user_bs_indices[t] == node_id:
                    cover_user_slist[t].append(i)
        return cover_user_slist



    def allocate_core(self, core_id: int)-> None:
        if core_id > self.core_num - 1:
            raise ValueError(f"core_id({core_id}) is out of range(0,{self.core_num})")
        self.core_useage[core_id] = True

    def release_core(self, core_id: int)-> None:
        if core_id > self.core_num - 1:
            raise ValueError(f"core_id({core_id}) is out of range(0,{self.core_num})")
        self.core_useage[core_id] = False

    def allocate_memory(self, memory: float)-> None:
        self.memory_useage_surplus = max(0, self.memory_useage_surplus - memory)

    def release_memory(self, memory: float)-> None:
        self.memory_useage_surplus = min(self.memory_capacity, self.memory_useage_surplus + memory)

    def allocate_storage(self, storage: float)-> None:
        self.storage_useage_surplus = max(0, self.storage_useage_surplus - storage)

    def release_storage(self, storage: float)-> None:
        self.storage_useage_surplus = min(self.storage_capacity, self.storage_useage_surplus + storage)

    @property
    def available_core(self) -> list[float]:
         return [x if not flag else 0 for x, flag in zip(self.core_capacity_list, self.core_useage)]

    @property
    def available_memory(self) -> float:
        return self.memory_useage_surplus
    @property
    def available_storage(self) -> float:
        return self.storage_useage_surplus
    @override
    def __repr__(self)-> str:
        return (f"EdgeServerNode(node_id={self.node_id})\n")

    @property
    def core_use_rate(self)-> float:
        total_capacity: float = sum(self.core_capacity_list)
        used_capacity: float = sum(
            cap for cap, used in zip(self.core_capacity_list, self.core_useage)
            if used
        )
        return used_capacity / total_capacity if total_capacity > 0 else 0.0

    @property
    def memory_use_rate(self)-> float:
        memory_used: float = max(0, self.memory_capacity - self.memory_useage_surplus)
        return memory_used / self.memory_capacity if self.memory_capacity > 0 else 0.0

    @property
    def storage_use_rate(self)-> float:
        storage_used: float = max(0, self.storage_capacity - self.storage_useage_surplus)
        return storage_used / self.storage_capacity if self.storage_capacity > 0 else 0.0

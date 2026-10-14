# smpo/base/task_compute.py
import numpy as np

from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .esng import EsnGraph
from .ms_instance import MsInstance
from .ms_type import MsType


class MsComputeTask:
    def __init__(
        self,
        user_id: int,
        chain_id: int,
        task_id: int,
        task_size: float,
        task_deadline: int,
        memory_scale: float = para.chain_config["memory_scale"]
    ) -> None:
        # 任务属性
        self.user_id: int = user_id
        self.chain_id: int = chain_id
        self.task_id: int = task_id
        self.io_ratio: float = MsType(ms_id=task_id).io_ratio
        self.task_size: float = task_size
        self.task_deadline: int = task_deadline
        self.memory_scale: float = memory_scale     #内存使用是task_size的memory_scale倍

        # 初始化计算任务
        self.compute_intensity: float = MsType(ms_id=task_id).compute_intensity
        self.remain_compute_cycle: float = task_size * self.compute_intensity    # 计算剩余计算周期, 单位cycle

        # 任务资源分配
        self.located_esn_id: int|None = None
        self.decided_esn_id: int|None = None
        self.instance: MsInstance|None = None


        # 迁移等待时间
        self.migrate_wait_slot: int = 0

        # 任务执行时间管理
        self.start_slot: int = -1
        self.complete_slot: int = -1
        self.migrate_reduced_slot: list[int] = []
        self.update_slot: int = -1
        self.complete_on_time: bool|None = None

    @property
    def allocated_compute_capacity(self) -> float:
        return self.instance.allocated_compute_capacity if self.instance and self.instance.allocated_compute_capacity is not None else 0.0

    @property
    def allocated_memory(self) -> float:
        return self.instance.allocated_memory if self.instance is not None else 0

    @property
    def task_mid_size(self) -> float:
        progress_rate: float = 1 - self.remain_compute_cycle / (self.task_size * self.compute_intensity)
        mid_task_size: float = self.task_size * (1 + (self.io_ratio - 1) * progress_rate)
        return mid_task_size

    def start(self, start_slot: int, located_esn_id: int) -> None:
        self.start_slot = start_slot
        self.update_slot = start_slot
        self.located_esn_id = located_esn_id

    def update(
        self,
        esn_id: int,

    ) -> None:
        # 1) 根据决策分配instance
        self._alloc_instance(node_id=esn_id)
        # 2) 根据决策计算迁移
        self._migrate(node_id=esn_id)
        # 3) 如果前置准备完成，则开始计算
        self._compute()
        # 4) 更新时隙
        self.update_slot += 1

    def _compute(
        self,
        delta: int = para.simu_config["time_slot"]
    )-> None:
        # 如果当前处于迁移等待期内，跳过计算
        if self.migrate_wait_slot > 0:
            return
        # 如果任务之前就已经完成了，直接返回
        if self.is_completed:
            return
        # --- 先扣除计算量 ---
        compute_capacity: float = max(0, self.allocated_compute_capacity)
        self.remain_compute_cycle -= compute_capacity * delta
        # 如果任务已完成，无需继续计算
        if self.remain_compute_cycle <= 1e-9:
            self.remain_compute_cycle = 0 # 修正为0
            # 仅在首次计算完成时设置
            if self.complete_on_time is None:
                # 记录时间戳
                self.complete_slot = self.update_slot
                compute_spend_slot: int = self.complete_slot - self.start_slot
                self.complete_on_time = compute_spend_slot <= self.task_deadline
                # 释放资源
                if self.instance is not None:
                    self.instance.record_call(success_call=self.complete_on_time)
                    self.instance.release()
                    self.instance = None  # 防止重复释放


    def _migrate(self, node_id: int)-> None:
        """判断决策是否造成迁移，计算迁移所需时长"""
        # 决策更改，或首次决策
        if self.decided_esn_id != node_id:
            # 计算迁移时延
            assert self.located_esn_id is not None
            self.migrate_wait_slot = self.__class__._migrate_wait_slot(source_idx=self.located_esn_id,target_idx=node_id,task_size=self.task_mid_size)
        # 更新决策
        self.decided_esn_id = node_id
        # 迁移中：等待倒计时完成
        if self.migrate_wait_slot > 0:
            self.migrate_wait_slot = max(0,self.migrate_wait_slot - 1)
        # 迁移完成：更新实际位置
        else:
            self.located_esn_id = node_id

    def _alloc_instance(
        self,
        node_id: int,
        instance_num: int = para.ms_config["instance_candidate_num"],
        random_seed = para.ms_config["random_seed"],
        delta: int = para.simu_config["time_slot"]
    ) -> None:
        """根据节点 ID 随机分配一个满足内存需求的实例，更新迁移节省的slot记录。(此处self.decided_esn_id是上一个未更新的决策)"""
        # 如果节点未变更，直接归零节省槽位，无需分配
        if (self.decided_esn_id is not None and
            self.decided_esn_id == node_id and
            self.instance is not None and
            self.allocated_compute_capacity > 0):
            # 仅仅记录
            self.migrate_reduced_slot.append(0)
            return
        # 保存旧实例的计算能力，用于后续迁移节省计算
        pre_compute_capacity: float | None = self.instance.allocated_compute_capacity if self.instance else None
        # 释放旧实例（如果存在）
        if self.instance is not None:
            self.instance.release()
            self.instance = None
        # 生成候选实例的内存列表，并筛选出内存足够的实例索引
        random_generator: SharedRandomGenerator = SharedRandomGenerator(seed=random_seed)
        # 1. 获取所有候选实例对象
        candidates = [MsInstance(esn_id=node_id, image_id=self.task_id, instance_id=i) for i in range(instance_num)]
        # 2. 筛选：必须同时满足 [内存足够] AND [未被占用]
        valid_indice_list: list[int] = [
            i for i, inst in enumerate(candidates)
            if inst.allocated_memory >= self.task_mid_size * self.memory_scale and not inst.is_in_occupation
        ]
        # 随机选择一个有效实例，若无则保持 self.instance 为 None
        alloc_instance_id: int|None = random_generator.choice(valid_indice_list) if valid_indice_list else None
        # 分配新实例
        if alloc_instance_id is not None:
            temp_inst: MsInstance = MsInstance(esn_id=node_id, image_id=self.task_id, instance_id=alloc_instance_id)
            if temp_inst.is_deployed and not temp_inst.is_in_occupation:
                self.instance = temp_inst
                self.instance.allocate()
            else:
                self.instance = None
        # 计算本次迁移带来的节省时间槽
        new_compute_capacity: float | None = self.instance.allocated_compute_capacity if self.instance else None
        # 仅对迁移部分进行计算
        if pre_compute_capacity is not None and new_compute_capacity is not None:
            assert self.located_esn_id is not None and self.decided_esn_id is not None
            # 计算因计算能力提升而减少的执行时间槽（单位：时间片）
            shortened_slot: int = int(np.ceil(self.remain_compute_cycle * (1 / pre_compute_capacity - 1 / new_compute_capacity) / delta))
            # 获取原计划迁移等待时间与当前实际迁移等待时间的差值
            pre_migrat_wait_slot: int = self.__class__._migrate_wait_slot(
                source_idx=self.located_esn_id,
                target_idx=self.decided_esn_id,
                task_size=self.task_mid_size
            )
            new_migrat_wait_slot: int = self.__class__._migrate_wait_slot(
                source_idx=self.located_esn_id,
                target_idx=node_id,
                task_size=self.task_mid_size
            )
            # 总节省 = 执行时间缩短 + 原计划等待减少 - 新等待增加
            migrate_reduced_slot: int = shortened_slot + pre_migrat_wait_slot - new_migrat_wait_slot
            self.migrate_reduced_slot.append(migrate_reduced_slot)
        else:
            # 无效分配，或初次分配
            self.migrate_reduced_slot.append(0)

    @staticmethod
    def _migrate_wait_slot(
        source_idx: int,
        target_idx: int,
        task_size: float,
        down_time: float = para.ms_config["microservice_downtime"],
        delta: int = para.simu_config["time_slot"]
    )-> int:
        if source_idx == target_idx:
            return 0
        transmit_cost_weight: float = EsnGraph().min_distance_delay_cost(source_node_idx=source_idx, target_node_idx=target_idx)
        # 计算总迁移时间（含服务中断时间）
        migrat_time: float = task_size * transmit_cost_weight  # 单位ms
        migrate_wait_slot: int = int(np.ceil((migrat_time + down_time) / delta)) #单位slot
        return migrate_wait_slot


    @property
    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.complete_slot != -1

    @property
    def remain_deadline(self)-> int:
        return  max(0, self.task_deadline-(self.update_slot-self.start_slot))

    @property
    def is_on_schedule(self, delta: int = para.simu_config["time_slot"])-> bool:
        '''预计是否能按时'''
        if self.allocated_compute_capacity > 0:
            expect_complete_slot: int = self.migrate_wait_slot + int(np.ceil(self.remain_compute_cycle/self.allocated_compute_capacity/delta))
            return self.update_slot + expect_complete_slot <= self.start_slot + self.task_deadline
        else:
            return False

    @property
    def expect_exec_deadline_ratio(self, delta: int = para.simu_config["time_slot"])-> float:
        '''在当前决策下从零开始是否能按时'''
        if self.allocated_compute_capacity > 0:
            # 不考虑迁移时延，因为无法保证初始节点
            expect_complete_slot: int = int(np.ceil(self.task_size*self.compute_intensity/self.allocated_compute_capacity/delta))
            if self.task_deadline > 0:
                return expect_complete_slot / self.task_deadline
            else:
                return 2.0 if expect_complete_slot > 0 else 0.0
        else:
            return 2.0

# smpo/base/task_chain.py
from typing import Any, cast

import numpy as np

from .ms_type import MsType
from .task_compute import MsComputeTask
from .task_transmit import MsDownlinkTask, MsUplinkTask
from .user_request import UserRequestTask


class MsChain:
    """微服务任务链类, 一个完整的任务链：上行传输 -> 微服务计算序列 -> 下行传输"""
    def __init__(
        self,
        user_request_task: UserRequestTask
    ) -> None:
        self.user_id: int = user_request_task['user_id']
        self.chain_id: int = user_request_task['task_chain_id']
        self.chain_list: list[MsUplinkTask|MsComputeTask|MsDownlinkTask] = []
        self.chain_id_list: list[int] = user_request_task['task_chain_list']
        self.chain_num: int = len(self.chain_id_list)
        self.chain_input: float = user_request_task['task_chain_input']
        random_deadline: int = user_request_task['task_chain_deadline']
        self.chain_deadline: int = 20 if random_deadline <= 20 else random_deadline
        self.current_task_idx: int = 0
        # 计算每个task的input
        io_ratio_temp: float = 1.0
        io_ratio_list: list[float] = [1.0,1.0]
        for temp_i in range(2,self.chain_num):
            io_ratio_temp *= MsType(ms_id=self.chain_id_list[temp_i-1]).io_ratio
            io_ratio_list.append(io_ratio_temp)
        task_inputs: list[float] = [self.chain_input * temp_i for temp_i in io_ratio_list]
        # 计算每个task的deadline
        average_deadline: int = int(np.ceil(self.chain_deadline / self.chain_num))
        # 平均分配
        task_deadlines: list[int] = [average_deadline for _ in range(self.chain_num)]
        # 比例分配
        # # compute部分
        # compute_cycles: list[float] = [task_inputs[i+1]*MsType(ms_id=self.chain_id_list[i+1]).compute_intensity for i in range(self.chain_num-2)]
        # compute_cycle_ratios: list[float] = [i / sum(compute_cycles) for i in compute_cycles]
        # total_compute_deadline: int = average_deadline * (self.chain_num - 2)
        # compute_task_deadlines: list[int] = [round(number=total_compute_deadline*i) for i in compute_cycle_ratios]
        # # transmit部分
        # transmit_size_ratios: list[float] = [task_inputs[0]/(task_inputs[0]+task_inputs[-1]),task_inputs[-1]/(task_inputs[0]+task_inputs[-1])]
        # total_transmit_deadline: int =  2 * average_deadline
        # transmit_task_deadline: list[int] = [round(number=total_transmit_deadline*i) for i in transmit_size_ratios]
        # task_deadlines: list[int] = [transmit_task_deadline[0]]+compute_task_deadlines+[transmit_task_deadline[-1]]

        # 归一化以确保不超过总 Deadline
        current_total = sum(task_deadlines)
        if current_total > self.chain_deadline:
            # 简单的按比例缩减，或者直接砍掉多余的
            diff = current_total - self.chain_deadline
            # 从最长的任务里扣除，或者平均扣
            # 这里简单粗暴地从每个任务里扣 1，直到扣完
            while diff > 0:
                for i in range(len(task_deadlines)):
                    if task_deadlines[i] > 1:
                        task_deadlines[i] -= 1
                        diff -= 1
                        if diff == 0:
                            break

        # 创建上行任务
        uplink_task: MsUplinkTask = MsUplinkTask(
            user_id=self.user_id,
            chain_id=self.chain_id,
            task_size=task_inputs[0],
            task_deadline=task_deadlines[0]
        )
        self.chain_list.append(uplink_task)

        # 创建计算任务序列
        for temp_i, temp_chain_id in enumerate(self.chain_id_list[1:-1],1):
            compute_task: MsComputeTask = MsComputeTask(
                user_id=self.user_id,
                chain_id=self.chain_id,
                task_id=temp_chain_id,
                task_size=task_inputs[temp_i],
                task_deadline=task_deadlines[temp_i]
            )
            self.chain_list.append(compute_task)

        # 创建下行任务（传输最后一个计算任务的输出）
        downlink_task: MsDownlinkTask = MsDownlinkTask(
            user_id=self.user_id,
            chain_id=self.chain_id,
            task_size=task_inputs[-1],
            task_deadline=task_deadlines[-1]
        )
        self.chain_list.append(downlink_task)

        # 任务链执行状态
        self.start_slot: int = -1
        self.update_slot: int = -1
        self.complete_slot: int = -1
        self.complete_on_time: bool | None = None

    def start(self, start_slot: int) -> None:
        """启动任务链"""
        if self.start_slot != -1:
            raise RuntimeError("The task chain has already been started!")
        self.start_slot = start_slot
        self.update_slot = start_slot
        assert isinstance(self.chain_list[0], MsUplinkTask)
        self.chain_list[0].start(start_slot=start_slot)

    def update(self, esn_id: int|None = None) -> None:
        """更新任务链状态"""
        if self.current_task is None:
            return
        # 根据任务类型调用相应的更新方法
        if isinstance(self.current_task, MsComputeTask):
            if esn_id is None:
                raise ValueError(f"compute task {self.current_task.task_id} need parameter esn_id!")
            self.current_task.update(esn_id)
        else:
            # 上行/下行任务不需要esn_id
            self.current_task.update()
        # 检查当前任务是否完成
        if self.current_task.is_completed:
            # 如果是最后一个任务，标记任务链完成
            if self.current_task_idx == len(self.chain_list) - 1:
                self.complete_slot = self.update_slot
                total_slots: int = self.complete_slot - self.start_slot
                self.complete_on_time = total_slots <= self.chain_deadline
            else:
                # 启动下一个任务
                next_task_idx: int = self.current_task_idx + 1
                next_task: MsComputeTask|MsDownlinkTask = cast(MsComputeTask|MsDownlinkTask,self.chain_list[next_task_idx])
                # 只有计算任务需要确定ESN ID
                if isinstance(next_task, MsComputeTask):
                    if isinstance(self.current_task, MsUplinkTask):
                        correspond_esn_id: int = self.current_task.nearest_esn_id
                    elif isinstance(self.current_task, MsComputeTask):
                        assert(self.current_task.located_esn_id is not None)
                        correspond_esn_id = self.current_task.located_esn_id
                    else:
                        # 对于DownlinkTask不应该走到这里
                        raise RuntimeError("Cannot start compute task after downlink task!")
                    # 启动计算任务
                    next_task.start(start_slot=self.update_slot, located_esn_id=correspond_esn_id)
                elif isinstance(next_task, MsDownlinkTask):
                    # 启动下行任务
                    next_task.start(start_slot=self.update_slot)
                # 更新current_task_idx
                self.current_task_idx = next_task_idx
        # 最后更新update_slot
        self.update_slot += 1

    @property
    def is_completed(self) -> bool:
        """检查任务链是否完成"""
        return self.complete_slot != -1

    @property
    def current_task(self)-> MsUplinkTask|MsComputeTask|MsDownlinkTask|None:
        """获取当前正在执行的任务"""
        return self.chain_list[self.current_task_idx] if not self.is_completed else None

    @property
    def schedule_rate(self) -> float:
        """计算任务链的整体进度(0~1之间)，基于已完成的任务数量比例"""
        if self.is_completed:
            # 任务链已完成，返回100%
            return 1.0
        # 当前任务索引从0开始，加1表示已经执行到第几个任务
        # 例如：第0个任务进行中 → 1/5 = 0.2
        completed_tasks: int = self.current_task_idx + 1
        total_tasks: int = len(self.chain_list)
        # 已完成的任务数除以总任务数
        return completed_tasks / total_tasks



    @property
    def is_current_task_compute(self) -> bool:
        """检查当前任务是否为计算任务（需要ESN决策）"""
        return isinstance(self.current_task, MsComputeTask) if not self.is_completed else False

    @property
    def current_compute_esn_id(self) -> int | None:
        """获取当前计算任务决策的esn_id"""
        if isinstance(self.current_task, MsComputeTask):
            return self.current_task.decided_esn_id

    @property
    def timeout_status(self)-> int:
        """当前状态已完成的task的已超时情况"""
        timeout: int = 0
        for task_id in range(self.current_task_idx):
            temp_task: MsUplinkTask|MsComputeTask|MsDownlinkTask = self.chain_list[task_id]
            if temp_task.complete_slot == -1:
                continue  # 跳过未完成任务
            timeout += (temp_task.complete_slot - temp_task.start_slot) - temp_task.task_deadline
        return max(0, timeout)

    def get_status(self) -> dict[str, Any]:
        """获取任务链状态摘要"""
        return {
            'chain_id': self.chain_id,
            'is_completed': self.is_completed,
            'complete_on_time': self.complete_on_time,
            'current_task_idx': self.current_task_idx,
            'current_task_type': type(self.current_task).__name__ if self.current_task else None,
            'total_tasks': len(self.chain_list),
            'current_compute_esn_id': self.current_compute_esn_id
        }

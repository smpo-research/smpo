# smpo/base/user_link.py
from __future__ import annotations

import heapq
from typing import Any, TypedDict

from .config_para import Parameter as para
from .esnb import BandAllocEvent, BdEvent, BdEventBus, BdEventType, LinkTransEvent
from .user_band import UserBand


class linkTask(TypedDict):
    """上下行链路的数据类型"""
    chain_id: int
    task_size: float
    task_deadline: int
    band_allocate: float

class UserLink:
    _instances: dict[str, UserLink] = {}

    def __new__(cls, user_id: int, *args: Any, **kwargs: Any)-> UserLink:
        key: str = f"{user_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, user_id: int)-> None:
        if hasattr(self, "user_id"):
            return
        self.user_id: int = user_id
        self.band: UserBand = UserBand(user_id)
        self.band_event_bus: BdEventBus = BdEventBus()
        # 上下行链路
        self.uplink_task: dict[str, linkTask] = {}
        self.uplink_min_heap: list[int] = []
        self.downlink_task: dict[str, linkTask] = {}
        self.downlink_min_heap: list[int] = []
        # 记录完成任务
        self.complete_uplink_task: list[int] = []
        self.complete_downlink_task: list[int] = []
        # 订阅带宽分配
        self.band_event_bus.subscribe(event_type=BdEventType.BANDWIDTH_ALLOCATE, callback=self._get_bandwidth_allocation_and_transmit)
        # 更新控制
        self.update_slot: int = -1
        self.update_uplink: bool = False
        self.update_downlink: bool = False

    def reset(self)-> None:
        self.uplink_task = {}
        self.uplink_min_heap = []
        self.downlink_task = {}
        self.downlink_min_heap = []
        self.complete_uplink_task = []
        self.complete_downlink_task = []
        self.update_slot = -1
        self.update_uplink = False
        self.update_downlink = False


    def add_uplink_task(
        self,
        chain_id: int,
        task_size: float,
        task_deadline: int
    )-> None:
        """任务初始化时添加新任务数据"""
        # 添加任务
        self.uplink_task[f"{chain_id}"] = {
            "chain_id":chain_id,
            "task_size":task_size,
            "task_deadline":task_deadline,
            "band_allocate":0.0
        }

    def add_downlink_task(
        self,
        chain_id: int,
        task_size: float,
        task_deadline: int
    )-> None:
        """任务初始化时添加新任务数据"""
        self.downlink_task[f"{chain_id}"]={
            "chain_id":chain_id,
            "task_size":task_size,
            "task_deadline":task_deadline,
            "band_allocate":0.0
        }

    def uplink_start(self, chain_id: int)-> None:
        """任务第一次开始激活上行链路"""
        if f"{chain_id}" not in self.uplink_task:
            return
        # 当最小堆中不存在时入堆
        if chain_id not in self.uplink_min_heap:
            heapq.heappush(self.uplink_min_heap, chain_id)

    def downlink_start(self, chain_id: int)-> None:
        """任务第一次开始激活下行链路"""
        if f"{chain_id}" not in self.downlink_task:
            return
        # 当最小堆中不存在时入堆
        if chain_id not in self.downlink_min_heap:
            heapq.heappush(self.downlink_min_heap, chain_id)

    def update(self, slot_t: int)-> None:
        # 如果已经更新过则退出
        if self.update_slot != slot_t:
            self.update_uplink = False
            self.update_downlink = False
        # 查看是否有uplink更新
        if self.uplink_min_heap and not self.update_uplink:
            uplink_chain_id: int = self.uplink_min_heap[0]
            uplink_task_size: float = self.uplink_task[f"{uplink_chain_id}"]["task_size"]
            uplink_task_deadline: int = self.uplink_task[f"{uplink_chain_id}"]["task_deadline"]
            # 发布数据更新事件
            self.band_event_bus.publish(
                event=LinkTransEvent(
                    type=BdEventType.UPLINK_DATA_UPDATE,
                    slot = slot_t,
                    data={"user_id":self.user_id, "task_size":uplink_task_size, "task_deadline":uplink_task_deadline}
                )
            )
            # 上行链路已更新
            self.update_uplink = True
        # 查看是否有downlink更新
        if self.downlink_min_heap and not self.update_downlink:
            downlink_chain_id: int = self.downlink_min_heap[0]
            downlink_task_size: float = self.downlink_task[f"{downlink_chain_id}"]["task_size"]
            downlink_task_deadline: int = self.downlink_task[f"{downlink_chain_id}"]["task_deadline"]
            # 发布数据更新事件
            self.band_event_bus.publish(
                event=LinkTransEvent(
                    type=BdEventType.DOWNLINK_DATA_UPDATE,
                    slot = slot_t,
                    data={"user_id":self.user_id, "task_size":downlink_task_size, "task_deadline":downlink_task_deadline}
                )
            )
            # 下行链路已更新
            self.update_downlink = True
        # 记录已经更新过的时隙
        self.update_slot = slot_t



    def _get_bandwidth_allocation_and_transmit(self, event: BdEvent)-> None:
        """带宽调度器发布BANDWIDTH_ALLOCATION后，处理带宽分配"""
        if not isinstance(event, BandAllocEvent):
            return
        if event.type != BdEventType.BANDWIDTH_ALLOCATE:
            return
        if self.user_id not in event.data["user_list"]:
            return
        user_idx: int = event.data["user_list"].index(self.user_id)
        # 记录分配带宽
        if event.data["uplink_alloc_list"][user_idx] > 0 and self.uplink_min_heap:
            uplink_chain_id: int = self.uplink_min_heap[0]
            self.uplink_task[f"{uplink_chain_id}"]["band_allocate"] = event.data["uplink_alloc_list"][user_idx]
            self._transmit_uplink(slot_t=event.slot)
        if event.data["downlink_alloc_list"][user_idx] > 0 and self.downlink_min_heap:
            downlink_chain_id: int = self.downlink_min_heap[0]
            self.downlink_task[f"{downlink_chain_id}"]["band_allocate"] = event.data["downlink_alloc_list"][user_idx]
            self._transmit_downlink(slot_t=event.slot)



    def _transmit_uplink(
        self,
        slot_t: int,
        delta: int = para.simu_config["time_slot"],
        s_to_ms: float = 10 ** 3    # 计算link_rate / s_to_ms换算为bit/ms
    )-> None:
        """根据分配的带宽传输数据"""
        if self.uplink_min_heap:
            uplink_chain_id: int = self.uplink_min_heap[0]
            uplink_band_alloc: float = self.uplink_task[f"{uplink_chain_id}"]["band_allocate"]
            # 计算可以传输的数据量
            uplink_rate: float = self.band.uplink_rate(alpha=uplink_band_alloc, t=slot_t)
            self.uplink_task[f"{uplink_chain_id}"]["task_size"] -= uplink_rate / s_to_ms * delta
            # 如果当前数据包传输完成
            if self.uplink_task[f"{uplink_chain_id}"]["task_size"] <= 0:
                # 删除任务列表
                _ = self.uplink_task.pop(f"{uplink_chain_id}")
                _ = heapq.heappop(self.uplink_min_heap)
                # 添加任务完成列表
                self.complete_uplink_task.append(uplink_chain_id)

    def _transmit_downlink(
        self,
        slot_t: int,
        delta: int = para.simu_config["time_slot"],
        s_to_ms: float = 10 ** 3    # 计算link_rate / s_to_ms换算为bit/ms
    )-> None:
        """根据分配的带宽传输数据"""
        if self.downlink_min_heap:
            downlink_chain_id: int = self.downlink_min_heap[0]
            downlink_band_alloc: float = self.downlink_task[f"{downlink_chain_id}"]["band_allocate"]
            # 计算可以传输的数据量
            downlink_rate: float = self.band.downlink_rate(alpha=downlink_band_alloc, t=slot_t)
            self.downlink_task[f"{downlink_chain_id}"]["task_size"] -= downlink_rate / s_to_ms * delta
            # 如果当前数据包传输完成
            if self.downlink_task[f"{downlink_chain_id}"]["task_size"] <= 0:
                # 删除任务列表
                _ = self.downlink_task.pop(f"{downlink_chain_id}")
                _ = heapq.heappop(self.downlink_min_heap)
                # 添加任务完成列表
                self.complete_downlink_task.append(downlink_chain_id)



    # def connect_esn_id(self, slot: int)-> int:
    #     return self.connect_esn_id_slist[slot]

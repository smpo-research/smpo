# smpo/base/task_transmit.py
from .user_link import UserLink
from .user_mobility import UserMobility


class MsUplinkTask:
    def __init__(
        self,
        user_id: int,
        chain_id: int,
        task_size: float,
        task_deadline: int
    )-> None:
        # 任务属性
        self.user_id: int = user_id
        self.chain_id: int = chain_id
        self.task_size: float = task_size
        self.task_deadline: int = task_deadline
        self.user_link: UserLink = UserLink(user_id)
        # 初始化user_link任务
        self.user_link.add_uplink_task(chain_id=chain_id,task_size=task_size,task_deadline=task_deadline)
        # 任务执行时间管理
        self.start_slot: int = -1
        self.complete_slot: int = -1
        self.update_slot: int = -1
        self.complete_on_time: bool|None = None

    def start(
        self,
        start_slot: int
    )-> None:
        # 利用user_link传输任务
        self.user_link.uplink_start(chain_id=self.chain_id)
        self.start_slot = start_slot
        self.update_slot = start_slot

    def update(self)-> None:
        if self.is_completed:
            return
        self.user_link.update(slot_t=self.update_slot)
        if self.chain_id in self.user_link.complete_uplink_task:
            if self.complete_slot == -1:
                self.complete_slot = self.update_slot
                self.complete_on_time = self.task_deadline >= (self.complete_slot - self.start_slot)
            return
        self.update_slot += 1

    @property
    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.complete_slot != -1

    @property
    def nearest_esn_id(self)-> int:
        return UserMobility(user_id=self.user_id).connected_bs_idx[self.update_slot]


class MsDownlinkTask:
    def __init__(
        self,
        user_id: int,
        chain_id: int,
        task_size: float,
        task_deadline: int
    )-> None:
        # 任务属性
        self.user_id: int = user_id
        self.chain_id: int = chain_id
        self.task_size: float = task_size
        self.task_deadline: int = task_deadline
        self.user_link: UserLink = UserLink(user_id)
        # 初始化user_link任务
        self.user_link.add_downlink_task(chain_id=chain_id,task_size=task_size,task_deadline=task_deadline)
        # 任务执行时间管理
        self.start_slot: int = -1
        self.complete_slot: int = -1
        self.update_slot: int = -1
        self.complete_on_time: bool|None = None

    def start(
        self,
        start_slot: int
    )-> None:
        # 利用user_link传输任务
        self.user_link.downlink_start(chain_id=self.chain_id)
        self.start_slot = start_slot
        self.update_slot = start_slot

    def update(self)-> None:
        if self.is_completed:
            return
        self.user_link.update(slot_t=self.update_slot)
        if self.chain_id in self.user_link.complete_downlink_task:
            if self.complete_slot == -1:
                self.complete_slot = self.update_slot
                self.complete_on_time = self.task_deadline >= (self.complete_slot - self.start_slot)
            return
        self.update_slot += 1

    @property
    def is_completed(self) -> bool:
        """检查任务是否完成"""
        return self.complete_slot != -1

    @property
    def nearest_esn_id(self)-> int:
        return UserMobility(user_id=self.user_id).connected_bs_idx[self.update_slot]

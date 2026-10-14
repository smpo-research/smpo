# smpo/base/esns.py
from __future__ import annotations

from typing import Any

from .config_para import Parameter as para
from .esnb import BdScheduler
from .user_link import UserLink


class MmecBdScheduler:
    _instance: MmecBdScheduler|None = None

    def __new__(cls, *args: Any, **kwargs: Any)-> MmecBdScheduler:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        node_num:int = para.esnc_config["esn_num"]
    )-> None:
        if hasattr(self, "_initialized"):
            return
        self.band_scheduler_list: list[BdScheduler] = [BdScheduler(node_id=i) for i in range(node_num)]
        self._initialized = True

    def alloc_band_pslot(self)-> None:
        '''负责在每个slot最后更新带宽分配'''
        for band_scheduler in self.band_scheduler_list:
            band_scheduler.band_allocate()

    def reset(self, user_num: int = para.user_config["user_num"])->None:
        """重置所有BdScheduler和所有UserLink"""
        for band_scheduler in self.band_scheduler_list:
            band_scheduler.reset()
        for user_id in range(user_num):
            UserLink(user_id=user_id).reset()

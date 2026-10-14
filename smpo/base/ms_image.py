#smpo/base/ms_image.py
from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from .config_para import Parameter as para
from .esn import EdgeServerNode
from .ms_type import MsType

# 全局变量
MAX_WAIT_SLOT: int = para.simu_config["total_time_slots"]


class MsImage:
    """微服务image定义"""
    _instances: dict[str,MsImage] = {}

    def __new__(cls, esn_id: int, image_id: int, *args: Any, **kwargs: Any)-> MsImage:
        key: str = f"{esn_id}_{image_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        esn_id: int,
        image_id: int
    ) -> None:
        if hasattr(self, "image_id"):
            return
        self.esn_id: int = esn_id
        self.image_id: int = image_id
        self.ms_type: MsType = MsType(ms_id=image_id)
        self.esn: EdgeServerNode = EdgeServerNode(node_id=esn_id)

        # 部署决策相关, 全部是slot
        self.current_decision: bool = False
        self.wait_slot: int = MAX_WAIT_SLOT # 等待的slot, 而不是时长
        self.update_slot: int = 0

    def _sync_to_control(self, field: str,  value: bool) -> None:
        from .ms_control import ImageControl
        if field == 'deployed':
            ImageControl().state_deployed[self.esn_id, self.image_id] = value
        if field == 'decided':
            ImageControl().state_decided[self.esn_id, self.image_id] = value

    def update(
        self,
        decision: bool,
        ec_rate: float =para.esng_config["eclink_rate"],
        ec_propagation: float =para.esng_config["eclink_propagation_delays"],
        delta: int = para.simu_config["time_slot"]
    )->None:
        """
        按照system_time更新决策与部署过程
        """
        if decision:    # 决策为True
            if not self.current_decision: # 第一次决策部署
                if self._check_esn_storage():   # 存储检查
                    # 分配存储
                    self.esn.allocate_storage(storage=self.ms_type.storage_require)
                    # 记录决策信息
                    self.wait_slot=int(np.ceil((self.ms_type.storage_require/ec_rate + ec_propagation)/delta))
                    self.current_decision = True
                else: # 部署image超出存储空间，部署失败
                    self.wait_slot = MAX_WAIT_SLOT
                    self.current_decision = False
            else:  # 已在部署中，倒计时
                 self.wait_slot = max(0, self.wait_slot - 1)
        elif self.current_decision:   # 之前决策为True，决策更改释放存储
            # 释放存储
            self.esn.release_storage(storage=self.ms_type.storage_require)
            # 记录决策信息
            self.wait_slot = MAX_WAIT_SLOT
            self.current_decision = False
        # 更新时隙
        self.update_slot += 1

        # [Sync] 同步状态
        is_decided: bool = self.current_decision
        is_deployed: bool = self.current_decision and (self.wait_slot <= 0)

        self._sync_to_control(field="deployed", value=is_deployed)
        self._sync_to_control(field="decided",value=is_decided)


    def reset(self)-> None:
        self.update(decision=False)
        self.wait_slot = MAX_WAIT_SLOT # 等待的slot, 而不是时长
        self.update_slot = 0



    def _check_esn_storage(self) -> bool:
        """检查ESN是否具备足够的存储空间以满足当前ms_type的需求。"""
        return self.esn.available_storage >= self.ms_type.storage_require

    @property
    def is_deployed(self) -> bool:
        """判断当前镜像是否已成功部署并生效"""
        return self.current_decision and self.wait_slot <= 0

    @override
    def __repr__(self) -> str:
        return (f"MsImage(ms_id={self.image_id})")

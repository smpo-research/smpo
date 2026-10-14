#smpo/base/ms_type.py
from __future__ import annotations

from typing import Any

from typing_extensions import override

from .config_para import MsConfig
from .config_para import Parameter as para
from .config_random import SharedRandomGenerator


class MsType:
    """微服务类型定义"""
    _instances: dict[str, MsType] = {}

    def __new__(cls, ms_id: int, *args: Any, **kwargs: Any)-> MsType:
        key: str = f"{ms_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
            self,
            ms_id: int,
            config: MsConfig = para.ms_config
    )->None:
        # 如果没有相关属性，则初始化
        if hasattr(self, "ms_id"):
            return
        # 解析字典
        switch_io_ratio: bool = config["switch_io_ratio"]
        switch_storage_require: bool = config["switch_storage_require"]
        io_ratio_base: float = config["io_ratio_base"]
        io_ratio_fluc: float = config["io_ratio_fluctuate"]
        compute_intensity: float = config["computational_intensity"]
        storage_require_base: float = config["storage_require_base"]
        storage_require_fluc: float = config["storage_require_fluctuate"]
        # 设置随机
        random_generator: SharedRandomGenerator = SharedRandomGenerator(seed=config["random_seed"])
        # ms_type属性
        self.ms_id: int = ms_id
        self.io_ratio: float = io_ratio_base + switch_io_ratio * random_generator.uniform(low=-1, high=1) * io_ratio_fluc
        self.compute_intensity: float = compute_intensity
        self.storage_require: float = storage_require_base + switch_storage_require * random_generator.randint(low=-1, high=1) * storage_require_fluc

    @override
    def __repr__(self) -> str:
        return (
            f"MsType(id={self.ms_id}, \
            io_ratio={self.io_ratio}, \
            compute_intensity={self.compute_intensity},\
            storage_require={self.storage_require})\n"
        )

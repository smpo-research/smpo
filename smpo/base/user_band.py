# SMPO/base/user_band.py
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .user_mobility import UserMobility


class SmallScaleFading:
    """小尺度衰落"""
    _instances: dict[str, SmallScaleFading] = {}

    def __new__(cls, user_id: int, *args: Any, **kwargs: Any)-> SmallScaleFading:
        key: str = f"{user_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, user_id: int)-> None:
        if hasattr(self, "_user_id"):
            return
        self.user_id: int = user_id  # 存储用户ID
        self.slot_list: list[float] = self.__class__._generate_fading_list()

    @staticmethod
    def _generate_fading_list(
        h_square_mean: float = para.band_config["h_square_mean"],
        total_time_slots: int = para.simu_config["total_time_slots"],
        random_seed: int = para.band_config["random_seed"]
        )-> list[float]:
        """生成小尺度衰落值的时间列表"""
        random_generator: SharedRandomGenerator = SharedRandomGenerator(seed=random_seed)
        small_sacll_fading: list[float] = random_generator.exponential(scale=h_square_mean, size=total_time_slots).tolist()
        return small_sacll_fading


class LargeScaleFading:
    """大尺度衰落"""
    _instances: dict[str, LargeScaleFading] = {}

    def __new__(cls, user_id: int, *args: Any, **kwargs: Any)-> LargeScaleFading:
        key: str = f"{user_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, user_id: int)-> None:
        if hasattr(self, '_user_id'):
            return
        self.user_id: int = user_id
        self.slot_list: list[float] = self.__class__._generate_fading_list(user_id)

    @staticmethod
    def _generate_fading_list(
        user_id: int,
        g0: float = para.band_config["g0"],
        d0: float = para.band_config["d0"],
        zeta: float = para.band_config["zeta"]
        ) -> list[float]:
        """计算大尺度衰落"""
        connected_bs_distance: NDArray[np.float64] =  np.array(UserMobility(user_id).connected_bs_distance)
        large_scale_fading: NDArray[np.float64] = g0 * ((d0 / connected_bs_distance) ** zeta)
        return large_scale_fading.tolist()




class SignalNoiseRation:
    _instances: dict[str, SignalNoiseRation] = {}

    def __new__(cls, user_id: int, *args: Any, **kwargs: Any)-> SignalNoiseRation:
        key: str = f"{user_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, user_id: int)-> None:
        if not hasattr(self, "_user_id"):
            self.user_id: int = user_id
            self.uplink_slist: list[float] = self.__class__._generate_uplink_snr(user_id)
            self.downlink_slist: list[float] = self.__class__._generate_downlink_snr(user_id)

    @staticmethod
    def _generate_uplink_snr(
        user_id: int,
        p_ul: float = para.band_config["uplink_power"],
        w_ul: float = para.band_config["uplink_bandwidth"],
        n0: float= para.band_config["N0"],
        small_scale_fading: list[float]|None = None,
        large_scale_fading: list[float]|None = None
    )->list[float]:
        if not small_scale_fading:
            small_scale_fading = SmallScaleFading(user_id).slot_list
        if not large_scale_fading:
            large_scale_fading = LargeScaleFading(user_id).slot_list
        small: NDArray[np.float64] = np.array(small_scale_fading)
        large: NDArray[np.float64] = np.array(large_scale_fading)
        uplink_snr: NDArray[np.float64] = p_ul * small * large / (n0 * w_ul)
        return uplink_snr.tolist()

    @staticmethod
    def _generate_downlink_snr(
        user_id: int,
        p_dl: float = para.band_config["downlink_power"],
        w_dl: float = para.band_config["downlink_bandwidth"],
        n0: float = para.band_config["N0"],
        small_scale_fading: list[float]|None = None,
        large_scale_fading: list[float]|None = None
    )->list[float]:
        if not small_scale_fading:
            small_scale_fading = SmallScaleFading(user_id).slot_list
        if not large_scale_fading:
            large_scale_fading = LargeScaleFading(user_id).slot_list
        small: NDArray[np.float64] = np.array(small_scale_fading)
        large: NDArray[np.float64] = np.array(large_scale_fading)
        downlink_snr: NDArray[np.float64] = p_dl * small * large / (n0 * w_dl)
        return downlink_snr.tolist()





class UserBand:
    _instances: dict[str, UserBand] = {}

    def __new__(cls, user_id: int, *args: Any, **kwargs: Any)-> UserBand:
        key: str = f"{user_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        user_id: int,
        w_ul: float = para.band_config["uplink_bandwidth"],
        w_dl: float = para.band_config["downlink_bandwidth"]
    )-> None:
        if not hasattr(self, "_user_id"):
            self.user_id: int = user_id
            self.uplink_w: float = w_ul
            self.downlink_w: float = w_dl
            self.uplink_snr_slist: list[float] = SignalNoiseRation(user_id).uplink_slist
            self.downlink_snr_slist: list[float] = SignalNoiseRation(user_id).downlink_slist

    def uplink_rate(
        self,
        alpha: float|None,
        t: int
    )-> float:
        """计算上行速率
        Args:
            alpha (float): The alpha value.
            t (int): The time slot.
        Returns:
            float: The uplink rate.
        """
        if not alpha:
            return 0.0
        return alpha * self.uplink_w * np.log2(1 + self.uplink_snr_slist[t] / alpha)

    def downlink_rate(
        self,
        alpha: float|None,
        t: int
    )-> float:
        """计算下行速率
        Args:
            alpha (float): The alpha value.
            t (int): The time slot.
        Returns:
            float: The downlink rate.
        """
        if not alpha:
            return 0.0
        return alpha * self.downlink_w * np.log2(1 + self.downlink_snr_slist[t] / alpha)

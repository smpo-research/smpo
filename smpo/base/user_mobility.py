# SMPO/base/user_mobility.py
from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np
from numpy.typing import NDArray

from .config_para import Parameter as para
from .config_para import UserConfig
from .config_random import SharedRandomGenerator
from .esnc import EsnCoordinates


class ConnectBSResult(NamedTuple):
    """
    一次计算后返回的所有结果，避免重复计算。
    支持解包为 (indices, distances) 以兼容旧接口。
    """
    indices: list[int] # 连接基站索引
    distances: list[float] # 连接基站的距离



class UserMobility:
    _instances: dict[str, UserMobility] = {}  # 存储单例实例的字典

    def __new__(cls, user_id: int, *args: Any, **kwargs: Any)-> UserMobility:
        key: str = f"{user_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, user_id: int) -> None:
        """初始化用户移动模型 (每个user_id仅执行一次)"""
        # 防止重复初始化
        if hasattr(self, 'user_id'):
            return
        self.user_id: int = user_id
        self.path: list[tuple[float, float]] = self.__class__.generate_path()
        connect_result: ConnectBSResult = self.__class__.get_nearest_bs(self.path)
        self.connected_bs_idx: list[int] = connect_result.indices
        self.connected_bs_distance: list[float] = connect_result.distances


    @staticmethod
    def generate_path(
        map_width: int = para.esnc_config["area_scope_x"],
        map_height: int =para.esnc_config["area_scope_y"],
        user_config: UserConfig = para.user_config,
        steps: int = para.simu_config["total_time_slots"],
        random_seed: int = para.user_config["random_seed"],
        delta: int = para.simu_config["time_slot"]
    ) -> list[tuple[float, float]]:
        # 解析字典
        speed_base: float=user_config["user_speed_base"]
        speed_fluc: float=user_config["user_speed_fluctuate"]
        switch_speed: bool = user_config["switch_speed"]
        # 设置随机
        random_generator: SharedRandomGenerator=SharedRandomGenerator(seed=random_seed)

        # === 1. 预生成所有随机量（避免循环内随机数开销）===
        # 停留掩码 (5%概率停留)
        stay_mask: NDArray[np.bool] = random_generator.random(steps) < 0.05
        # 角度变化 (90%概率±15°, 10%概率±45°)
        small_turn: NDArray[np.float64] = random_generator.uniform(-np.pi/12, np.pi/12, steps)
        large_turn: NDArray[np.float64] = random_generator.uniform(-np.pi/3, np.pi/3, steps)
        angle_change: NDArray[np.float64] = np.where(random_generator.random(steps) < 0.9, small_turn, large_turn)
        angle_disturbance: NDArray[np.float64] = random_generator.uniform(-np.pi/4, np.pi/4, steps)
        # 速度 (基础速度+波动, 5%概率小步移动)
        speed: NDArray[np.float64] = speed_base + switch_speed * random_generator.uniform(-speed_fluc, speed_fluc, steps)
        speed *= np.where(random_generator.random(steps) < 0.05, 0.3, 1.0)
        # === 2. 初始化状态 ===
        path_x: NDArray[np.float64] = np.empty(steps)
        path_y: NDArray[np.float64] = np.empty(steps)
        # 初始方向
        direction_tmp: float = random_generator.uniform(0, 2 * np.pi)
        # 随机初始位置
        x_tmp: float = random_generator.uniform(0, map_width)
        y_tmp: float = random_generator.uniform(0, map_height)
        # === 3. 向量化路径生成（核心逻辑修正点）===
        for i in range(steps):
            # 5%概率完全停留（不改变方向）
            if stay_mask[i]:
                path_x[i] = x_tmp
                path_y[i] = y_tmp
                continue
            # 方向更新 - 严格遵守原始逻辑
            target_direction: float = (direction_tmp + angle_change[i]) % (2 * np.pi)
            direction_tmp = 0.8 * direction_tmp + 0.2 * target_direction
            # 计算位移
            dx: float = speed[i] * delta * np.cos(direction_tmp)
            dy: float = speed[i] * delta * np.sin(direction_tmp)
            # 计算新位置
            new_x: float = x_tmp + dx
            new_y: float = y_tmp + dy
            # 边界反射
            if new_x < 0 or new_x > map_width:
                direction_tmp = np.pi - direction_tmp + angle_disturbance[i] # 用于后续步骤
                new_x = 0 if new_x < 0 else map_width
            # Y边界反射 (修改方向影响后续步骤)
            if new_y < 0 or new_y > map_height:
                direction_tmp = -direction_tmp + angle_disturbance[i] # 用于后续步骤
                new_y = 0 if new_y < 0 else map_height
            # 更新当前状态
            x_tmp, y_tmp = new_x, new_y
            path_x[i], path_y[i] = new_x, new_y
        # === 4. 格式化为输出要求 ===
        return list(zip(path_x.astype(float), path_y.astype(float)))

    @staticmethod
    def get_nearest_bs(
        path: list[tuple[float, float]],
        bs_coordinates: Optional[list[tuple[float,float]]] = None
        ) -> ConnectBSResult:
        """优化后的最近基站查找函数（速度提升3-10倍）"""
        node_coordinates: list[tuple[float,float]] = bs_coordinates if bs_coordinates else EsnCoordinates().node_coordinates
        path_array: NDArray[np.float32] = np.asarray(path, dtype=np.float32)  # 使用float32节省内存和计算
        bs_array: NDArray[np.float32] = np.asarray(node_coordinates, dtype=np.float32)
        # 计算各点模长平方
        path_sq: NDArray[np.float32] = np.sum(path_array**2, axis=1)  # 长度N
        bs_sq: NDArray[np.float32] = np.sum(bs_array**2, axis=1)      # 长度M
        # 计算交叉项（向量化点积）
        cross_term: NDArray[np.float32] = np.dot(path_array, bs_array.T)  # N×M矩阵
        # 组合得到平方距离矩阵
        sq_distances: NDArray[np.float32] = path_sq[:, np.newaxis] + bs_sq - 2 * cross_term
        # 批量获取最近基站
        min_indices: NDArray[np.int64] = np.argmin(sq_distances, axis=1)
        # 仅计算需要的N个真实距离（而非N*M个）
        min_distances: NDArray[np.float32] = np.sqrt(sq_distances[np.arange(len(path)), min_indices])
        # 防止除零错误（保持原始逻辑）
        min_distances = np.where(min_distances < 1e-5, 1e-5, min_distances)
        return ConnectBSResult(indices=min_indices.tolist(), distances=min_distances.tolist())

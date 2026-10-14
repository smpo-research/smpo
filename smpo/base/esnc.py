# smpo/base/esnc.py
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config_para import Parameter as para
from .config_random import SharedRandomGenerator


class EsnCoordinates:
    _instance: EsnCoordinates|None = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs:Any)-> EsnCoordinates:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        node_num: int = para.esnc_config["esn_num"],
        width: int = para.esnc_config["area_scope_x"],
        height: int = para.esnc_config["area_scope_y"],
    )-> None:
        if self.__class__._initialized:
            return
        self.node_num: int = node_num
        self.scope:tuple[int,int] = (width, height)
        self.node_coordinates: list[tuple[float,float]] = self.__class__.node_uniform_coordinates(scope=self.scope, node_num=self.node_num)

    @staticmethod
    def node_uniform_coordinates(
        scope: tuple[int, int],
        node_num: int,
        min_radius: int = 300,
        candidate_point_num: int = 30 # 这个参数其实没用了，但保留接口兼容
    ) -> list[tuple[float, float]]:
        """
        [修复版] 鲁棒的均匀分布生成算法 (带超时降级)
        防止因空间不足导致的无限循环。
        """
        random_generator = SharedRandomGenerator(seed=para.esng_config["random_seed"])

        points: list[NDArray[np.float64]] = []

        # 动态调整参数：如果尝试失败，逐步减小半径
        current_radius = float(min_radius)
        max_total_attempts = 10000 # 总熔断计数
        total_attempts = 0

        while len(points) < node_num:
            # 单个点的尝试次数限制
            found = False
            for _ in range(100): # 尝试 100 次
                new_point = random_generator.random(size=2) * scope

                # 检查距离
                valid = True
                # 优化：不需要网格搜索，直接算距离（对于几十个点，O(N) 甚至比维护网格更快且不改动类结构）
                for p in points:
                    if np.linalg.norm(new_point - p) < current_radius:
                        valid = False
                        break

                if valid:
                    points.append(new_point)
                    found = True
                    break

            # 如果尝试 100 次都没找到位置
            if not found:
                total_attempts += 100
                # 策略：稍微减小半径，继续尝试
                current_radius *= 0.9
                # print(f"[Warning] ESNC space crowded. Reducing radius to {current_radius:.2f}")

            # 最终熔断：如果实在不行，直接随机硬塞，保证不卡死
            if total_attempts > max_total_attempts:
                print("[Error] ESNC infinite loop detected. Force filling remaining points.")
                while len(points) < node_num:
                    points.append(random_generator.random(size=2) * scope)
                break

        points_array = np.round(np.asarray(points), 2)
        return points_array.tolist()

    def get_coordinate(self, node_id: int)-> tuple[float,float]:
        return self.node_coordinates[node_id]

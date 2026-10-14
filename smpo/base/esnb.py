# smpo/base/scheduler_band.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, TypedDict

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeAlias

from .esn import EdgeServerNode
from .user_band import UserBand


class BdEventType(Enum):
    UPLINK_DATA_UPDATE = auto()
    DOWNLINK_DATA_UPDATE = auto()
    BANDWIDTH_ALLOCATE = auto()

class LinkTransData(TypedDict):
    user_id: int
    task_size: float
    task_deadline: int

class BandAllocData(TypedDict):
    user_list: list[int]
    uplink_alloc_list: list[float]
    downlink_alloc_list: list[float]

@dataclass
class LinkTransEvent:
    type: BdEventType
    slot: int
    data: LinkTransData

@dataclass
class BandAllocEvent:
    type: BdEventType
    slot: int
    data: BandAllocData

BdEvent: TypeAlias = LinkTransEvent|BandAllocEvent



class BdEventBus:
    """事件总线，负责事件的发布和订阅"""
    _instance: BdEventBus|None = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any)-> BdEventBus:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self)-> None:
        if self.__class__._initialized:
            return
        self.subscribers: dict[BdEventType, list[Callable[[BdEvent],None]]] = {}
        self.__class__._initialized = True

    def subscribe(self, event_type: BdEventType, callback: Callable[[BdEvent],None])-> None:
        """订阅事件"""
        self.subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event: BdEvent)-> None:
        """发布事件"""
        callbacks: list[Callable[[BdEvent],None]]|None = self.subscribers.get(event.type)
        if callbacks is not None:
            for callback in callbacks:
                callback(event)


class BdScheduler:
    _instances: dict[str, BdScheduler] = {}

    def __new__(cls, node_id: int, *args: Any, **kwargs: Any)-> BdScheduler:
        key: str = f"{node_id}"
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, node_id: int)-> None:
        if hasattr(self, "esn_id"):
            return
        self.esn_id: int = node_id
        self.event_bus: BdEventBus = BdEventBus()  # 使用Band事件总线
        self.cover_user_slist: list[list[int]] = EdgeServerNode(node_id).cover_user_slist
        self.uplink_task: list[LinkTransData] = []
        self.downlink_task: list[LinkTransData] = []
        # 订阅相关事件
        self.event_bus.subscribe(event_type=BdEventType.UPLINK_DATA_UPDATE, callback=self._handle_uplink_update)
        self.event_bus.subscribe(event_type=BdEventType.DOWNLINK_DATA_UPDATE, callback=self._handle_downlink_update)
        # 调度的时刻
        self.update_slot: int = -1

    def reset(self)-> None:
        self.uplink_task = []
        self.downlink_task = []
        self.update_slot = -1

    def _handle_uplink_update(self, event: BdEvent)-> None:
        """处理用户数据更新"""
        if not isinstance(event, LinkTransEvent):
            return
        if event.type != BdEventType.UPLINK_DATA_UPDATE:
            return
        # 如果更新的时序与当前时序不同，则清空任务列表，收集当前时间时序任务
        if event.slot != self.update_slot:
            self.uplink_task.clear()
            self.downlink_task.clear()
            self.update_slot = event.slot
        # 判断是否和该event相关
        covered_users: list[int] = self.get_cover_users(slot_t=event.slot)
        if event.data["user_id"] not in covered_users:
            return
        # 收集更新调度器数据
        self.uplink_task.append(
            {
                "user_id": event.data["user_id"],
                "task_size": event.data["task_size"],
                "task_deadline": event.data["task_deadline"]
            }
        )

    def _handle_downlink_update(self, event: BdEvent)-> None:
        """处理用户数据更新"""
        if not isinstance(event, LinkTransEvent):
            return
        if event.type != BdEventType.DOWNLINK_DATA_UPDATE:
            return
        # 如果更新的时序与当前时序不同，则清空任务列表，收集当前时间时序任务
        if event.slot != self.update_slot:
            self.uplink_task.clear()
            self.downlink_task.clear()
            self.update_slot = event.slot
        # 判断esn是否和该event相关
        covered_users: list[int] = self.get_cover_users(slot_t=event.slot)
        if event.data["user_id"] not in covered_users:
            return
        # 收集更新调度器数据
        self.downlink_task.append(
            {
                "user_id": event.data["user_id"],
                "task_size": event.data["task_size"],
                "task_deadline": event.data["task_deadline"]
            }
        )

    def band_allocate(self)-> None:
        """计算带宽分配"""
        # 上行链路带宽分配
        if not self.uplink_task and not self.downlink_task:
            return
        # uplink
        uplink_users: list[int] = [item["user_id"] for item in self.uplink_task]
        uplink_alloc: list[float] = []
        if uplink_users:
            uplink_task_sizes: list[float] = [item["task_size"] for item in self.uplink_task]
            uplink_task_deadlines: list[int] = [item["task_deadline"] for item in self.uplink_task]
            uplink_d: list[float] = [0.0 for _ in range(len(uplink_users))]
            uplink_snr: list[float] = [0.0 for _ in range(len(uplink_users))]
            for index, user_id in enumerate(uplink_users):
                uplink_d[index] = uplink_task_sizes[index]/UserBand(user_id).uplink_w/(1+uplink_task_deadlines[index])
                uplink_snr[index] = UserBand(user_id).uplink_snr_slist[self.update_slot]
            uplink_alloc = BandAllocator(d=uplink_d, snr=uplink_snr).optimize()
        # downlink
        downlink_users: list[int] = [item["user_id"] for item in self.downlink_task]
        downlink_alloc: list[float] = []
        if downlink_users:
            downlink_task_sizes: list[float] = [item["task_size"] for item in self.downlink_task]
            downlink_task_deadlines: list[int] = [item["task_deadline"] for item in self.downlink_task]
            downlink_d: list[float] = [0.0 for _ in range(len(downlink_users))]
            downlink_snr: list[float] = [0.0 for _ in range(len(downlink_users))]
            for index, user_id in enumerate(downlink_users):
                downlink_d[index] = downlink_task_sizes[index]/UserBand(user_id).downlink_w/(1+downlink_task_deadlines[index])
                downlink_snr[index] = UserBand(user_id).downlink_snr_slist[self.update_slot]
            downlink_alloc = BandAllocator(d=downlink_d, snr=downlink_snr).optimize()
        # 整合参数
        user_list: list[int] = sorted(set(uplink_users + downlink_users))
        uplink_alloc_list: list[float] = []
        uplink_map: dict[int, float] = dict(zip(uplink_users, uplink_alloc))
        for u in user_list:
            uplink_alloc_list.append(uplink_map.get(u, 0))
        downlink_alloc_list: list[float] = []
        downlink_map: dict[int, float] = dict(zip(downlink_users, downlink_alloc))
        for u in user_list:
            downlink_alloc_list.append(downlink_map.get(u, 0))
        # 发布带宽分配事件
        self.event_bus.publish(
            event=BandAllocEvent(
                type=BdEventType.BANDWIDTH_ALLOCATE,
                slot = self.update_slot,
                data={
                    "user_list": user_list,
                    "uplink_alloc_list": uplink_alloc_list,
                    "downlink_alloc_list": downlink_alloc_list
                }
            )
        )



    def get_cover_users(self, slot_t: int) -> list[int]:
        if not (0 <= slot_t < len(self.cover_user_slist)):
            return []  # 或抛异常，根据业务需求
        return self.cover_user_slist[slot_t]




class BandAllocator:
    """
    信道资源分配优化器，求解以下优化问题：
        min ∑ [D_i / (a_i * log_2(1 + SNR_i/a_i))]
        subject to ∑ a_i ≤ 1
    参数:
        D (array_like): 权重系数 D_i 的数组
        SNR (array_like): 信噪比 SNR_i 的数组
        tol_binary (float, optional): 二分搜索的容差 (默认 1e-6)
        tol_newton (float, optional): 牛顿法的容差 (默认 1e-8)
        max_iter_binary (int, optional): 二分搜索最大迭代次数 (默认 100)
        max_iter_newton (int, optional): 牛顿法最大迭代次数 (默认 100)
    """

    def __init__(
        self,
        d: list[float],
        snr: list[float],
        tol_binary: float = 1e-6,
        tol_newton: float = 1e-6,
        max_iter_binary: int = 100,
        max_iter_newton: int = 100,
    ) -> None:
        d_arr: NDArray[np.float64] = np.array(d, dtype=np.float64)
        snr_arr: NDArray[np.float64] = np.array(snr, dtype=np.float64)

        if len(d_arr) != len(snr_arr):
            raise ValueError("D and SNR must have the same length")
        if np.any(d_arr < 0) or np.any(snr_arr < 0):
            raise ValueError("D and SNR must be non-negative")

        # 归一化 D，鲁棒化 SNR
        self.para_d: NDArray[np.float64] = d_arr
        self.para_snr: NDArray[np.float64] = np.where(snr_arr < 1e-5, 1e-5, snr_arr)
        self.tol_binary: float = tol_binary
        self.tol_newton: float = tol_newton
        self.max_iter_binary: int = max_iter_binary
        self.max_iter_newton: int = max_iter_newton

        # C_i = log(2) * D_i / SNR_i^2
        self.para_c: NDArray[np.float64] = np.log(2) * self.para_d / (self.para_snr ** 2)

        # 标记 D_i == 0 的索引，直接分配 α_i = 0
        self.zero_mask: NDArray[np.bool] = (self.para_d <= 0)
        self.non_zero_mask: NDArray[np.bool] = ~self.zero_mask

        self.optimal_alphas: list[float] = []



    # @jit(nopython=True)
    @staticmethod
    def _h_x_vec(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """向量化 h(x) 计算"""
        result: NDArray[np.float64] = np.zeros_like(x)
        small_mask: NDArray[np.bool] = x < 1e-2
        large_mask: NDArray[np.bool] = ~small_mask
        # 小 x 泰勒近似
        result[small_mask] = (x[small_mask] ** 2) / 2.0
        # 大 x 精确计算
        x_large: NDArray[np.float64] = x[large_mask]
        ln_term: NDArray[np.float64] = np.log(1 + x_large)
        numerator: NDArray[np.float64] = (x_large ** 2) * (ln_term - x_large / (x_large + 1))
        denominator: NDArray[np.float64] = ln_term ** 2
        result[large_mask] = numerator / denominator
        return result

    # @jit(nopython=True)
    @staticmethod
    def _h_derivative_vec(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """向量化 h'(x) 计算"""
        result: NDArray[np.float64] = np.zeros_like(x)
        small_mask: NDArray[np.bool] = x < 1e-5
        large_mask: NDArray[np.bool] = ~small_mask
        # 小 x 近似
        result[small_mask] = x[small_mask]
        # 大 x 精确计算
        x_large: NDArray[np.float64] = x[large_mask]
        ln_term: NDArray[np.float64] = np.log(1 + x_large)
        u: NDArray[np.float64] = (x_large ** 2) * (ln_term - x_large / (x_large + 1))
        u_prime: NDArray[np.float64] = 2 * x_large * (ln_term - x_large / (x_large + 1)) + (x_large ** 3) / ((x_large + 1) ** 2)
        v: NDArray[np.float64] = ln_term ** 2
        v_prime: NDArray[np.float64] = 2 * ln_term / (x_large + 1)
        numerator: NDArray[np.float64] = u_prime * v - u * v_prime
        denominator: NDArray[np.float64] = v ** 2
        safe_denom: NDArray[np.float64] = np.where(denominator != 0, denominator, 1e-15)
        result[large_mask] = numerator / safe_denom
        return result

    def _solve_all_xi_for_lambda(self, lambda_val: float) -> NDArray[np.float64]:
        """
        向量化牛顿法求解所有 x_i 满足 c_i * h(x_i) = lambda_val
        """
        x: NDArray[np.float64] = np.ones_like(self.para_c)  # 初始猜测
        x[self.zero_mask] = 1.0  # D_i=0 的不影响，随便设
        c_nonzero: NDArray[np.float64] = self.para_c[self.non_zero_mask]

        for _ in range(self.max_iter_newton):
            x_nonzero: NDArray[np.float64] = x[self.non_zero_mask]
            if len(x_nonzero) == 0:
                break
            h_vals: NDArray[np.float64] = self.__class__._h_x_vec(x=x_nonzero)
            h_derivs: NDArray[np.float64] = self.__class__._h_derivative_vec(x=x_nonzero)
            g_vals: NDArray[np.float64] = c_nonzero * h_vals
            g_derivs: NDArray[np.float64] = c_nonzero * h_derivs
            diff: NDArray[np.float64] = np.abs(g_vals - lambda_val)
            converged_nonzero: NDArray[np.bool] = diff < self.tol_newton
            if np.all(converged_nonzero):
                break
            delta: NDArray[np.float64] = (g_vals - lambda_val) / (g_derivs + 1e-15)  # 防除零
            x_new_nonzero: NDArray[np.float64] = x_nonzero - delta
            # 保证正数
            x_new_nonzero = np.maximum(x_new_nonzero, 1e-8)
            x[self.non_zero_mask] = x_new_nonzero
            # 检查变化量收敛
            if np.max(np.abs(x_new_nonzero - x_nonzero)) < self.tol_newton:
                break
        return x

    def _compute_alpha_sum(self, lambda_val: float) -> tuple[np.float64,NDArray[np.float64]]:
        """计算当前 lambda 下的总带宽分配 ∑α_i"""
        x_i: NDArray[np.float64] = self._solve_all_xi_for_lambda(lambda_val)
        alpha_i: NDArray[np.float64] = np.zeros_like(x_i)
        # α_i = SNR_i / x_i
        alpha_i[self.non_zero_mask] = self.para_snr[self.non_zero_mask] / x_i[self.non_zero_mask]
        alpha_i[self.zero_mask] = 0.0
        return np.sum(alpha_i), alpha_i

    def optimize(self) -> list[float]:
        """主优化函数：二分搜索最优 λ"""
        # 特殊情况处理
        if not np.any(self.non_zero_mask):
            self.optimal_alphas = [0.0] * len(self.para_d)
            return self.optimal_alphas
        if np.sum(self.non_zero_mask) == 1:
            alphas: list[float] = [0.0] * len(self.para_d)
            idx: NDArray[np.int64] = np.where(self.non_zero_mask)[0][0]
            alphas[idx] = 1.0
            self.optimal_alphas = alphas
            return alphas
        # 快速估算 lambda_high
        lambda_low: float = 0.0
        lambda_high: float = 1.0
        # 快速放大 lambda_high 直到 sum(α_i) < 1
        while True:
            alpha_sum, _ = self._compute_alpha_sum(lambda_val=lambda_high)
            if alpha_sum < 1.0:
                break
            lambda_high *= 2
            if lambda_high > 1e15:  # 防止无限增长
                lambda_high = 1e15
                break
        # 二分搜索
        for _ in range(self.max_iter_binary):
            if lambda_high - lambda_low < self.tol_binary:
                break
            lambda_mid: float = (lambda_low + lambda_high) / 2.0
            alpha_sum, _ = self._compute_alpha_sum(lambda_val=lambda_mid)

            if alpha_sum > 1.0:
                lambda_low = lambda_mid
            else:
                lambda_high = lambda_mid

        # 使用最终 lambda 计算最优 α_i
        _, alpha_i = self._compute_alpha_sum(lambda_val=lambda_high)
        alpha_i = np.where(alpha_i < 1e-6, 0.0, alpha_i)  # 小值置零

        total: np.float64 = np.sum(alpha_i)
        if total > 0:
            alpha_i /= total  # 归一化确保和为1

        self.optimal_alphas = alpha_i.tolist()
        return self.optimal_alphas



# 示例使用
if __name__ == "__main__":
    # 示例数据
    D = [0, 2.0, 100.0]  # D_i 值
    SNR = [0.5, 1.0, 1.5]  # SNR_i 值
    # 优化
    optimal_alphas = BandAllocator(D, SNR).optimize()
    # 输出结果
    print("Optimal resource allocation:")
    for i, alpha in enumerate(optimal_alphas):
        print(f"  alpha_{i+1} = {alpha:.6f}")
    print(f"  alpha_sum = {sum(optimal_alphas)}")

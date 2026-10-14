#smpo/base/ms_control.py
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .ms_image import MsImage
from .ms_instance import MsInstance


class ImageControl:
    _instance: ImageControl|None = None
    _initialized: bool = False
    def __new__(cls, *args: Any, **kwargs: Any)-> ImageControl:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        esn_num: int = para.esnc_config["esn_num"],
        image_num: int = para.ms_config["microservice_num"]
    )->None:
        """
        管理所有esn平台的image:param all_esn_image: [esn_id][image_id]
        """
        if self.__class__._initialized:
            return
        self.esn_num: int = esn_num
        self.image_num: int = image_num
        self.all_image: list[list[MsImage]] = [[MsImage(esn_id=esn_id,image_id=image_id) for image_id in range(image_num)] for esn_id in range(esn_num)]

        # [Vectorization] 全局状态矩阵 (ESN, Image)
        self.state_decided: NDArray[np.bool_] = np.zeros((esn_num, image_num), dtype=bool)
        self.state_deployed: NDArray[np.bool_] = np.zeros((esn_num, image_num), dtype=bool)

        self.__class__._initialized = True

    def update(
        self,
        decisions:list[list[bool]],
    )->None:
        """更新指定微服务镜像的部署决策"""
        for esn_index, image_decisions in enumerate(decisions):
            for image_index, dec in enumerate(image_decisions):
                self.all_image[esn_index][image_index].update(decision=dec)

    def reset(self)-> None:
        # 把所有决策重置为False
        for esn_id in range(self.esn_num):
            for image_id in range(self.image_num):
                self.all_image[esn_id][image_id].reset()
        # [Vectorization] 重置矩阵
        self.state_decided.fill(False)
        self.state_deployed.fill(False)



class InstanceControl:
    _instance: InstanceControl|None = None
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any)-> InstanceControl:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        esn_num: int = para.esnc_config["esn_num"],
        image_num: int = para.ms_config["microservice_num"],
        instance_num: int = para.ms_config["instance_candidate_num"],
        random_seed = para.ms_config["random_seed"]
    )->None:
        if self.__class__._initialized:
            return
        self.esn_num: int = esn_num
        self.image_num: int = image_num
        self.instance_num: int = instance_num

        # [Vectorization] 全局状态张量 (ESN, Image, Instance)
        # 1. 内存占用 (Float)
        self.state_memory: NDArray[np.float32] = np.zeros((esn_num, image_num, instance_num), dtype=np.float32)

        # 2. 决策状态 (Bool)
        self.state_decided: NDArray[np.bool_] = np.zeros((esn_num, image_num, instance_num), dtype=bool)
        # 3. 是否部署 (Bool)
        self.state_deployed: NDArray[np.bool_] = np.zeros((esn_num, image_num, instance_num), dtype=bool)
        # 4. 是否被占用 (Bool)
        self.state_occupied: NDArray[np.bool_] = np.zeros((esn_num, image_num, instance_num), dtype=bool)
        # 5. 调用次数 (Int)
        self.state_call_count: NDArray[np.int32] = np.zeros((esn_num, image_num, instance_num), dtype=np.int32)
        # 6. 成功调用次数 (Int)
        self.state_success_count: NDArray[np.int32] = np.zeros((esn_num, image_num, instance_num), dtype=np.int32)

        # 初始化对象 (对象内部现在会更新上述数组)
        self.all_instance: list[list[list[MsInstance]]]=[[
            [MsInstance(esn_id=esn_id, image_id=image_id, instance_id=instance_id) for instance_id in range(instance_num)]
            for image_id in range(image_num) ] for esn_id in range(esn_num)
        ]
        self.random_generator: SharedRandomGenerator = SharedRandomGenerator(seed=random_seed)

        self.__class__._initialized = True

    def update(
        self,
        decisions:list[list[int]] # 参数为增加或减少instance个数
        )-> None:
        """更新所有实例的部署决策"""
        for esn_index, instance_decisions in enumerate(decisions):
            for image_index, dec_num in enumerate(instance_decisions):
                # 获取当前该 (esn, image) 下所有实例的部署状态
                current_decisions = self.state_decided[esn_index, image_index]
                dec_zero_indices: np.ndarray = np.where(~current_decisions)[0]
                dec_one_indices: np.ndarray = np.where(current_decisions)[0]
                # 检查父级 Image 是否还存活
                sample_instance: MsInstance = self.all_instance[esn_index][image_index][0]
                is_image_alive: bool = sample_instance.image.current_decision
                # 如果 Image 死了，必须强制清空所有 Instance，无论 dec_num 是多少
                if not is_image_alive and len(dec_one_indices) > 0:
                    dec_num: int = -len(dec_one_indices) # 强制设定为移除所有
                # 记录本轮操作过的实例索引 (避免重复 update)
                processed_indices: set[int] = set()
                if dec_num > 0:
                    num_to_deploy: int = min(dec_num, len(dec_zero_indices))
                    if num_to_deploy > 0:
                        selected = self.random_generator.choice(dec_zero_indices, size=num_to_deploy, replace=False)
                        # 更新对象状态 (对象内部会自动同步数组)
                        for idx in selected:
                            self.all_instance[esn_index][image_index][idx].update(decision=True)
                            processed_indices.add(idx)
                elif dec_num < 0:
                    num_to_remove: int = min(-dec_num, len(dec_one_indices))
                    if num_to_remove > 0:
                        # 获取当前已部署实例的 success_rate (用于移除表现最差的)
                        candidate_objs: list[MsInstance] = [self.all_instance[esn_index][image_index][idx] for idx in dec_one_indices]
                        success_rates: np.ndarray = np.array([inst.success_call_rate for inst in candidate_objs])
                        # 根据 success_call 升序排列
                        sorted_local_indices: np.ndarray = np.argsort(success_rates)
                        # 映射回全局 index
                        selected_global_indices: np.ndarray = dec_one_indices[sorted_local_indices[:num_to_remove]]
                        for idx in selected_global_indices:
                            self.all_instance[esn_index][image_index][idx].update(decision=False)
                            processed_indices.add(idx)
                # 维护现有实例的时钟 (Tick Tock)
                # 刷新视图（因为上面可能修改了状态，但对象内部自动同步了数组，所以这里直接读数组是准的）
                final_decided_view = self.state_decided[esn_index, image_index]
                final_active_indices: np.ndarray = np.where(final_decided_view)[0]
                for idx in final_active_indices:
                    if idx not in processed_indices:
                        # 保持存活，拨动时钟
                        self.all_instance[esn_index][image_index][idx].update(decision=True)

    def reset(self)-> None:
        # 重置所有对象
        for esn_id in range(self.esn_num):
            for image_id in range(self.image_num):
                for instance_id in range(self.instance_num):
                    self.all_instance[esn_id][image_id][instance_id].reset()

        # [Vectorization] 双重保险：强制清零所有矩阵
        self.state_memory.fill(0)
        self.state_decided.fill(False)
        self.state_deployed.fill(False)
        self.state_occupied.fill(False)
        self.state_call_count.fill(0)
        self.state_success_count.fill(0)

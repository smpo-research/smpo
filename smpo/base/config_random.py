# smpo/base/config_random.py
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.random import Generator


class SharedRandomGenerator:
    # 使用字典存储实例
    _instances:dict[str, SharedRandomGenerator] = {}

    def __new__(cls, seed: int|None)-> SharedRandomGenerator:
        # 根据seed的值决定实例键
        if seed is not None:
            instance_key: str = f"seed_{seed}"
        else:
            instance_key = "no_seed_mode"

        if instance_key not in cls._instances:
            cls._instances[instance_key] = super().__new__(cls)
        return cls._instances[instance_key]

    def __init__(self, seed: int|None)-> None:
        # 只初始化一次
        if hasattr(self, '_initialized'):
            return
        self._random_generator: Generator = np.random.default_rng(seed)
        self._initialized: bool = True

    def __getattr__(self, name: str)-> Callable[...,Any]:
        if '_random_generator' in self.__dict__:
            if name == 'randint':
                return getattr(self._random_generator, 'integers')
            return getattr(self._random_generator, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")





# 使用示例
if __name__ == "__main__":
    # use_random_seed=1 的配置
    seed = None

    # use_random_seed=1: 相同seed共享实例
    random_a = SharedRandomGenerator(seed = seed)


    print(f"random_a: {random_a.uniform(0, 1)}")

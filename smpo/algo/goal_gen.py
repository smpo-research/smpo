# smpo/algo/goal_gen.py
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.cluster import SpectralClustering

from base.config_para import Parameter as para

warnings.filterwarnings("ignore", message="Graph is not fully connected")

class GoalGenerator:
    _instance: GoalGenerator|None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> GoalGenerator:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        esn_num: int = para.esnc_config["esn_num"],
        image_num: int = para.ms_config["microservice_num"]
    ) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return
        # 基础配置
        self.esn_num: int = esn_num
        self.image_num: int = image_num
        self.adj_matrix: np.ndarray = np.zeros((image_num, image_num))
        self.node_counts: np.ndarray = np.zeros(image_num)
        self.hub_nodes: set[int] = set()
        self._initialized = True

    def get_goal(self, chain_list: list[list[int]])-> np.ndarray:
        #  AllRequestInfo().all_generate_chain
        self.fit_from_chains(chain_list)
        goal: np.ndarray = self.generate_matrix(hub_ratio=0.2, random_strength=0.1)
        return goal

    def fit_from_chains(self, chains_list: list[list[int]]) -> None:
        """
        解析调用链，构建邻接矩阵和节点计数。
        优化点：减少 Python 循环开销，增加逻辑判断。
        """
        n_total: int = len(chains_list)
        if n_total == 0:
            return

        # 采样逻辑保持不变
        n_sample: int = min(n_total, 1000)
        sampled_indices: np.ndarray = np.random.choice(n_total, size=n_sample, replace=False)

        # 重置矩阵
        self.adj_matrix.fill(0)
        self.node_counts.fill(0)

        for i in sampled_indices:
            chain: list[int] = chains_list[i]
            # 注意：这里隐式过滤掉了 -1, -2 等控制符，这符合逻辑
            valid: list[int] = [n for n in chain if 0 <= n < self.image_num]
            if not valid:
                continue
            # 统计节点频率 (Numpy index add 用于加速，但针对短链 loop 也可以)
            for n in valid:
                self.node_counts[n] += 1
            if len(valid) < 2:
                continue
            # 构建邻接矩阵 (无向图)
            # 优化：对于短链，循环是可以的；如果链很长，可构建 edge list 后一次性 add
            for k in range(len(valid) - 1):
                u, v = valid[k], valid[k+1]
                if u == v:
                    continue
                self.adj_matrix[u, v] += 1
                self.adj_matrix[v, u] += 1 # 对称

    def generate_matrix(self, hub_ratio=0.15, random_strength=0.1) -> np.ndarray:
        """
        生成 Goal 矩阵：Hub 节点共享，Tail 节点聚类分片。
        """
        # 1. 识别 Hub
        # argsort 返回的是从小到大的索引，[::-1] 反转为从大到小
        sorted_idx: np.ndarray = np.argsort(self.node_counts)[::-1]

        # 确保至少有1个 Hub，且不超过 image_num
        num_hubs: int = max(1, int(self.image_num * hub_ratio))
        num_hubs = min(num_hubs, self.image_num)

        self.hub_nodes = set(sorted_idx[:num_hubs])
        # 使用 Numpy 运算获取 tail nodes，比列表推导式快
        all_indices: np.ndarray = np.arange(self.image_num)
        is_tail: np.ndarray = np.isin(all_indices, list(self.hub_nodes), invert=True)
        tail_nodes: np.ndarray = all_indices[is_tail]

        # 初始化 Goal 矩阵 (esn_num, image_num, 1)
        goal: np.ndarray = np.random.uniform(0, 0.05, (self.esn_num, self.image_num, 1))

        # 2. Hub 广播 (Hub 对所有 ESN 可见，值较高)
        if self.hub_nodes:
            hub_list: list[int] = list(self.hub_nodes)
            # 利用 Numpy 广播赋值，避免循环
            # random_values shape: (esn_num, num_hubs)
            rand_vals: np.ndarray = np.random.uniform(1.0 - random_strength, 1.0, size=(self.esn_num, len(hub_list)))
            # 赋值: goal[:, [h1, h2...], 0] = rand_vals
            goal[:, hub_list, 0] = rand_vals

        # 3. Tail 聚类
        num_tails = len(tail_nodes)
        if num_tails > 0:
            # 优化：直接使用 Numpy 高级索引提取子矩阵，代替双重循环 O(1) vs O(N^2)
            # np.ix_ 生成网格索引
            sub_adj: np.ndarray = self.adj_matrix[np.ix_(tail_nodes, tail_nodes)]

            # 数值微扰 (Numerical Perturbation)
            # 即使图不连通，加一个极小的底噪也能让它在数值上变为"弱连通"，防止死锁

            if num_tails >= self.esn_num:
                    # 转换为 float 以便加噪
                sub_adj_stable = sub_adj.astype(np.float64)
                # 加极小底噪
                sub_adj_stable += 1e-6
                # 保持对称性 (虽然加常数不影响对称，但为了严谨)
                sub_adj_stable = (sub_adj_stable + sub_adj_stable.T) / 2
            else:
                sub_adj_stable = sub_adj

            labels: None|np.ndarray = None

            # 策略分支：如果尾部节点太少，或者没有连接，无法进行谱聚类
            if num_tails < self.esn_num or np.sum(sub_adj) == 0:
                # 随机分配或均匀分配
                labels = np.random.randint(0, self.esn_num, size=num_tails)
            else:
                try:
                    # 使用 discretized 策略通常更稳定
                    sc: SpectralClustering = SpectralClustering(
                        n_clusters=self.esn_num,
                        affinity='precomputed',
                        random_state=42,
                        assign_labels='discretize',
                        n_init=10 # 增加尝试次数防止局部最优
                    )
                    labels = sc.fit_predict(sub_adj_stable)
                except Exception:
                    labels = np.random.randint(0, self.esn_num, size=num_tails)
                if labels is None or len(labels) != num_tails:
                    labels = np.random.randint(0, self.esn_num, size=num_tails)

            # 4. 赋值 Tail Goal
            # labels[i] 是第 i 个 tail node 所属的 cluster (对应 esn_id)
            # tail_nodes[i] 是该节点在全局 image_num 中的索引

            # 向量化赋值：
            # 我们需要设置 goal[cluster_id, node_id, 0] = high_value
            # 构建索引数组
            high_vals:np.ndarray = np.random.uniform(1.0 - random_strength, 1.0, size=num_tails)
            goal[labels, tail_nodes, 0] = high_vals
        return goal

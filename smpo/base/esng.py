# SMPO/base/esng.py
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
from typing_extensions import override

from .config_para import EsngConfig
from .config_para import Parameter as para
from .config_random import SharedRandomGenerator
from .esn import EdgeServerNode
from .esnc import EsnCoordinates


# 地图单例类
class EsnGraph:
    _instance: EsnGraph|None = None
    _initialized: bool = False  # 添加初始化标志

    def __new__(cls, *args: Any, **kwargs:Any)-> EsnGraph:
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    def __init__(self)->None:
        if self.__class__._initialized:
            return
        self.node_num: int = EsnCoordinates().node_num
        self.scope:tuple[int,int] = EsnCoordinates().scope
        self.node_coordinates: list[tuple[float,float]] = EsnCoordinates().node_coordinates
        self.node_list: list[EdgeServerNode] = [EdgeServerNode(node_id=i) for i in range(self.node_num)]
        self.graph: nx.Graph[EdgeServerNode] = self.__class__.generate_graph(self.node_list)
        self.__class__._initialized = True


    @staticmethod
    def generate_graph(
        node_list: list[EdgeServerNode],
        config: EsngConfig  = para.esng_config,
    )->nx.Graph[EdgeServerNode]:
        '''
        随机生成ESN的图结构
        '''
        # 字典解析
        switch_wired_rate: bool = config["switch_wired_rate"]
        switch_edge_addition: bool = config["switch_edge_addition"]
        wired_rate_base: float = config["eelink_rate_base"]
        wired_rate_fluc: float = config["eelink_rate_fluctuate"]
        # 设置随机
        random_generator: SharedRandomGenerator = SharedRandomGenerator(seed = config["random_seed"])

        def euclidean_distance(p1:tuple[float, float], p2:tuple[float, float])-> float: # 欧拉距离
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        node_num: int=len(node_list)
        esn_coordinates: list[tuple[float,float]] = [node_list[i].location for i in range(node_num)]
        # nx.Graph初始化
        nx_graph: nx.Graph[EdgeServerNode] = nx.Graph()
        nx_graph.add_nodes_from(nodes_for_adding=node_list)
        # 阶段1: 确保每个节点都连接到最近的邻居
        nearest_edges:set[tuple[EdgeServerNode,EdgeServerNode]] = set()
        for i in range(node_num):
            min_distance: float = np.inf
            nearest_idx: int = -1
            for j in range(node_num):
                if i != j:
                    distance: float = euclidean_distance(p1=esn_coordinates[i], p2=esn_coordinates[j])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_idx = j
            edge: tuple[EdgeServerNode,EdgeServerNode] = (node_list[i],node_list[nearest_idx]) if i < nearest_idx else (node_list[nearest_idx],node_list[i])
            if edge not in nearest_edges:
                nearest_edges.add(edge)
                weight: float = wired_rate_base + random_generator.uniform(-wired_rate_fluc, wired_rate_fluc) * switch_wired_rate
                weight = np.round(weight,2) # 取两位小数
                nx_graph.add_edge(u_of_edge=edge[0], v_of_edge=edge[1], weight=weight)
        # 阶段2: 确保图连通
        while not nx.is_connected(nx_graph):
            components: list[set[EdgeServerNode]] = list(nx.connected_components(nx_graph))
            min_dist: float = np.inf
            best_edge: tuple[EdgeServerNode,EdgeServerNode]|None = None
            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    for node1 in components[i]:
                        for node2 in components[j]:
                            idx1: int = node_list.index(node1)
                            idx2: int = node_list.index(node2)
                            dist: float = euclidean_distance(p1=esn_coordinates[idx1], p2=esn_coordinates[idx2])
                            if dist < min_dist:
                                min_dist = dist
                                best_edge = (node1, node2)
            if best_edge:
                weight = wired_rate_base + random_generator.uniform(-wired_rate_fluc, wired_rate_fluc)*switch_wired_rate
                weight = np.round(weight,2) # 取两位小数
                nx_graph.add_edge(u_of_edge=best_edge[0], v_of_edge=best_edge[1], weight=weight)
        # 阶段3: 基于相对距离的动态概率添加随机边
        if switch_edge_addition:
            for i in range(node_num):
                # 计算节点 i 连接的节点的最小距离
                connected_distances: list[float] = []
                for j in range(node_num):
                    if i != j and nx_graph.has_edge(u=node_list[i], v=node_list[j]):
                        connected_distances.append(euclidean_distance(p1=esn_coordinates[i], p2=esn_coordinates[j]))
                min_connected_dist: float = min(connected_distances)
                threshold: float = min_connected_dist + 250
                for j in range(i+1, node_num):
                    if not nx_graph.has_edge(u=node_list[i], v=node_list[j]):
                        distance = euclidean_distance(p1=esn_coordinates[i], p2=esn_coordinates[j])
                        # 计算参考距离
                        if distance < threshold:
                            reference_dist: float = (distance - min_connected_dist) / (min_connected_dist + 1e-8)
                        else:
                            reference_dist = float('inf')
                        base_prob: float = 1.0
                        # 动态概率计算（指数衰减）
                        probability = base_prob * np.exp(-reference_dist)
                        if random_generator.random() < probability:
                            weight = wired_rate_base*1.5 + random_generator.uniform(-wired_rate_fluc*1.5, wired_rate_fluc*1.5)*switch_wired_rate
                            weight = np.round(weight,2) # 取两位小数
                            nx_graph.add_edge(u_of_edge=node_list[i], v_of_edge=node_list[j], weight=weight)
        return nx_graph

    def min_distance_delay_cost(
        self,
        source_node_idx: int,
        target_node_idx: int
    )-> float:
        """
        计算两个ESN节点之间的最优路由路径, 路径权重为各边R_{n_s,n_d}^{WL}的倒数之和
        return: 单位数据延迟时间，单位ms/bit
        """
        try:
            # 使用dijkstra算法找最短路径
            source_node: EdgeServerNode = self.node_list[source_node_idx]
            target_node: EdgeServerNode = self.node_list[target_node_idx]
            if source_node not in self.graph or target_node not in self.graph:
                raise ValueError("Source or target node not in graph.")
            path:list[EdgeServerNode] = nx.shortest_path( # pyright:ignore[reportUnknownMemberType]
                self.graph,
                source=source_node,
                target=target_node,
                weight=lambda u, v, data: 1 / data['weight'] if data.get('weight', 0) > 0 else np.inf
            )
            # 计算路径总权重
            total_weight: float = sum(1/self.graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
            return total_weight
        except nx.NetworkXNoPath: # 如果没有路径存在
            return np.inf


    def __len__(self)-> int:
        return self.node_num
    def __getitem__(self,index: int)->EdgeServerNode:
            return self.node_list[index]
    @override
    def __repr__(self)-> str:
        return f"EsnGraph({self.graph})"

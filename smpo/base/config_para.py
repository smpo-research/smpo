# SMPO/base/config_para.py
from typing import TypedDict

# 所有数据类型规范


class EsnConfig(TypedDict):
    random_seed: int
    switch_core_num: bool
    switch_compute_capacity: bool
    core_num_base: int
    core_num_fluctuate: int
    compute_capacity_base: float
    compute_capacity_fluctuate: float
    memory_capacity_per_inst: float
    storage_capacity_per_inst: float


class EsncConfig(TypedDict):
    random_seed: int
    esn_num: int
    area_scope_x: int
    area_scope_y: int


class EsngConfig(TypedDict):
    random_seed: int
    switch_wired_rate: bool
    switch_edge_addition: bool
    eelink_rate_base: float  # bit/ms
    eelink_rate_fluctuate: float  # bit/ms
    eclink_rate: float  # bit/ms
    eclink_propagation_delays: float  # ms


class MsConfig(TypedDict):
    random_seed: int
    switch_io_ratio: bool
    switch_storage_require: bool
    io_ratio_base: float
    io_ratio_fluctuate: float
    computational_intensity: float  # cycles/bit
    storage_require_base: float  # bit
    storage_require_fluctuate: float  # bit
    instantiation_delay: float  # ms
    microservice_downtime: float  # ms
    microservice_num: int  # 种类数
    instance_candidate_num: int  # 每种实例数


class ChainConfig(TypedDict):
    random_seed: int
    switch_task_size: bool
    switch_chain_num: bool
    switch_task_deadline: bool
    chain_num_base: int
    chain_num_fluctuate: int
    task_size_base: float  # bit
    task_size_fluctuate: float  # bit
    task_deadline_fluctuate: int  # ms
    max_execute_tolerable_slot: int
    memory_scale: float


class BandConfig(TypedDict):
    random_seed: int
    uplink_bandwidth: float  # Hz
    downlink_bandwidth: float  # Hz
    uplink_power: float  # W
    downlink_power: float  # W
    h_square_mean: float  # small-scale fading coefficient
    g0: float  # 路径损耗系数
    d0: float  # 参考距离 (m)
    zeta: float  # 路径损耗指数
    N0: float  # 噪声功率谱密度 (W/Hz)


class UserConfig(TypedDict):
    random_seed: int
    switch_speed: bool
    user_num: int
    user_speed_base: float  # m/ms
    user_speed_fluctuate: float  # m/ms
    user_mobility: int  # 1 for mobility, 0 for static
    user_mobility_radius: int  # m
    request_rate_base: float
    request_rate_fluctuate: float
    task_deadline_slack: int

class HyperparaConfig(TypedDict):
    obs_drift_threshold: float
    base_supply_ratio: float
    # chain
    migrate_alpha: float
    exec_beta: float
    task_exec_lambda: float
    fair_gama: float
    migrate_penalty_postitive: float
    migrate_penalty_negative: float
    timeout_penalty: float
    # mmec
    performance_alpha: float
    performance_w_acceptance: float
    performance_w_sla: float
    fairness_beta: float
    cost_gamma: float
    penalty_delta: float
    penalty_lazy: float

class AgentConfig(TypedDict):
    buffer_size: int
    batch_size: int
    start_train_steps: int
    model_save_freq: int
    hiro_buffer_freq: int
    hiro_train_freq: int
    log_path: str
    data_path: str
    model_path: str


# ==================== Simulation 配置 ====================
class SimuConfig(TypedDict):
    time_slot: int
    total_time_slots: int
    image_decision_interval: int


# 模块级全局默认设置（在类定义前定义）
GLOBAL_RANDOM_SEED: int|None = -1


class Parameter:
    # ESN
    esn_config: EsnConfig = {
        # 随机设置
        "random_seed": 1,
        "switch_core_num": True,
        "switch_compute_capacity": True,
        # 数据设置
        "core_num_base": 10,  # int
        "core_num_fluctuate": 4,  # "randint"
        "compute_capacity_base": 5 * 10**6,  # cycle/ms
        "compute_capacity_fluctuate": 2 * 10**6,  # cycle/ms  "randint"
        "memory_capacity_per_inst": 2.5 * 8 * 1024**3,  # bit, 1 inst for 1GB
        "storage_capacity_per_inst": 1.2 * 8 * 1024**3,  # bit, 1 inst for 1GB
    }

    # Esn坐标
    esnc_config: EsncConfig = {
        "random_seed": 1,
        "esn_num": 5,  # ESN num
        "area_scope_x": 1000,  # m, boundary 1Km
        "area_scope_y": 1000,  # m, boundary 1Km
    }

    # Graph
    esng_config: EsngConfig = {
        "random_seed": 1,
        "switch_wired_rate": True,
        "switch_edge_addition": True,
        "eelink_rate_base": 5.5 * 10**6,  # bit/ms ,5.5Gbps
        "eelink_rate_fluctuate": 4.5 * 10**6,  # bit/ms ,4.5Gbps "uniform"
        "eclink_rate": 10 * 10**6,  # bit/ms ,10Gbps
        "eclink_propagation_delays": 20,  # ms
    }

    # Microservice
    ms_config: MsConfig = {
        "random_seed": 1,
        "switch_io_ratio": True,
        "switch_storage_require": True,
        "io_ratio_base": 0.8,
        "io_ratio_fluctuate": 0.5,  # "uniform"
        "computational_intensity": 737.5,  # cycles/bit
        "storage_require_base": 1.5 * 8 * 1024**3,  # 1GB，bit
        "storage_require_fluctuate": 0.5 * 8 * 1024**3,  # 1GB
        "instantiation_delay": 200,  # image实例化时延 10～100 单位ms
        "microservice_downtime": 100,  # ms, 微服务停机时间
        "microservice_num": 10,  # 所有微服务的种类数
        "instance_candidate_num": 10,  # 每种微服务的实例数
    }

    # Task
    chain_config: ChainConfig = {
        "random_seed": 1,
        "switch_task_size": True,
        "switch_chain_num": True,
        "switch_task_deadline": True,
        "chain_num_base": 4,  # 任务数
        "chain_num_fluctuate": 3,  # randint
        "task_size_base": 5.5 * 8 * 1024**2,  # bit, 30MB
        "task_size_fluctuate": 4.5 * 8 * 1024**2,  # bit, 20MB
        # "task_deadline_base":5000, # ms
        "task_deadline_fluctuate": 20000,  # ms, 20s
        "max_execute_tolerable_slot": 200,
        "memory_scale": 10.0      # 3MB数据任务进入内容计算要扩大10倍
    }

    band_config: BandConfig = {
        "random_seed": 1,
        "uplink_bandwidth": 10 * 10**6,  # Hz, # 上行带宽，10MHz
        "downlink_bandwidth": 10 * 10**6,  # Hz, # 下行带宽，10MHz
        "uplink_power": 0.5,  # W， # 上行功率，500mW
        "downlink_power": 5,  # W, # 下行功率，1000mW
        "h_square_mean": 1,  # small-scale fading coefficient, 服从exp(1)分布
        "g0": 10**-4,  # 路径损耗系数, 大小为-40dB
        "d0": 1,  # 参考距离 (m)
        "zeta": 4,  # 路径损耗指数
        "N0": 10 ** (-17.4 - 3),  # 噪声功率谱密度 (W/Hz), -174dBm/Hz
    }

    # Users
    user_config: UserConfig = {
        "random_seed": 1,
        "switch_speed": True,
        "user_num": 10,  # 用户数
        "user_speed_base": 0.3 * 10**-3,  # m/ms, 用户速度基准值
        "user_speed_fluctuate": 0.2 * 10**-3,  # m/ms, 用户速度波动范围
        "user_mobility": 1,  # 1 for mobility, 0 for static
        "user_mobility_radius": 100,  # m, 用户移动半径
        "request_rate_base": 0.5*10**-4,  # 请求到达率, 10^-4/ms就是上界
        "request_rate_fluctuate": 0.4*10**-4,  # 请求到达率, 10^-4/ms就是上界
        "task_deadline_slack": 20  # 用来设置用户的可以容忍的deadline
    }

    hyperpara_config: HyperparaConfig = {
        "obs_drift_threshold": 0.3,
        "base_supply_ratio":0.5,
        # task_chain
        "migrate_alpha": 0.4,   # r_migrate
        "exec_beta": 0.6, # r_exec
        "fair_gama": 0.1, # r_fair

        "task_exec_lambda": 1.5,
        "timeout_penalty": 1.2,
        "migrate_penalty_postitive": 0.5,
        "migrate_penalty_negative": 0.8,
        # 权重
        "performance_alpha": 0.4,                         # 性能内部权重：SLA
        "fairness_beta": 0.1,  # 公平性权重
        "cost_gamma": 0.4,                                     # 资源成本权重 (默认值，可根据需要调整)
        "penalty_delta": 0.1,  # 迁移惩罚系数
        "penalty_lazy": 0.1,
        # performance内部权重
        "performance_w_acceptance": 0.5,                              # 性能内部权重：接纳率
        "performance_w_sla": 0.5,

    }

    simu_config: SimuConfig = {
        "time_slot": 1000,  # 毫秒
        "total_time_slots": 10000,  # 总时隙数

        # env
        "image_decision_interval": 10
    }

    agent_config: AgentConfig = {
        "buffer_size": 300 * 2000,   # 20个epi
        "batch_size": 512,
        "start_train_steps": 10000,
        "hiro_train_freq": simu_config["image_decision_interval"],  # 该参数为manager的训练频率
        "hiro_buffer_freq": simu_config["image_decision_interval"], # 该参数为manager的一个step的缓存
        "model_save_freq": simu_config["total_time_slots"] // 50,
        "model_path": "result/models",
        "log_path": "result/logs",
        "data_path": "result/data"

    }

    complete_random_seed: None = None


if __name__ == "__main__":
    print(Parameter.esn_config["random_seed"])

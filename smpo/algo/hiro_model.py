# smpo/algo/hiro_model.py

import torch
import torch.nn as nn
from torch import Tensor


# ==========================================
# 通用初始化函数 (增强版)
# ==========================================
def weights_init(m)-> None:
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # 线性层：正交初始化，增益 sqrt(2)
        # 添加 # type: ignore 忽略类型检查器的误报
        nn.init.orthogonal_(tensor=m.weight, gain=2**0.5)  # type: ignore
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0.0)
    elif classname.find('Conv') != -1:
        nn.init.orthogonal_(tensor=m.weight, gain=2**0.5)  # type: ignore
        if m.bias is not None:
            nn.init.constant_(tensor=m.bias, val=0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(tensor=m.weight, val=1.0)
        nn.init.constant_(tensor=m.bias, val=0.0)


# ==========================================
# 1. Backbone (Transformer)
# ==========================================
class AttentionBackbone(nn.Module):
    def __init__(
        self,
        n_nodes,
        m_tasks,
        input_feature_dim,
        hidden_dim: int = 256
    )-> None:
        super(AttentionBackbone, self).__init__()
        self.n = n_nodes
        self.m = m_tasks
        self.hidden_dim = hidden_dim

        # Positional Embedding
        self.pos_embedding: nn.Parameter = nn.Parameter(data=torch.randn(1, m_tasks, hidden_dim) * 0.01)

        # Input Projection
        self.input_proj: nn.Linear = nn.Linear(in_features=n_nodes * input_feature_dim, out_features=hidden_dim)

        # Transformer Encoder
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True, dim_feedforward=256
        )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- 初始化逻辑 ---
        # 1. 对 Linear 等基础层应用通用初始化
        self.apply(weights_init)

        # 2. [关键] 对 Transformer 部分使用 Xavier 初始化
        # Transformer 的 Attention 权重通常 Xavier 表现更好
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: Tensor)-> Tensor:
        batch_size: int = x.shape[0]
        input_f: int = x.shape[-1]

        x = x.permute(0, 2, 1, 3).reshape(batch_size, self.m, self.n * input_f)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        feature: Tensor = self.transformer(x)
        return feature

# ==========================================
# 2. Manager & Worker Actor
# ==========================================
class BaseActor(nn.Module):
    def __init__(
        self,
        state_shape: tuple[int,int,int],
        goal_shape: tuple[int,int,int],
        action_shape: tuple[int,int,int],
        hidden_dim = 256,
        action_scale: float = 1.0
    )-> None:
        super(BaseActor, self).__init__()
        n_s, m_s, f_dim = state_shape
        n_g, m_g, g_dim = goal_shape
        n_a, m_a, a_dim = action_shape

        assert n_s == n_g == n_a, f"Nodes dim mismatch: {n_s}, {n_g}, {n_a}"
        assert m_s == m_g == m_a, f"Tasks dim mismatch: {m_s}, {m_g}, {m_a}"

        self.n_dim: int = n_s
        self.m_dim: int = m_s
        self.action_last_dim: int = a_dim
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))

        combined_dim: int = f_dim + g_dim


        self.backbone: AttentionBackbone = AttentionBackbone(
            n_nodes=self.n_dim, m_tasks=self.m_dim,
            input_feature_dim=combined_dim, hidden_dim=hidden_dim
        )

        # Output Layer
        self.output_layer: nn.Linear = nn.Linear(in_features=hidden_dim, out_features=self.n_dim * self.action_last_dim)

        # --- 初始化逻辑 ---
        # 1. 基础层初始化 (Backbone 内部已经初始化过，这里主要针对 output_layer)
        self.apply(weights_init)

        # 2. [关键] Actor 输出层使用极小权重 (Gain=0.01)
        # 确保初始动作 Logits 接近 0，经过 Tanh 后动作均值接近 0，
        # 防止初始策略陷入边界（-1 或 1），最大化初始探索。
        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)    # type: ignore
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, state: Tensor, goal: Tensor)-> Tensor:
        batch_size = state.shape[0]
        x: Tensor = torch.cat([state, goal], dim=-1)
        features: Tensor = self.backbone(x)
        logits = self.output_layer(features)
        logits = logits.view(batch_size, self.m_dim, self.n_dim, self.action_last_dim)
        action_logits = logits.permute(0, 2, 1, 3)
        action: Tensor = torch.tanh(input=action_logits)
        action = action * self.action_scale
        return action

class ManagerActor(BaseActor):
    pass
class WorkerActor(BaseActor):
    pass

# ==========================================
# 3. Critics
# ==========================================
class BaseCritic(nn.Module):
    def __init__(
        self,
        state_shape: tuple[int,int,int],
        goal_shape: tuple[int,int,int],
        action_shape: tuple[int,int,int],
        hidden_dim: int = 256  # 与 Backbone 默认一致
    )-> None:
        super(BaseCritic, self).__init__()
        n_s, m_s, f_dim = state_shape
        n_g, m_g, g_dim = goal_shape
        n_a, m_a, a_dim = action_shape

        combined_dim: int = f_dim + g_dim + a_dim

        # 计算线性层的输入维度
        fused_feature_dim: int = hidden_dim * 2

        # --- Q1 网络 ---
        self.backbone1: AttentionBackbone = AttentionBackbone(
            n_nodes=n_s, m_tasks=m_s, input_feature_dim=combined_dim,
            hidden_dim=hidden_dim  # 显式传入 hidden_dim
        )
        self.out_layer1: nn.Linear = nn.Linear(in_features=fused_feature_dim, out_features=1)

        # --- Q2 网络 (新增独立路径) ---
        self.backbone2: AttentionBackbone = AttentionBackbone(
            n_nodes=n_s, m_tasks=m_s, input_feature_dim=combined_dim,
            hidden_dim=hidden_dim  # 显式传入 hidden_dim
        )
        self.out_layer2: nn.Linear = nn.Linear(in_features=fused_feature_dim, out_features=1)

        # --- 初始化逻辑 ---
        self.apply(weights_init)

    def _forward_impl(self, backbone, out_layer, x) -> Tensor:
        """内部实现：特征提取 + 聚合 + 输出"""
        features: Tensor = backbone(x)
        # 全局池化策略：Mean + Max
        global_feature_mean = torch.mean(features, dim=1)
        global_feature_max, _ = torch.max(features, dim=1)
        global_feature = torch.cat([global_feature_mean, global_feature_max], dim=-1)
        q_value: Tensor = out_layer(global_feature)
        return q_value

    def forward(self, state: Tensor, goal: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        """返回 Q1 和 Q2 的值"""
        x: Tensor = torch.cat([state, goal, action], dim=-1)

        q1 = self._forward_impl(self.backbone1, self.out_layer1, x)
        q2 = self._forward_impl(self.backbone2, self.out_layer2, x)

        return q1, q2

    def q1(self, state: Tensor, goal: Tensor, action: Tensor) -> Tensor:
        """只计算 Q1 (用于 Actor 更新，节省计算量)"""
        x: Tensor = torch.cat([state, goal, action], dim=-1)
        return self._forward_impl(self.backbone1, self.out_layer1, x)

class ManagerCritic(BaseCritic):
    pass
class WorkerCritic(BaseCritic):
    pass

# ==========================================
# 5. 维度验证 (Proof of Dimensions)
# ==========================================
# ==========================================
# 5. 维度验证 (Proof of Dimensions) - 重构增强版
# ==========================================
if __name__ == "__main__":
    # ======================
    # 配置参数（模拟真实场景）
    # ======================
    BATCH_SIZE = 2
    N_NODES = 10      # 节点数
    M_TASKS = 100     # 微服务/任务数
    STATE_FEATURE_DIM = 9
    GOAL_FEATURE_DIM = 1
    ACTION_FEATURE_DIM = 1

    STATE_SHAPE = (N_NODES, M_TASKS, STATE_FEATURE_DIM)
    GOAL_SHAPE = (N_NODES, M_TASKS, GOAL_FEATURE_DIM)
    ACTION_SHAPE = (N_NODES, M_TASKS, ACTION_FEATURE_DIM)

    print("=" * 60)
    print("🧪 维度验证：SMPO-HIRO 模型结构完整性检查")
    print("=" * 60)
    print(f"配置参数：")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Nodes: {N_NODES}, Tasks: {M_TASKS}")
    print(f"  State Feature Dim: {STATE_FEATURE_DIM}")
    print(f"  Goal Feature Dim: {GOAL_FEATURE_DIM}")
    print(f"  Action Feature Dim: {ACTION_FEATURE_DIM}")
    print()

    # ======================
    # 生成模拟输入数据
    # ======================
    state = torch.randn(BATCH_SIZE, *STATE_SHAPE, dtype=torch.float32)
    global_goal = torch.randn(BATCH_SIZE, *GOAL_SHAPE, dtype=torch.float32)

    print("📥 输入数据形状：")
    print(f"  State:  {state.shape}  → (B, N, M, F)")
    print(f"  Goal:   {global_goal.shape}  → (B, N, M, F)")
    print()

    # ======================
    # 实例化模型
    # ======================
    manager_actor = ManagerActor(STATE_SHAPE, GOAL_SHAPE, ACTION_SHAPE)
    worker_actor = WorkerActor(STATE_SHAPE, GOAL_SHAPE, ACTION_SHAPE)
    manager_critic = ManagerCritic(STATE_SHAPE, GOAL_SHAPE, ACTION_SHAPE)

    print("🏗️  模型实例化完成")
    print()

    # ======================
    # 1. Manager Actor 验证
    # ======================
    print("🔹 Manager Actor 前向传播验证：")
    subgoal = manager_actor(state, global_goal)
    print(f"  输入: state={state.shape}, goal={global_goal.shape}")
    print(f"  输出: subgoal={subgoal.shape}")
    expected_subgoal_shape = (BATCH_SIZE, *ACTION_SHAPE)
    assert subgoal.shape == expected_subgoal_shape, \
        f"❌ Manager 输出维度错误！期望 {expected_subgoal_shape}，实际 {subgoal.shape}"
    print(f"  ✅ Manager 输出维度正确：{subgoal.shape} == {expected_subgoal_shape}")
    print()

    # ======================
    # 2. Worker Actor 验证
    # ======================
    print("🔹 Worker Actor 前向传播验证：")
    worker_action = worker_actor(state, subgoal)
    print(f"  输入: state={state.shape}, goal={subgoal.shape}")
    print(f"  输出: action={worker_action.shape}")
    expected_action_shape = (BATCH_SIZE, *ACTION_SHAPE)
    assert worker_action.shape == expected_action_shape, \
        f"❌ Worker 输出维度错误！期望 {expected_action_shape}，实际 {worker_action.shape}"
    print(f"  ✅ Worker 输出维度正确：{worker_action.shape} == {expected_action_shape}")
    print()

    # ======================
    # 3. Manager Critic 验证
    # ======================
    print("🔹 Manager Critic 前向传播验证：")
    q1, q2 = manager_critic(state, global_goal, subgoal)
    print(f"  输入: state={state.shape}, goal={global_goal.shape}, action={subgoal.shape}")
    print(f"  输出: Q1={q1.shape}, Q2={q2.shape}")
    expected_q_shape = (BATCH_SIZE, 1)
    assert q1.shape == expected_q_shape and q2.shape == expected_q_shape, \
        f"❌ Critic 输出维度错误！期望 {expected_q_shape}，实际 Q1={q1.shape}, Q2={q2.shape}"
    print(f"  ✅ Critic 输出维度正确：Q1={q1.shape}, Q2={q2.shape}")
    print()

    # ======================
    # 4. Q1 单独调用验证（用于 Actor 更新）
    # ======================
    print("🔹 Critic.q1() 单独调用验证：")
    q1_single = manager_critic.q1(state, global_goal, subgoal)
    print(f"  Q1 单独输出: {q1_single.shape}")
    assert q1_single.shape == expected_q_shape, \
        f"❌ Critic.q1() 输出维度错误！期望 {expected_q_shape}，实际 {q1_single.shape}"
    print(f"  ✅ Critic.q1() 输出维度正确：{q1_single.shape}")
    print()

    # ======================
    # 5. 模型参数维度一致性检查（可选增强）
    # ======================
    print("🔍 模型参数维度一致性检查（Backbone）：")
    backbone = manager_actor.backbone
    print(f"  Backbone 输入投影层: {backbone.input_proj.in_features} → {backbone.input_proj.out_features}")
    expected_input_proj_in = N_NODES * (STATE_FEATURE_DIM + GOAL_FEATURE_DIM)
    assert backbone.input_proj.in_features == expected_input_proj_in, \
        f"❌ Backbone 输入投影维度错误！期望 {expected_input_proj_in}，实际 {backbone.input_proj.in_features}"
    print(f"  ✅ 输入投影: {backbone.input_proj.in_features} → {backbone.input_proj.out_features} (正确)")
    print()

    # ======================
    # 6. 输出层维度检查（Actor）
    # ======================
    print("🔍 Actor 输出层维度检查：")
    output_layer = manager_actor.output_layer
    expected_output_dim = N_NODES * ACTION_FEATURE_DIM
    assert output_layer.out_features == expected_output_dim, \
        f"❌ Actor 输出层维度错误！期望 {expected_output_dim}，实际 {output_layer.out_features}"
    print(f"  ✅ 输出层: {output_layer.in_features} → {output_layer.out_features} (正确)")
    print()

    # ======================
    # 7. Transformer 输出维度检查 (修正版)
    # ======================
    print("🔍 Transformer 输出维度检查：")
    # 修正：Backbone 期望的是原始的 4D 输入 (B, Nodes, Tasks, Features)
    # 之前的 dummy_input 是 3D 的，导致了 permute 错误
    input_dim = STATE_FEATURE_DIM + GOAL_FEATURE_DIM
    dummy_input = torch.randn(BATCH_SIZE, N_NODES, M_TASKS, input_dim)

    with torch.no_grad():
        transformer_out = backbone(dummy_input)

    print(f"  Transformer 输入: {dummy_input.shape} (B, N, M, F)")
    print(f"  Transformer 输出: {transformer_out.shape}")

    # Transformer 会将输入重塑为 (B, M, N*F) 进行处理，保持 M 为序列长度
    expected_trans_out = (BATCH_SIZE, M_TASKS, backbone.hidden_dim)

    assert transformer_out.shape == expected_trans_out, \
        f"❌ Transformer 输出维度错误！期望 {expected_trans_out}，实际 {transformer_out.shape}"
    print(f"  ✅ Transformer 输出维度正确：{transformer_out.shape}")



    # ======================
    # 总结
    # ======================
    print("=" * 60)
    print("🎉 所有维度验证通过！模型结构完整，符合设计预期。")
    print("=" * 60)

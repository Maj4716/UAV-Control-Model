import torch
import torch.nn as nn
import torch.nn.functional as F

from loitering_munition_damage_twin.stage0.component_supervision import CRITICAL_COMPONENT_IDS


# [munition, task(K/M/F/C), ordinal level(>=1/>=2)]
# Stage-0 v2 currently defines Small/K2 and Small/C2 as structural zeros.
DEFAULT_ORDINAL_APPLICABILITY = (
    ((True, False), (True, True), (True, True), (True, False)),
    ((True, True), (True, True), (True, True), (True, True)),
    ((True, True), (True, True), (True, True), (True, True)),
    ((True, True), (True, True), (True, True), (True, True)),
)

_COMPONENT_INDEX = {
    component_id: index
    for index, component_id in enumerate(CRITICAL_COMPONENT_IDS)
}


def _component_probability_or(
        probabilities: list[torch.Tensor],
        rho: float = 0.5) -> torch.Tensor:
    stacked = torch.stack(probabilities, dim=-1)
    independent = 1.0 - torch.prod(
        1.0 - stacked, dim=-1)
    maximum = torch.max(stacked, dim=-1).values
    return float(rho) * maximum + (
        1.0 - float(rho)) * independent


def _component_probability_and(
        probabilities: list[torch.Tensor]) -> torch.Tensor:
    return torch.prod(
        torch.stack(probabilities, dim=-1), dim=-1)


def _component_probability_ratio(
        probabilities: list[torch.Tensor],
        threshold: float) -> torch.Tensor:
    stacked = torch.stack(probabilities, dim=-1)
    mean_count = stacked.sum(dim=-1)
    variance = (
        stacked * (1.0 - stacked)).sum(
            dim=-1).clamp_min(1e-6)
    target_count = float(threshold) * stacked.shape[-1]
    z_score = (
        mean_count - target_count + 0.5
    ) / torch.sqrt(variance)
    return torch.sigmoid(
        torch.clamp(1.7 * z_score, min=-60.0, max=60.0))


def component_probabilities_to_ordinal(
        component_probabilities: torch.Tensor) -> torch.Tensor:
    """Apply the simulator's differentiable damage-tree rule topology.

    Args:
        component_probabilities: ``(N, 51)`` probabilities ordered exactly as
            :data:`component_supervision.CRITICAL_COMPONENT_IDS`.
    Returns:
        ``(N, 4, 2)`` cumulative probabilities for K/M/F/C.
    """
    if (
        component_probabilities.ndim != 2
        or component_probabilities.shape[1]
        != len(CRITICAL_COMPONENT_IDS)
    ):
        raise ValueError(
            "component_probabilities must have shape "
            f"(N,{len(CRITICAL_COMPONENT_IDS)}).")
    probabilities = component_probabilities.clamp(0.0, 1.0)

    def values(component_ids):
        return [
            probabilities[:, _COMPONENT_INDEX[int(component_id)]]
            for component_id in component_ids
        ]

    k1 = values((3,))[0]
    k2 = values((46,))[0]
    m1 = _component_probability_or([
        _component_probability_ratio(
            values(tuple(range(6, 18))), 0.2),
        _component_probability_or(values((30, 31))),
    ])
    m2 = _component_probability_or([
        _component_probability_or(values((1, 2, 3))),
        _component_probability_or(values((4, 5))),
        _component_probability_ratio(
            values(tuple(range(32, 40))), 0.25),
        values((58,))[0],
    ])
    f1 = _component_probability_or([
        _component_probability_or(
            values((50, 51, 52, 53, 54, 47, 48))),
        values((41,))[0] * (1.0 - values((42,))[0]),
        (
            values((43,))[0] * (1.0 - values((44,))[0])
            + values((44,))[0] * (1.0 - values((43,))[0])
        ),
        values((60,))[0] * (1.0 - values((59,))[0]),
    ])
    f2 = _component_probability_or([
        _component_probability_and(values((45, 49))),
        _component_probability_and(values((41, 42))),
        _component_probability_and(values((43, 44))),
        _component_probability_and(values((59, 60))),
    ])
    c1 = _component_probability_ratio(
        values(tuple(range(58, 68))), 0.2)
    c2 = _component_probability_ratio(
        values(tuple(range(58, 68))), 0.6)

    ordinal = []
    for first, second, nested in (
            (k1, k2, False),
            (m1, m2, False),
            (f1, f2, False),
            (c1, c2, True)):
        ge1 = (
            first
            if nested
            else 1.0 - (1.0 - first) * (1.0 - second)
        )
        ge2 = torch.minimum(second, ge1)
        ordinal.append(torch.stack((ge1, ge2), dim=-1))
    return torch.stack(ordinal, dim=1).clamp(0.0, 1.0)


class MonotoneOrdinalProjection(nn.Module):
    """Produce two cumulative logits with P(level>=2) <= P(level>=1).

    ``raw[..., 0]`` is the level-1 logit.  The second raw value parameterizes
    a non-negative logit gap, so monotonicity is guaranteed by construction
    instead of repaired by a loss penalty or an inference-only clamp.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.raw = nn.Linear(in_features, 2)

    def forward(self, x):
        raw = self.raw(x)
        logit_ge1 = raw[..., :1]
        logit_ge2 = logit_ge1 - F.softplus(raw[..., 1:2])
        return torch.cat((logit_ge1, logit_ge2), dim=-1)


class NominalSoftmaxOrdinalProjection(nn.Module):
    """Learn L0/L1/L2 directly and expose monotone cumulative logits.

    The proper three-class simplex gives the narrow middle class an explicit
    logit.  The public output remains ``P(level>=1)`` and ``P(level>=2)``, so
    the loss, calibration and deployment contracts do not change.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.raw = nn.Linear(in_features, 3)

    def forward(self, x):
        class_probabilities = torch.softmax(self.raw(x), dim=-1)
        probability_ge1 = class_probabilities[..., 1:].sum(
            dim=-1, keepdim=True)
        probability_ge2 = class_probabilities[..., 2:3]
        cumulative = torch.cat(
            (probability_ge1, probability_ge2), dim=-1)
        cumulative = torch.clamp(cumulative, min=1e-6, max=1.0 - 1e-6)
        return torch.log(cumulative) - torch.log1p(-cumulative)


def _make_ordinal_projection(in_features: int, parameterization: str):
    normalized = str(parameterization).strip().lower()
    if normalized == "cumulative_logits":
        return MonotoneOrdinalProjection(in_features)
    if normalized == "nominal_softmax":
        return NominalSoftmaxOrdinalProjection(in_features)
    raise ValueError(
        "ordinal_parameterization must be 'cumulative_logits' or "
        f"'nominal_softmax', got {parameterization!r}.")


class ResidualLinearGELU(nn.Module):
    """
    自适应的残差全连接层。
    只有当 in_features == out_features 时，才会施加 + x 跨越连接（残差）。
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (in_features == out_features)

    def forward(self, x):
        identity = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.gelu(out)
        out = self.dropout(out)
        if self.use_residual:
            return out + identity
        return out


class ComponentMechanismAuxiliaryHead(nn.Module):
    """Predict dense fragment/shock probabilities for critical components."""

    def __init__(self, in_features: int, component_count: int):
        super().__init__()
        self.component_count = int(component_count)
        self.network = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 2 * self.component_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)
        return logits.view(
            features.shape[0], 2, self.component_count)


class IndependentComponentPhysicsBranch(nn.Module):
    """Predict calibrated component risks without perturbing the direct path.

    A34 attached the 102 component targets to the shared task representation.
    Validation showed useful complementary information, but also broad
    negative transfer into the direct K/C rankings.  This branch has a
    separate munition embedding, encoder and per-munition experts so its
    gradients cannot alter a warm-started direct surrogate.
    """

    def __init__(self, in_features: int, num_munitions: int,
                 component_count: int, munition_emb_dim: int = 16):
        super().__init__()
        if int(in_features) <= 0:
            raise ValueError("in_features must be positive.")
        if int(num_munitions) <= 0:
            raise ValueError("num_munitions must be positive.")
        if int(munition_emb_dim) <= 0:
            raise ValueError(
                "component munition embedding dimension must be positive.")
        self.num_munitions = int(num_munitions)
        self.component_count = int(component_count)
        self.munition_embedding = nn.Embedding(
            self.num_munitions, int(munition_emb_dim))
        encoded_dim = 384
        self.encoder = nn.Sequential(
            nn.Linear(
                int(in_features) + int(munition_emb_dim),
                encoded_dim),
            nn.LayerNorm(encoded_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(encoded_dim, encoded_dim),
            nn.LayerNorm(encoded_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.experts = nn.ModuleList([
            ComponentMechanismAuxiliaryHead(
                encoded_dim, self.component_count)
            for _ in range(self.num_munitions)
        ])

    def forward(self, features: torch.Tensor,
                munition_id: torch.Tensor) -> torch.Tensor:
        embedding = self.munition_embedding(munition_id)
        encoded = self.encoder(
            torch.cat((features, embedding), dim=-1))
        all_logits = torch.stack(
            [expert(encoded) for expert in self.experts],
            dim=1,
        )
        gather_index = munition_id.view(
            -1, 1, 1, 1).expand(
                -1, 1, 2, self.component_count)
        return all_logits.gather(
            1, gather_index).squeeze(1)

    def initialize_output_priors(
            self, component_means_by_munition: torch.Tensor) -> None:
        """Start each expert at its train-only empirical soft-label prior."""
        means = torch.as_tensor(
            component_means_by_munition,
            dtype=torch.float32,
        )
        expected_shape = (
            self.num_munitions, 2, self.component_count)
        if tuple(means.shape) != expected_shape:
            raise ValueError(
                "component_means_by_munition must have shape "
                f"{expected_shape}, got {tuple(means.shape)}")
        means = means.clamp(1e-5, 1.0 - 1e-5)
        prior_logits = (
            torch.log(means) - torch.log1p(-means)
        )
        with torch.no_grad():
            for munition_index, expert in enumerate(self.experts):
                output_layer = expert.network[-1]
                output_layer.weight.zero_()
                output_layer.bias.copy_(
                    prior_logits[munition_index].reshape(-1).to(
                        device=output_layer.bias.device,
                        dtype=output_layer.bias.dtype,
                    )
                )


class TerminalFourierExpansion(nn.Module):
    """Expand normalized terminal state with deterministic multiscale features.

    The public model input remains the same 13 observable terminal-state
    variables.  Only the position/velocity channels are expanded inside the
    network, so no active-sampling metadata or deployment-only signal is
    introduced.
    """

    def __init__(self, in_dim: int, feature_indices=(0, 1, 2, 3, 4, 5),
                 frequencies=(1.0, 2.0, 4.0, 8.0)):
        super().__init__()
        indices = tuple(int(index) for index in feature_indices)
        if not indices:
            raise ValueError("feature_indices must not be empty.")
        if min(indices) < 0 or max(indices) >= int(in_dim):
            raise ValueError(
                "feature_indices must refer to valid terminal-state inputs.")
        frequency_values = tuple(float(value) for value in frequencies)
        if not frequency_values or any(value <= 0.0 for value in frequency_values):
            raise ValueError("frequencies must contain positive values.")
        self.in_dim = int(in_dim)
        self.register_buffer(
            "feature_indices",
            torch.tensor(indices, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "frequencies",
            torch.tensor(frequency_values, dtype=torch.float32),
            persistent=True,
        )
        self.out_dim = (
            self.in_dim
            + 2 * len(indices) * len(frequency_values)
        )

    def forward(self, x):
        selected = x.index_select(1, self.feature_indices)
        angles = (
            selected.unsqueeze(-1)
            * self.frequencies.view(1, 1, -1)
            * (2.0 * torch.pi)
        )
        fourier = torch.cat(
            (torch.sin(angles), torch.cos(angles)), dim=-1
        ).flatten(start_dim=1)
        return torch.cat((x, fourier), dim=-1)


class BoundedCellResidualAdapter(nn.Module):
    """A zero-initialized bounded scalar correction for one ordinal head."""

    def __init__(self, in_dim: int, hidden_dim: int,
                 maximum_absolute_logit: float):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive.")
        if float(maximum_absolute_logit) <= 0.0:
            raise ValueError("maximum_absolute_logit must be positive.")
        self.maximum_absolute_logit = float(maximum_absolute_logit)
        self.hidden = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.output = nn.Linear(int(hidden_dim), 1)
        # The warm-start model must reproduce its source checkpoint exactly
        # before the first optimization step.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, expanded_terminal_state):
        raw = self.output(self.hidden(expanded_terminal_state))
        return self.maximum_absolute_logit * torch.tanh(raw)


class MunitionHead(nn.Module):
    """
    独立弹型的毁伤预测专家（Expert / Head）。
    内部包含 K(核心), M(机动), F(火力), C(乘员) 四条独立毁伤推断链路。
    其中 K 分支的预测概率先验会作为级联被送入 MFC 的推断中。

    [R19] M 分支单独加深为两层 MLP (192 → 96), 其它分支维持单层 128.
       动机: M-level 触发条件依赖于命中 tracks/idlers/drives/power 等分布在车体
       底部各处的子系统集合, 对 Small 弹种而言这是个"高几何敏感、低能量裕度"的判别问题
       —— 单层 128 隐藏的 m_branch 在共享底座 h(512维) 出来后几乎是个线性投影,
       难以表达"该 Small 样本的弹道虽然像 Heavy×M=2 但弹种能量不够触发 M-positive"
       这种条件性几何推理. 加深一层 + 加宽到 192 给 m_branch 留出空间, 同时不影响
       表现已经稳的 K/F/C 分支.
    """
    def __init__(self, in_dim: int = 512,
                 use_k_cascade: bool = True,
                 deep_m_branch: bool = True,
                 ordinal_parameterization: str = "cumulative_logits"):
        super().__init__()
        self.use_k_cascade = use_k_cascade
        self.ordinal_parameterization = str(
            ordinal_parameterization).strip().lower()
        # Branch K
        self.k_branch = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            _make_ordinal_projection(
                128, self.ordinal_parameterization),
        )

        # 级联维度
        cascade_dim = in_dim + 2 if use_k_cascade else in_dim

        # [R19] M 分支加深: cascade_dim → 192 → 96 → 2; F/C 保持原 128 单层
        if deep_m_branch:
            self.m_branch = nn.Sequential(
                nn.Linear(cascade_dim, 192), nn.GELU(), nn.Dropout(0.10),
                nn.Linear(192, 96),          nn.GELU(),
                _make_ordinal_projection(
                    96, self.ordinal_parameterization),
            )
        else:
            self.m_branch = nn.Sequential(
                nn.Linear(cascade_dim, 128), nn.GELU(),
                _make_ordinal_projection(
                    128, self.ordinal_parameterization),
            )
        self.f_branch = nn.Sequential(
            nn.Linear(cascade_dim, 128), nn.GELU(),
            _make_ordinal_projection(
                128, self.ordinal_parameterization),
        )
        self.c_branch = nn.Sequential(
            nn.Linear(cascade_dim, 128), nn.GELU(),
            _make_ordinal_projection(
                128, self.ordinal_parameterization),
        )

    def forward(self, h):
        # 优先推导 K 逻辑
        k_logits = self.k_branch(h)
        # 用 sigmoid 抽取 [0,1] 的平滑概率供给后续，并先做物理保序投影：
        # P(K>=2) 不得高于 P(K>=1)，避免把自相矛盾的 K 特征送入 M/F/C 级联。
        p_k = torch.sigmoid(k_logits)
        p_k1 = p_k[:, :1]
        p_k2 = torch.minimum(p_k[:, 1:2], p_k1)
        p_k_cascade = torch.cat([p_k1, p_k2], dim=-1)

        # K probabilities are business outputs.  Do not let M/F/C losses turn
        # them into an unconstrained hidden communication channel.  The shared
        # bottom still receives gradients from every task; only the explicit
        # probability cascade is stop-gradient.
        h_cascade = (
            torch.cat([h, p_k_cascade.detach()], dim=-1)
            if self.use_k_cascade else h
        )
        m_logits = self.m_branch(h_cascade)
        f_logits = self.f_branch(h_cascade)
        c_logits = self.c_branch(h_cascade)

        # 返回形状: (N, 4_branches, 2_levels)
        return torch.stack([k_logits, m_logits, f_logits, c_logits], dim=1)


class DamageAssessmentMTL(nn.Module):
    """
    遵循 V4 方案修改：多任务混合 Conditioned Shared Bottom 网络架构。
    [P1-C] 在 shared_bottom 入口注入 Munition Embedding，让底座表征能按弹型条件化，
    缓解稀缺弹型（Small/Med）特征被 Heavy 样本淹没的问题。
    [R19] 在 head 入口再注入一次 mun_emb (skip-connection 弹型路径),
       同时把 emb_dim 从 8 提到 16. 单点的 mun_emb 经 3 层共享底座 + dropout 后
       会被稀释, head 拿到的 h 实际是"近似与弹型无关的几何特征". 让 head 通过
       skip-connection 直接看到 mun_emb, 可以在不重训共享底座的情况下显著提升
       Small head 在 M task 上的弹型条件化判别能力 —— 这正是 Small × M = L0 边界
       样本被错推到 L≥1/L=2 的根因 (混淆矩阵 4.9% L0→L2 双置信度误识别).
    """
    def __init__(self, in_dim: int = 13, num_munitions: int = 4,
                 base_input_dim: int = None,
                 munition_emb_dim: int = 16,
                 use_munition_embedding: bool = True,
                 use_munition_experts: bool = True,
                 use_physics_skip: bool = True,
                 use_k_cascade: bool = True,
                 deep_m_branch: bool = True,
                 ordinal_parameterization: str = "cumulative_logits",
                 use_mechanism_decomposition: bool = False,
                 use_mechanism_auxiliary_heads: bool = False,
                 mechanism_encoder_mode: str = "shared",
                 use_component_auxiliary_heads: bool = False,
                 component_ids=None,
                 component_branch_mode: str = "shared_auxiliary",
                 component_branch_munition_emb_dim: int = 16,
                 component_tree_fusion_alpha=None,
                 residual_adapter_cells=None,
                 residual_adapter_hidden_dim: int = 64,
                 residual_adapter_feature_indices=(0, 1, 2, 3, 4, 5),
                 residual_adapter_frequencies=(1.0, 2.0, 4.0, 8.0),
                 residual_adapter_max_logit: float = 2.0,
                 ordinal_applicability=None):
        # Stage-0: 仅接收终端时刻可观测状态。瞄准点派生的 d_los/c_imp
        # 属于主动采样策略元数据，禁止进入代理模型以避免目标选择泄漏。
        # 训练/导出入口仍会显式传入 len(feature_columns)。
        super().__init__()
        self.in_dim = int(in_dim)
        self.base_input_dim = (
            self.in_dim
            if base_input_dim is None else int(base_input_dim)
        )
        if not (0 < self.base_input_dim <= self.in_dim):
            raise ValueError(
                "base_input_dim must satisfy "
                f"0 < base_input_dim <= in_dim ({self.in_dim}).")
        # [P1-C + R19] Munition Embedding: 16 维 (R19 由 8 提升)
        self.use_munition_embedding = use_munition_embedding and munition_emb_dim > 0
        self.use_munition_experts = use_munition_experts
        self.use_physics_skip = use_physics_skip
        self.use_mechanism_decomposition = bool(
            use_mechanism_decomposition)
        self.use_mechanism_auxiliary_heads = bool(
            use_mechanism_auxiliary_heads)
        self.mechanism_encoder_mode = str(
            mechanism_encoder_mode).strip().lower()
        if self.mechanism_encoder_mode not in {"shared", "independent"}:
            raise ValueError(
                "mechanism_encoder_mode must be 'shared' or 'independent'.")
        self.use_component_auxiliary_heads = bool(
            use_component_auxiliary_heads)
        self.component_branch_mode = str(
            component_branch_mode).strip().lower()
        if self.component_branch_mode not in {
            "shared_auxiliary", "independent_experts"
        }:
            raise ValueError(
                "component_branch_mode must be 'shared_auxiliary' or "
                "'independent_experts'.")
        if (
            self.use_mechanism_decomposition
            and self.use_mechanism_auxiliary_heads
        ):
            raise ValueError(
                "Mechanism decomposition and auxiliary-only mechanism heads "
                "are mutually exclusive.")
        self.has_mechanism_outputs = bool(
            self.use_mechanism_decomposition
            or self.use_mechanism_auxiliary_heads)
        if (
            self.mechanism_encoder_mode == "independent"
            and not self.has_mechanism_outputs
        ):
            raise ValueError(
                "Independent mechanism encoders require mechanism outputs.")
        if (
            self.has_mechanism_outputs
            and self.use_component_auxiliary_heads
        ):
            raise ValueError(
                "Task-level mechanism heads and component auxiliary heads "
                "are intentionally isolated experimental factors.")
        resolved_component_ids = tuple(
            CRITICAL_COMPONENT_IDS
            if component_ids is None else
            (int(value) for value in component_ids)
        )
        if (
            self.use_component_auxiliary_heads
            and resolved_component_ids != CRITICAL_COMPONENT_IDS
        ):
            raise ValueError(
                "component_ids must match the immutable damage-tree "
                "component order.")
        self.component_ids = resolved_component_ids
        self.has_component_outputs = bool(
            self.use_component_auxiliary_heads)
        if (
            not self.has_component_outputs
            and self.component_branch_mode != "shared_auxiliary"
        ):
            raise ValueError(
                "An independent component branch requires "
                "use_component_auxiliary_heads=True.")
        applicability = (
            DEFAULT_ORDINAL_APPLICABILITY
            if ordinal_applicability is None else ordinal_applicability
        )
        applicability_tensor = torch.as_tensor(
            applicability, dtype=torch.bool)
        if tuple(applicability_tensor.shape) != (num_munitions, 4, 2):
            raise ValueError(
                "ordinal_applicability must have shape "
                f"({num_munitions}, 4, 2), got "
                f"{tuple(applicability_tensor.shape)}")
        self.register_buffer(
            "ordinal_applicability", applicability_tensor, persistent=True)
        raw_component_fusion = (
            torch.zeros(
                num_munitions, 4, 2, dtype=torch.float32)
            if component_tree_fusion_alpha is None
            else torch.as_tensor(
                component_tree_fusion_alpha,
                dtype=torch.float32)
        )
        if tuple(raw_component_fusion.shape) != (
                num_munitions, 4, 2):
            raise ValueError(
                "component_tree_fusion_alpha must have shape "
                f"({num_munitions},4,2), got "
                f"{tuple(raw_component_fusion.shape)}")
        if (
            not torch.isfinite(raw_component_fusion).all()
            or (raw_component_fusion < 0.0).any()
            or (raw_component_fusion > 1.0).any()
        ):
            raise ValueError(
                "component_tree_fusion_alpha must be finite and in [0,1].")
        if (
            raw_component_fusion[~applicability_tensor] != 0.0
        ).any():
            raise ValueError(
                "component_tree_fusion_alpha must be zero for structural "
                "ordinal cells.")
        if (
            not self.has_component_outputs
            and (raw_component_fusion > 0.0).any()
        ):
            raise ValueError(
                "Component-tree fusion requires component outputs.")
        # Alpha is immutable model configuration rather than learned state.
        # Non-persistent registration preserves strict loading of A31/A34
        # checkpoints created before this fusion path existed.
        self.register_buffer(
            "component_tree_fusion_alpha",
            raw_component_fusion,
            persistent=False,
        )
        self.component_tree_fusion_enabled = bool(
            torch.any(raw_component_fusion > 0.0).item())
        normalized_adapter_cells = []
        seen_adapter_cells = set()
        for raw_cell in (residual_adapter_cells or []):
            if not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 3:
                raise ValueError(
                    "Each residual adapter cell must be "
                    "[munition_id, task_id, ordinal_level_index].")
            cell = tuple(int(value) for value in raw_cell)
            munition_index, task_index, level_index = cell
            if not (0 <= munition_index < num_munitions):
                raise ValueError(f"Invalid residual adapter munition: {cell}")
            if not (0 <= task_index < 4 and 0 <= level_index < 2):
                raise ValueError(f"Invalid residual adapter task/level: {cell}")
            if not bool(applicability_tensor[cell].item()):
                raise ValueError(
                    f"Residual adapter cannot target structural zero: {cell}")
            if cell in seen_adapter_cells:
                raise ValueError(f"Duplicate residual adapter cell: {cell}")
            seen_adapter_cells.add(cell)
            normalized_adapter_cells.append(cell)
        self.residual_adapter_cells = tuple(normalized_adapter_cells)
        if self.use_mechanism_decomposition and self.residual_adapter_cells:
            raise ValueError(
                "Mechanism decomposition cannot be combined with legacy "
                "post-fusion residual adapters; the latter would violate the "
                "fixed fragment/shock OR contract.")
        self.munition_emb_dim = munition_emb_dim if self.use_munition_embedding else 0
        self.mun_emb = (
            nn.Embedding(num_munitions, self.munition_emb_dim)
            if self.use_munition_embedding else None
        )
        # 入口维度扩展为 in_dim + emb_dim
        augmented_in = self.base_input_dim + self.munition_emb_dim
        # ================== 1. 共享基底提取 (Shared Bottom) ==================
        self.shared_1 = ResidualLinearGELU(augmented_in, 256, dropout=0.1)
        self.shared_2 = ResidualLinearGELU(256, 512, dropout=0.2)
        self.shared_3 = ResidualLinearGELU(512, 512, dropout=0.2)

        # Fragment geometry is sparse and discontinuous, while shock damage
        # is a smooth distance/armor function.  A shared bottom lets the much
        # easier shock regression dominate its representation.  The optional
        # independent mode gives both physical mechanisms their own complete
        # encoder while retaining the legacy shared mode for old artifacts.
        if self.mechanism_encoder_mode == "independent":
            self.fragment_shared_1 = ResidualLinearGELU(
                augmented_in, 256, dropout=0.1)
            self.fragment_shared_2 = ResidualLinearGELU(
                256, 512, dropout=0.2)
            self.fragment_shared_3 = ResidualLinearGELU(
                512, 512, dropout=0.2)
            self.shock_shared_1 = ResidualLinearGELU(
                augmented_in, 256, dropout=0.1)
            self.shock_shared_2 = ResidualLinearGELU(
                256, 512, dropout=0.2)
            self.shock_shared_3 = ResidualLinearGELU(
                512, 512, dropout=0.2)
        else:
            self.fragment_shared_1 = None
            self.fragment_shared_2 = None
            self.fragment_shared_3 = None
            self.shock_shared_1 = None
            self.shock_shared_2 = None
            self.shock_shared_3 = None

        # ================== 2. 独立专家路由 (Task-Specific Heads) ============
        # [R19] head in_dim 扩为 512 + munition_emb_dim, 让每个分支都能直接读取
        # 弹型嵌入 (skip-connection), 不再依赖共享底座是否成功"记住"了弹型.
        self.physics_skip_dim = 64 if self.use_physics_skip else 0
        self.physics_skip = (
            nn.Sequential(
                nn.Linear(self.base_input_dim, self.physics_skip_dim),
                nn.LayerNorm(self.physics_skip_dim),
                nn.GELU(),
                nn.Dropout(0.05),
            )
            if self.use_physics_skip else None
        )
        if (
            self.mechanism_encoder_mode == "independent"
            and self.use_physics_skip
        ):
            def mechanism_skip():
                return nn.Sequential(
                    nn.Linear(self.base_input_dim, self.physics_skip_dim),
                    nn.LayerNorm(self.physics_skip_dim),
                    nn.GELU(),
                    nn.Dropout(0.05),
                )
            self.fragment_physics_skip = mechanism_skip()
            self.shock_physics_skip = mechanism_skip()
        else:
            self.fragment_physics_skip = None
            self.shock_physics_skip = None
        if (
            self.use_mechanism_decomposition
            and self.mechanism_encoder_mode == "independent"
        ):
            # The direct shared path is not part of fixed-OR inference in
            # this mode.  Keep the modules in the state schema for a simple
            # implementation, but exclude their unused parameters from the
            # optimizer and parameter-count contract.
            for unused_module in (
                self.shared_1,
                self.shared_2,
                self.shared_3,
                self.physics_skip,
            ):
                if unused_module is not None:
                    for parameter in unused_module.parameters():
                        parameter.requires_grad_(False)
        head_in = 512 + self.munition_emb_dim + self.physics_skip_dim
        head_kwargs = {
            "in_dim": head_in,
            "use_k_cascade": use_k_cascade,
            "deep_m_branch": deep_m_branch,
            "ordinal_parameterization": ordinal_parameterization,
        }
        if self.has_mechanism_outputs:
            if self.use_munition_experts:
                self.fragment_heads = nn.ModuleList([
                    MunitionHead(**head_kwargs)
                    for _ in range(num_munitions)
                ])
                self.shock_heads = nn.ModuleList([
                    MunitionHead(**head_kwargs)
                    for _ in range(num_munitions)
                ])
                self.fragment_shared_head = None
                self.shock_shared_head = None
            else:
                self.fragment_heads = None
                self.shock_heads = None
                self.fragment_shared_head = MunitionHead(**head_kwargs)
                self.shock_shared_head = MunitionHead(**head_kwargs)
        else:
            self.fragment_heads = None
            self.shock_heads = None
            self.fragment_shared_head = None
            self.shock_shared_head = None

        if self.use_mechanism_decomposition:
            self.heads = None
            self.shared_head = None
        else:
            if self.use_munition_experts:
                self.heads = nn.ModuleList([
                    MunitionHead(**head_kwargs)
                    for _ in range(num_munitions)
                ])
                self.shared_head = None
            else:
                self.heads = None
                self.shared_head = MunitionHead(**head_kwargs)

        self.component_auxiliary_head = (
            ComponentMechanismAuxiliaryHead(
                head_in, len(self.component_ids))
            if (
                self.use_component_auxiliary_heads
                and self.component_branch_mode == "shared_auxiliary"
            ) else None
        )
        self.independent_component_branch = (
            IndependentComponentPhysicsBranch(
                in_features=in_dim,
                num_munitions=num_munitions,
                component_count=len(self.component_ids),
                munition_emb_dim=int(
                    component_branch_munition_emb_dim),
            )
            if (
                self.use_component_auxiliary_heads
                and self.component_branch_mode == "independent_experts"
            ) else None
        )

        self.residual_feature_expansion = None
        self.residual_adapters = nn.ModuleDict()
        if self.residual_adapter_cells:
            self.residual_feature_expansion = TerminalFourierExpansion(
                in_dim=self.base_input_dim,
                feature_indices=residual_adapter_feature_indices,
                frequencies=residual_adapter_frequencies,
            )
            for munition_index, task_index, level_index in (
                    self.residual_adapter_cells):
                key = (
                    f"munition{munition_index}_task{task_index}_"
                    f"level{level_index}"
                )
                self.residual_adapters[key] = BoundedCellResidualAdapter(
                    in_dim=self.residual_feature_expansion.out_dim,
                    hidden_dim=residual_adapter_hidden_dim,
                    maximum_absolute_logit=residual_adapter_max_logit,
                )
            adapter_munitions = [
                cell[0] for cell in self.residual_adapter_cells
            ]
            adapter_basis = torch.zeros(
                len(self.residual_adapter_cells), 4, 2,
                dtype=torch.float32,
            )
            for adapter_index, (_, task_index, level_index) in enumerate(
                    self.residual_adapter_cells):
                adapter_basis[adapter_index, task_index, level_index] = 1.0
            self.register_buffer(
                "residual_adapter_munitions",
                torch.tensor(adapter_munitions, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "residual_adapter_basis",
                adapter_basis,
                persistent=True,
            )

    @staticmethod
    def _route_heads(h_with_e, munition_id, heads, shared_head):
        if heads is None:
            return shared_head(h_with_e)
        all_logits = torch.stack(
            [head(h_with_e) for head in heads], dim=1)
        idx = munition_id.view(-1, 1, 1, 1).expand(-1, 1, 4, 2)
        return all_logits.gather(1, idx).squeeze(1)

    def _encode(self, x, munition_id):
        base_x = x[..., :self.base_input_dim]
        # [P1-C] 弹型嵌入与特征拼接 (底座入口路径)
        e = self.mun_emb(munition_id) if self.mun_emb is not None else None
        x_aug = (
            torch.cat([base_x, e], dim=-1)
            if e is not None else base_x
        )
        # --- 公共底座特征提取 ---
        h = self.shared_1(x_aug)
        h = self.shared_2(h)
        h = self.shared_3(h)
        # [R19] head 入口 skip-connection: 把 mun_emb 拼到 h 上,
        # 让每个 head 都能 fresh 地看到弹型信号, 不被共享底座稀释.
        head_inputs = [h]
        if e is not None:
            head_inputs.append(e)
        if self.physics_skip is not None:
            head_inputs.append(self.physics_skip(base_x))
        return torch.cat(head_inputs, dim=-1)

    def _encode_independent_mechanisms(self, x, munition_id):
        """Encode fragment and shock physics without shared hidden layers."""
        if self.mechanism_encoder_mode != "independent":
            shared = self._encode(x, munition_id)
            return shared, shared
        base_x = x[..., :self.base_input_dim]
        e = self.mun_emb(munition_id) if self.mun_emb is not None else None
        x_aug = (
            torch.cat([base_x, e], dim=-1)
            if e is not None else base_x
        )

        fragment = self.fragment_shared_1(x_aug)
        fragment = self.fragment_shared_2(fragment)
        fragment = self.fragment_shared_3(fragment)
        shock = self.shock_shared_1(x_aug)
        shock = self.shock_shared_2(shock)
        shock = self.shock_shared_3(shock)

        fragment_inputs = [fragment]
        shock_inputs = [shock]
        if e is not None:
            fragment_inputs.append(e)
            shock_inputs.append(e)
        if self.use_physics_skip:
            fragment_inputs.append(
                self.fragment_physics_skip(base_x))
            shock_inputs.append(
                self.shock_physics_skip(base_x))
        return (
            torch.cat(fragment_inputs, dim=-1),
            torch.cat(shock_inputs, dim=-1),
        )

    def _apply_residual_adapters(self, final_logits, x, munition_id):
        if self.residual_adapters:
            expanded_terminal_state = self.residual_feature_expansion(
                x[..., :self.base_input_dim])
            residual_delta = torch.zeros_like(final_logits)
            for adapter_index, adapter in enumerate(
                    self.residual_adapters.values()):
                active_munition = (
                    munition_id
                    == self.residual_adapter_munitions[adapter_index]
                ).to(final_logits.dtype)
                scalar_delta = (
                    adapter(expanded_terminal_state).squeeze(-1)
                    * active_munition
                )
                residual_delta = residual_delta + (
                    scalar_delta.view(-1, 1, 1)
                    * self.residual_adapter_basis[adapter_index]
                    .view(1, 4, 2)
                )
            return final_logits + residual_delta
        return final_logits

    def _mask_and_monotonize(self, logits, munition_id):
        applicability = self.ordinal_applicability[munition_id]
        structural_zero = torch.full_like(logits, -30.0)
        masked_logits = torch.where(
            applicability, logits, structural_zero)
        masked_l2 = torch.minimum(
            masked_logits[..., 1:2], masked_logits[..., 0:1])
        return torch.cat((masked_logits[..., 0:1], masked_l2), dim=-1)

    def forward_with_mechanisms(self, x, munition_id):
        """Return combined, fragment and shock ordinal logits.

        The combined probability is the simulator's physically motivated
        independent-mechanism union:

            P(damage) = 1 - (1 - P(fragment)) (1 - P(shock)).

        Mechanism outputs are returned for fixed-OR decomposition or for
        training-only auxiliary heads.  In auxiliary-only mode the direct
        combined head remains authoritative at inference.
        """
        h_with_e = (
            self._encode(x, munition_id)
            if (
                not self.use_mechanism_decomposition
                or self.mechanism_encoder_mode == "shared"
            ) else None
        )
        if (
            self.has_mechanism_outputs
            and self.mechanism_encoder_mode == "independent"
        ):
            fragment_h, shock_h = (
                self._encode_independent_mechanisms(x, munition_id)
            )
        else:
            fragment_h, shock_h = h_with_e, h_with_e
        if not self.use_mechanism_decomposition:
            combined = self._route_heads(
                h_with_e, munition_id, self.heads, self.shared_head)
            combined = self._apply_residual_adapters(
                combined, x, munition_id)
            combined = self._mask_and_monotonize(
                combined, munition_id)
            if not self.use_mechanism_auxiliary_heads:
                return combined, None, None
            fragment_logits = self._route_heads(
                fragment_h, munition_id,
                self.fragment_heads, self.fragment_shared_head)
            shock_logits = self._route_heads(
                shock_h, munition_id,
                self.shock_heads, self.shock_shared_head)
            fragment_logits = self._mask_and_monotonize(
                fragment_logits, munition_id)
            shock_logits = self._mask_and_monotonize(
                shock_logits, munition_id)
            return combined, fragment_logits, shock_logits

        fragment_logits = self._route_heads(
            fragment_h, munition_id,
            self.fragment_heads, self.fragment_shared_head)
        shock_logits = self._route_heads(
            shock_h, munition_id,
            self.shock_heads, self.shock_shared_head)
        fragment_logits = self._mask_and_monotonize(
            fragment_logits, munition_id)
        shock_logits = self._mask_and_monotonize(
            shock_logits, munition_id)

        # Keep fusion in fp32 under AMP.  This avoids half-precision rounding
        # to exactly one before the inverse sigmoid.
        fragment_probability = torch.sigmoid(fragment_logits.float())
        shock_probability = torch.sigmoid(shock_logits.float())
        combined_probability = (
            1.0
            - (1.0 - fragment_probability)
            * (1.0 - shock_probability)
        ).clamp(min=1e-6, max=1.0 - 1e-6)
        combined_logits = torch.log(combined_probability) - torch.log1p(
            -combined_probability)
        combined_logits = self._mask_and_monotonize(
            combined_logits, munition_id)
        return combined_logits, fragment_logits, shock_logits

    def forward_with_components(self, x, munition_id):
        """Return deployable task logits and dense component logits."""
        if not self.use_component_auxiliary_heads:
            raise RuntimeError(
                "Component auxiliary heads are not enabled.")
        h_with_e = self._encode(x, munition_id)
        combined = self._route_heads(
            h_with_e, munition_id,
            self.heads, self.shared_head)
        combined = self._apply_residual_adapters(
            combined, x, munition_id)
        combined = self._mask_and_monotonize(
            combined, munition_id)
        if self.component_branch_mode == "shared_auxiliary":
            component_logits = self.component_auxiliary_head(
                h_with_e)
        else:
            component_logits = self.independent_component_branch(
                x, munition_id)
        if self.component_tree_fusion_enabled:
            component_probability = torch.sigmoid(
                component_logits.float())
            combined_component_probability = (
                1.0
                - (1.0 - component_probability[:, 0])
                * (1.0 - component_probability[:, 1])
            )
            tree_probability = component_probabilities_to_ordinal(
                combined_component_probability)
            direct_probability = torch.sigmoid(
                combined.float())
            fusion_alpha = self.component_tree_fusion_alpha[
                munition_id]
            fused_probability = (
                (1.0 - fusion_alpha) * direct_probability
                + fusion_alpha * tree_probability
            ).clamp(1e-6, 1.0 - 1e-6)
            fused_probability = torch.cat((
                fused_probability[..., 0:1],
                torch.minimum(
                    fused_probability[..., 1:2],
                    fused_probability[..., 0:1]),
            ), dim=-1)
            fused_logits = (
                torch.log(fused_probability)
                - torch.log1p(-fused_probability)
            )
            # A zero fusion coefficient is an exact bypass, not merely the
            # mathematically equivalent sigmoid/logit round trip.  This keeps
            # frozen strong heads bitwise identical to the sealed A31 model.
            combined = torch.where(
                fusion_alpha > 0.0,
                fused_logits,
                combined.float(),
            )
            combined = self._mask_and_monotonize(
                combined, munition_id)
        return combined, component_logits

    def forward(self, x, munition_id):
        """Predict combined ordinal logits from deployable terminal inputs."""
        if (
            self.use_component_auxiliary_heads
            and self.component_tree_fusion_enabled
        ):
            combined_logits, _ = self.forward_with_components(
                x, munition_id)
            return combined_logits
        combined_logits, _, _ = self.forward_with_mechanisms(
            x, munition_id)
        return combined_logits

    def initialize_independent_component_priors(
            self, component_means_by_munition: torch.Tensor) -> None:
        if self.independent_component_branch is None:
            raise RuntimeError(
                "Independent component branch is not enabled.")
        self.independent_component_branch.initialize_output_priors(
            component_means_by_munition)

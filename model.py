import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops

_MASK_VALUE = -1e9
_EPSILON = 1e-8


def scatter_softmax(scores, index, num_nodes):
    """
    Manual scatter softmax implementation (no external torch_scatter needed).
    scores: [B, E]   raw attention scores per edge
    index:  [B, E]   target node index per edge
    num_nodes: int   maximum number of nodes
    Returns: [B, E] softmax-normalized attention weights per edge
    """
    B = scores.shape[0]
    device = scores.device
    # Numerical stability: subtract max per target group
    max_vals = torch.full((B, num_nodes), fill_value=_MASK_VALUE, device=device)
    max_vals.scatter_reduce_(1, index, scores, reduce='amax', include_self=True)
    max_gathered = torch.gather(max_vals, 1, index)
    exp_scores = torch.exp(scores - max_gathered)
    # Sum exp per target group
    sum_exp = torch.zeros(B, num_nodes, device=device)
    sum_exp.scatter_add_(1, index, exp_scores)
    sum_gathered = torch.gather(sum_exp, 1, index)
    return exp_scores / (sum_gathered + _EPSILON)


class EdgeDenoiser(nn.Module):
    """Edge-level denoising gate (Improvement 3)."""
    def __init__(self, d_model):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, h_r, r_query_expand, conf_embeds, edge_mask):
        # h_r: [B, Max_E, d_model], r_query_expand: [B, Max_E, d_model]
        # conf_embeds: [B, Max_E, d_model], edge_mask: [B, Max_E]
        gate_input = torch.cat([h_r, r_query_expand, conf_embeds], dim=-1)
        relevance = self.gate_net(gate_input)  # [B, Max_E, 1]
        relevance = relevance * edge_mask.unsqueeze(-1).float()
        return relevance  # [B, Max_E, 1]


class CrossStreamAttention(nn.Module):
    """Cross-stream attention between LRE and SFE streams (Improvement 4)."""
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query_seq, context_seq):
        # query_seq: [B, N_q, d_model], context_seq: [B, N_c, d_model]
        attn_out, _ = self.cross_attn(query=query_seq, key=context_seq, value=context_seq)
        return self.norm(query_seq + attn_out)


class ConfidenceEncoder(nn.Module):
    def __init__(self, d_model, scale=10.0):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('B', torch.randn(1, d_model // 2) * scale)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, scores, mask=None):
        if scores.dim() == 2:
            scores = scores.unsqueeze(-1) 
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(~mask, 0.0)
        x_proj = 2 * math.pi * scores @ self.B 
        x_features = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return self.projection(x_features)

class GlobalLinearAttention(nn.Module):
    """
    流内全局线性注意力 (The "Former" module)
    时间复杂度 O(N*d)，赋予局部多跳特征全局的子图视野
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        split = lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads) 
        merge = lambda t: einops.rearrange(t, 'b h n d -> b n (h d)')
        normalize = lambda t: F.normalize(t, dim=-1)
        
        q, k, v = map(split, [q, k, v])
        q, k = map(normalize, [q, k])

        kvs = einops.einsum(k, v, 'b h n d, b h n D -> b h d D') 
        numerator = einops.einsum(q, kvs, 'b h n d, b h d D -> b h n D') 
        numerator = numerator + einops.reduce(v, 'b h n d -> b h 1 d', 'sum') + v * N
        
        denominator = einops.einsum(q, einops.reduce(k, 'b h n d -> b h d', 'sum'), 'b h n d, b h d -> b h n')
        denominator = denominator + torch.full(denominator.shape, fill_value=N, device=x.device) + N
        denominator = einops.rearrange(denominator, 'b h n -> b h n 1')

        out = numerator / denominator
        out = merge(out)
        
        return self.norm(x + out)

class LogicReasoningEncoder(nn.Module):
    def __init__(self, n_rels, d_model, n_layers=3, tau=0.1,
                 disable_rel_attn=False, disable_denoise=False):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.tau = tau
        self.disable_rel_attn = disable_rel_attn
        self.disable_denoise = disable_denoise

        self.beta_net = nn.Linear(d_model, 1)

        self.msg_layers = nn.ModuleList([
            nn.Linear(d_model * 5, d_model) for _ in range(n_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

        # Improvement 1: relation-aware attention per layer
        # att input: [raw_msg || h_r || r_query] -> d_model*3 -> 1
        self.att_layers = nn.ModuleList([
            nn.Linear(d_model * 3, 1) for _ in range(n_layers)
        ])

        # Improvement 3: edge denoiser
        self.edge_denoiser = EdgeDenoiser(d_model)

    def forward(self, batch_data, r_query_embed, rel_embed_layer, conf_embeds,
                disable_rel_attn=None, disable_denoise=None):
        # Allow per-call override for forward compatibility
        if disable_rel_attn is None:
            disable_rel_attn = self.disable_rel_attn
        if disable_denoise is None:
            disable_denoise = self.disable_denoise

        edge_index = batch_data['edge_index']
        edge_type = batch_data['rels']
        edge_scores_raw = batch_data['scores'].unsqueeze(-1)
        edge_conf_mask = batch_data['edge_conf_mask']
        edge_mask = batch_data['edge_mask'].unsqueeze(-1).float()

        B, Max_N = batch_data['mask'].shape
        _, _, Max_E = edge_index.shape
        device = r_query_embed.device

        h_init = torch.zeros(B, Max_N, self.d_model).to(device)
        h_init[:, 0, :] = 1.0
        h = h_init.clone()

        # Pre-compute edge relation embeddings (static across layers)
        h_r_static = rel_embed_layer(edge_type)                              # [B, Max_E, d_model]
        r_query_expand = r_query_embed.unsqueeze(1).expand(-1, Max_E, -1)   # [B, Max_E, d_model]

        # Improvement 3: compute denoise mask once (static info)
        if not disable_denoise:
            edge_mask_1d = batch_data['edge_mask']  # [B, Max_E]
            denoise_mask = self.edge_denoiser(h_r_static, r_query_expand, conf_embeds, edge_mask_1d)
        else:
            denoise_mask = None

        target_indices_1d = edge_index[:, 1, :]  # [B, Max_E]

        H_ctx_list = []

        for k in range(self.n_layers):
            src_indices = edge_index[:, 0, :].unsqueeze(-1).expand(-1, -1, self.d_model)
            h_src = torch.gather(h, 1, src_indices)
            h_init_src = torch.gather(h_init, 1, src_indices)

            beta = torch.sigmoid(self.beta_net(h_r_static + r_query_expand))
            gate_known = torch.sigmoid((edge_scores_raw - beta) / self.tau)
            gate = torch.where(edge_conf_mask.unsqueeze(-1), gate_known,
                               torch.full_like(gate_known, 0.5))

            comp_feat = h_src * h_r_static
            msg_in = torch.cat([comp_feat, h_src, h_init_src, h_r_static, conf_embeds], dim=-1)
            raw_msg = F.relu(self.msg_layers[k](msg_in))

            if not disable_rel_attn:
                # Improvement 1: relation-aware attention aggregation
                att_input = torch.cat([raw_msg, h_r_static, r_query_expand], dim=-1)  # [B, Max_E, d_model*3]
                att_score = F.leaky_relu(self.att_layers[k](att_input)).squeeze(-1)    # [B, Max_E]
                # Mask out padding edges before softmax
                att_score = att_score.masked_fill(batch_data['edge_mask'] == 0, _MASK_VALUE)
                alpha = scatter_softmax(att_score, target_indices_1d, Max_N)  # [B, Max_E]
                alpha = alpha.unsqueeze(-1)  # [B, Max_E, 1]

                # Combine gate, alpha, denoising and raw message
                if denoise_mask is not None:
                    weighted_msg = gate * alpha * raw_msg * edge_mask * denoise_mask
                else:
                    weighted_msg = gate * alpha * raw_msg * edge_mask
            else:
                # Fallback to original scatter_add (no attention)
                if denoise_mask is not None:
                    weighted_msg = gate * raw_msg * edge_mask * denoise_mask
                else:
                    weighted_msg = gate * raw_msg * edge_mask

            target_indices = target_indices_1d.unsqueeze(-1).expand(-1, -1, self.d_model)
            aggr_out = torch.zeros_like(h)
            aggr_out.scatter_add_(1, target_indices, weighted_msg)

            update = self.update_layers[k](aggr_out)
            h = h + update
            h = self.layer_norm(h)

            H_ctx_list.append(h[:, 0, :])

        return torch.stack(H_ctx_list, dim=1)

class StructureFeatureEncoder(nn.Module):
    def __init__(self, n_rels, d_model, M=20, n_layers=3,
                 disable_rel_attn=False, disable_denoise=False):
        super().__init__()
        self.d_model = d_model
        self.M = M
        self.n_layers = n_layers
        self.disable_rel_attn = disable_rel_attn
        self.disable_denoise = disable_denoise

        self.dist_embed = nn.Embedding(10, d_model)

        self.msg_layers = nn.ModuleList([
            nn.Linear(d_model * 5, d_model) for _ in range(n_layers)
        ])
        self.update_layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.att_net = nn.Linear(d_model * 2, 1)

        # Improvement 1: relation-aware attention per layer
        self.att_layers = nn.ModuleList([
            nn.Linear(d_model * 3, 1) for _ in range(n_layers)
        ])

        # Improvement 2: JK-Net layer fusion
        self.jk_att = nn.Linear(d_model, 1)

        # Improvement 3: edge denoiser
        self.edge_denoiser = EdgeDenoiser(d_model)

    def forward(self, batch_data, r_query_embed, rel_embed_layer, conf_embeds,
                disable_rel_attn=None, disable_denoise=None):
        if disable_rel_attn is None:
            disable_rel_attn = self.disable_rel_attn
        if disable_denoise is None:
            disable_denoise = self.disable_denoise

        dists = batch_data['dists']
        edge_index = batch_data['edge_index']
        edge_type = batch_data['rels']
        node_mask = batch_data['mask']
        edge_mask = batch_data['edge_mask'].unsqueeze(-1).float()

        B, Max_N = dists.shape
        _, _, Max_E = edge_index.shape
        device = dists.device

        dist_emb = self.dist_embed(torch.clamp(dists, 0, 9))
        noise = torch.randn_like(dist_emb) * 0.1
        h = dist_emb + noise

        # Pre-compute static edge embeddings
        h_r_static = rel_embed_layer(edge_type)                              # [B, Max_E, d_model]
        r_query_expand = r_query_embed.unsqueeze(1).expand(-1, Max_E, -1)   # [B, Max_E, d_model]

        # Improvement 3: compute denoise mask once
        if not disable_denoise:
            edge_mask_1d = batch_data['edge_mask']
            denoise_mask = self.edge_denoiser(h_r_static, r_query_expand, conf_embeds, edge_mask_1d)
        else:
            denoise_mask = None

        target_indices_1d = edge_index[:, 1, :]  # [B, Max_E]

        # Improvement 2: collect layer outputs for JK-Net
        layer_outputs = []

        for k in range(self.n_layers):
            src_indices = edge_index[:, 0, :].unsqueeze(-1).expand(-1, -1, self.d_model)
            h_src = torch.gather(h, 1, src_indices)

            dist_src = torch.gather(dist_emb, 1, src_indices)

            comp_feat = h_src * h_r_static
            msg_input = torch.cat([comp_feat, h_src, dist_src, h_r_static, conf_embeds], dim=-1)
            msg = F.relu(self.msg_layers[k](msg_input))

            if not disable_rel_attn:
                # Improvement 1: relation-aware attention aggregation
                att_input = torch.cat([msg, h_r_static, r_query_expand], dim=-1)  # [B, Max_E, d_model*3]
                att_score = F.leaky_relu(self.att_layers[k](att_input)).squeeze(-1)  # [B, Max_E]
                att_score = att_score.masked_fill(batch_data['edge_mask'] == 0, _MASK_VALUE)
                alpha = scatter_softmax(att_score, target_indices_1d, Max_N)  # [B, Max_E]
                alpha = alpha.unsqueeze(-1)  # [B, Max_E, 1]

                if denoise_mask is not None:
                    msg = alpha * msg * edge_mask * denoise_mask
                else:
                    msg = alpha * msg * edge_mask
            else:
                if denoise_mask is not None:
                    msg = msg * edge_mask * denoise_mask
                else:
                    msg = msg * edge_mask

            target_indices = target_indices_1d.unsqueeze(-1).expand(-1, -1, self.d_model)
            aggr_out = torch.zeros_like(h)
            aggr_out.scatter_add_(1, target_indices, msg)

            h = self.update_layers[k](aggr_out) + h

            # Improvement 2: save layer output for JK-Net
            layer_outputs.append(h)

        # Improvement 2: JK-Net adaptive fusion across layers
        # layer_outputs: list of n_layers x [B, Max_N, d_model]
        layer_stack = torch.stack(layer_outputs, dim=1)  # [B, n_layers, Max_N, d_model]
        jk_scores = self.jk_att(layer_stack).squeeze(-1)  # [B, n_layers, Max_N]
        jk_weights = F.softmax(jk_scores, dim=1)          # [B, n_layers, Max_N]
        h = (jk_weights.unsqueeze(-1) * layer_stack).sum(dim=1)  # [B, Max_N, d_model]

        h = h * node_mask.unsqueeze(-1).float()

        t_state = h[:, 0, :]

        r_q_exp = r_query_embed.unsqueeze(1).expand(-1, Max_N, -1)
        att_input = torch.cat([h, r_q_exp], dim=-1)
        att_scores = F.leaky_relu(self.att_net(att_input)).squeeze(-1)

        att_scores = att_scores.masked_fill(~node_mask, _MASK_VALUE)
        alpha = F.softmax(att_scores, dim=1)

        curr_M = min(self.M, Max_N)
        topk_vals, topk_idx = torch.topk(alpha, k=curr_M, dim=1)

        topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
        H_evd = torch.gather(h, 1, topk_idx_exp)

        H_evd = H_evd * topk_vals.unsqueeze(-1)

        if curr_M < self.M:
            pad_size = self.M - curr_M
            padding = torch.zeros(B, pad_size, self.d_model).to(device)
            H_evd = torch.cat([H_evd, padding], dim=1)

        return H_evd, t_state


class KGReasoningModel(nn.Module):
    def __init__(self, n_ents, n_rels, d_model=64, n_layers=3, top_k_evd=20,
                 disable_sfe=False, disable_lre=False, disable_conf=False, disable_bico=False,
                 disable_former=False, disable_rel_attn=False, disable_denoise=False,
                 disable_cross=False, conf_mask_prob=0.9):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # 记录消融参数
        self.disable_sfe = disable_sfe
        self.disable_lre = disable_lre
        self.disable_conf = disable_conf
        self.disable_former = disable_former
        self.disable_cross = disable_cross
        self.conf_mask_prob = conf_mask_prob

        self.rel_embed = nn.Embedding(n_rels, d_model)

        # 1. 基础特征编码器
        self.conf_encoder = ConfidenceEncoder(d_model)
        self.lre = LogicReasoningEncoder(n_rels, d_model, n_layers=n_layers,
                                         disable_rel_attn=disable_rel_attn,
                                         disable_denoise=disable_denoise)
        self.sfe = StructureFeatureEncoder(n_rels, d_model, M=top_k_evd, n_layers=n_layers,
                                           disable_rel_attn=disable_rel_attn,
                                           disable_denoise=disable_denoise)

        # 2. 引入 DDS-Former 的灵魂：流内全局线性注意力
        self.lre_former = GlobalLinearAttention(d_model)
        self.sfe_former = GlobalLinearAttention(d_model)

        # 3. 独立特征池化层
        # Improvement 2: replace w_ctx (n_layers*d_model -> d_model) with ctx_layer_att
        self.ctx_layer_att = nn.Linear(d_model, 1)
        self.w_evd = nn.Linear(d_model * 2, d_model)

        # Improvement 4: dual-stream cross attention
        self.cross_ctx2evd = CrossStreamAttention(d_model)
        self.cross_evd2ctx = CrossStreamAttention(d_model)

        # 4. 事实置信度预测分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 6 + 1, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, 1)
        )

    def forward(self, h_idx, r_idx, t_idx, lre_data, sfe_data):
        r_query = self.rel_embed(r_idx)
        B = r_query.shape[0]
        device = r_query.device

        # ==========================================
        # 第一阶段：事实置信度注入 (已有消融)
        # ==========================================
        if self.disable_conf:
            lre_conf_emb = torch.zeros(B, lre_data['scores'].shape[1], self.d_model, device=device)
            sfe_conf_emb = torch.zeros(B, sfe_data['scores'].shape[1], self.d_model, device=device)
        else:
            lre_mask = lre_data['edge_conf_mask']
            sfe_mask = sfe_data['edge_conf_mask']
            if self.training and self.conf_mask_prob < 1.0:
                lre_rand = torch.rand(lre_data['scores'].shape + (1,), device=device)
                sfe_rand = torch.rand(sfe_data['scores'].shape + (1,), device=device)
                lre_mask = lre_mask.unsqueeze(-1) & (lre_rand < self.conf_mask_prob)
                sfe_mask = sfe_mask.unsqueeze(-1) & (sfe_rand < self.conf_mask_prob)
            lre_conf_emb = self.conf_encoder(lre_data['scores'], mask=lre_mask)
            sfe_conf_emb = self.conf_encoder(sfe_data['scores'], mask=sfe_mask)

        # ==========================================
        # 第二阶段：解耦双流特征提取 (Local + Global)
        # ==========================================
        if self.disable_lre:
            H_ctx_global = torch.zeros(B, self.n_layers, self.d_model, device=device)
        else:
            H_ctx_local = self.lre(lre_data, r_query, self.rel_embed, lre_conf_emb)
            H_ctx_global = H_ctx_local if self.disable_former else self.lre_former(H_ctx_local)

        if self.disable_sfe:
            H_evd_global = torch.zeros(B, self.sfe.M, self.d_model, device=device)
            t_state = torch.zeros(B, self.d_model, device=device)
        else:
            H_evd_local, t_state = self.sfe(sfe_data, r_query, self.rel_embed, sfe_conf_emb)
            H_evd_global = H_evd_local if self.disable_former else self.sfe_former(H_evd_local)

        # ==========================================
        # Improvement 4: 双流交叉注意力（Former 之后，Pooling 之前）
        # ==========================================
        if not self.disable_cross and not self.disable_lre and not self.disable_sfe:
            H_ctx_cross = self.cross_ctx2evd(H_ctx_global, H_evd_global)
            H_evd_cross = self.cross_evd2ctx(H_evd_global, H_ctx_global)
            H_ctx_global = H_ctx_cross
            H_evd_global = H_evd_cross

        # ==========================================
        # 第三阶段：特征池化 (Pooling)
        # ==========================================
        D = self.d_model

        if self.disable_lre:
            z_ctx = torch.zeros(B, D, device=device)
        else:
            # Improvement 2: JK-Net style attention pooling over layers
            # H_ctx_global: [B, n_layers, d_model]
            layer_scores = self.ctx_layer_att(H_ctx_global).squeeze(-1)  # [B, n_layers]
            layer_weights = F.softmax(layer_scores, dim=1)                # [B, n_layers]
            z_ctx = (layer_weights.unsqueeze(-1) * H_ctx_global).sum(dim=1)  # [B, d_model]
            z_ctx = F.relu(z_ctx)

        if self.disable_sfe:
            z_evd = torch.zeros(B, D, device=device)
        else:
            evd_mean = torch.mean(H_evd_global, dim=1)
            evd_max, _ = torch.max(H_evd_global, dim=1)
            z_evd = F.relu(self.w_evd(torch.cat([evd_mean, evd_max], dim=-1)))

        # ==========================================
        # 第四阶段：极简预测 (Late Fusion)
        # ==========================================
        ctx_t_cross = z_ctx * t_state
        evd_t_cross = z_evd * t_state
        bilinear_score = torch.sum(z_ctx * r_query * t_state, dim=-1, keepdim=True)

        feature = torch.cat([
            z_ctx, z_evd, r_query, t_state,
            ctx_t_cross, evd_t_cross,
            bilinear_score
        ], dim=-1)

        out = torch.sigmoid(self.classifier(feature))
        out = out.squeeze(-1) * 1.1 - 0.05
        out = torch.clamp(out, min=1e-5, max=1.0 - 1e-5)

        return out

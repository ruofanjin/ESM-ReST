import torch
import torch.nn as nn
import torch.nn.functional as F

class MutationContextModule(nn.Module):
    # ... (MutationContextModule code same as before, omitted here to save space) ...
    # ... (Please copy the complete MutationContextModule from your original code here) ...
    """
    Mutation interaction modeling module
    Captures contextual interaction effects between mutation sites through multi-head self-attention mechanism,
    replacing simple average aggregation to improve prediction capability for multi-mutant effects.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.1, debug_attention=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.debug_attention = debug_attention
        
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, mut_embs, attention_mask=None):
        if mut_embs.shape[1] <= 1:
            return mut_embs.squeeze(1)

        key_padding_mask = ~attention_mask if attention_mask is not None else None
        
        attn_output, _ = self.self_attention(
            mut_embs, mut_embs, mut_embs,
            key_padding_mask=key_padding_mask
        )
        mut_embs_norm1 = self.layer_norm1(mut_embs + self.dropout(attn_output))
        ff_output = self.feedforward(mut_embs_norm1)
        mut_embs_final = self.layer_norm2(mut_embs_norm1 + self.dropout(ff_output))
        
        attention_scores = self.attention_aggregator(mut_embs_final).squeeze(-1)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask, float('-inf'))
        
        attention_weights_agg = F.softmax(attention_scores, dim=-1)
        aggregated = torch.sum(mut_embs_final * attention_weights_agg.unsqueeze(-1), dim=1)
        return aggregated


class ESM2Effect(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_scales = 5
        
        # --- Create modules based on configuration switches ---
        if self.config.use_cnn:
            self.conv_scales = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(config.input_dim, config.hidden_dim // 4, kernel_size=k, padding=k//2),
                    nn.GELU(), nn.Dropout(config.dropout_rate)
                ) for k in [3, 5, 7, 9, 11]
            ])
            self.conv_projection = nn.Linear(config.hidden_dim // 4 * self.num_scales, config.hidden_dim)
            self.conv_residual = nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=1)
        else:
            # If not using CNN, use a simple linear layer to unify dimensions
            self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)

        self.feature_fusion = nn.Sequential(nn.LayerNorm(config.hidden_dim))
        
        if self.config.use_attention:
            self.self_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim, num_heads=config.num_heads,
                dropout=config.dropout_rate, batch_first=True
            )
            self.layer_norm_attn = nn.LayerNorm(config.hidden_dim)

        if self.config.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim, num_heads=config.num_heads,
                dropout=config.dropout_rate, batch_first=True
            )
            self.layer_norm_cross = nn.LayerNorm(config.hidden_dim)
        
        if self.config.use_mutation_context:
            self.mutation_context_module = MutationContextModule(
                hidden_dim=config.hidden_dim, dropout_rate=config.dropout_rate
            )

        self.position_projection = nn.Linear(config.hidden_dim + 1, config.hidden_dim)

        # --- Common modules ---
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2), nn.GELU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2), nn.GELU(), nn.LayerNorm(config.hidden_dim // 2)
        )
        self.score_head = nn.Linear(config.hidden_dim // 2, 1)
        self.confidence_head = nn.Linear(config.hidden_dim // 2, 1)

    def forward(self, wt_embedding, mut_embedding, pos, pos_mask, lengths):
        batch_size, seq_len, _ = mut_embedding.shape

        if self.config.use_cnn:
            mut_t = mut_embedding.transpose(1, 2)
            wt_t = wt_embedding.transpose(1, 2)
            mut_features = [conv(mut_t).transpose(1, 2) for conv in self.conv_scales]
            wt_features = [conv(wt_t).transpose(1, 2) for conv in self.conv_scales]
            mut_combined = self.conv_projection(torch.cat(mut_features, dim=-1))
            wt_combined = self.conv_projection(torch.cat(wt_features, dim=-1))
            mut_combined += self.conv_residual(mut_t).transpose(1, 2)
            wt_combined += self.conv_residual(wt_t).transpose(1, 2)
        else:
            mut_combined = self.input_projection(mut_embedding)
            wt_combined = self.input_projection(wt_embedding)

        mut_combined = self.feature_fusion(mut_combined)
        wt_combined = self.feature_fusion(wt_combined)

        if self.config.use_attention:
            mut_attended, _ = self.self_attention(mut_combined, mut_combined, mut_combined)
            mut_combined = self.layer_norm_attn(mut_combined + mut_attended)
        
        if self.config.use_cross_attention:
            cross_attended, _ = self.cross_attention(mut_combined, wt_combined, wt_combined)
            mut_combined = self.layer_norm_cross(mut_combined + cross_attended)

        # Extract mutation site features
        position_features = []
        for i in range(batch_size):
            valid_pos = pos[i][pos_mask[i]]
            if len(valid_pos) > 0:
                pos_feat = mut_combined[i, valid_pos, :]
                pos_indices = valid_pos.unsqueeze(-1).float() / seq_len
                pos_feat = torch.cat([pos_feat, pos_indices], dim=-1)
                position_features.append(self.position_projection(pos_feat))
            else:
                position_features.append(mut_combined[i].mean(dim=0, keepdim=True))

        max_mut = max(len(feat) for feat in position_features)
        padded_features = [F.pad(feat, (0, 0, 0, max_mut - len(feat))) for feat in position_features]
        position_tensor = torch.stack(padded_features)
        
        attention_mask_mut = torch.arange(max_mut, device=mut_combined.device)[None, :] < torch.tensor([len(f) for f in position_features], device=mut_combined.device)[:, None]

        if self.config.use_mutation_context:
            global_features = self.mutation_context_module(position_tensor, attention_mask=attention_mask_mut)
        else:
            masked_output = position_tensor * attention_mask_mut.unsqueeze(-1)
            global_features = masked_output.sum(dim=1) / attention_mask_mut.sum(dim=1, keepdim=True).clamp(min=1)

        mlp_output = self.mlp(global_features)
        score_pred = self.score_head(mlp_output)
        conf_pred = self.confidence_head(mlp_output)
        
        return score_pred, conf_pred
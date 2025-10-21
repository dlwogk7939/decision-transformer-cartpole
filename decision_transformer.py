import os

# Ensure MPS falls back to CPU for unsupported ops before torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim=4, act_dim=2, hidden_size=128, n_layer=3, n_head=2, seq_len=20):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.seq_len = seq_len

        # Embedding: RTG, state, action      
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(act_dim, hidden_size)

        # Position embedding
        self.pos_emb = nn.Embedding(seq_len, hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_head,
            dim_feedforward=4 * hidden_size,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        
        # Output head
        self.predict_action = nn.Linear(hidden_size, act_dim)

        causal_mask = torch.triu(
            torch.ones((seq_len * 3, seq_len * 3), dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, rtg, states, actions, padding_mask=None):
        """
        rtg: (batch, seq_len, 1)
        states: (batch, seq_len, state_dim)
        actions: (batch, seq_len) prev actions
        padding_mask: (batch, seq_len) True for valid tokens
        """

        batch_size, seq_len = states.shape[0], states.shape[1]
        positions = torch.arange(seq_len, device=states.device).unsqueeze(0).expand(batch_size, -1)

        # Embedding: RTG, state, action
        rtg_emb = self.embed_rtg(rtg)  # (B, T, H)
        state_emb = self.embed_state(states)  # (B, T, H)
        act_emb = self.embed_action(actions)  # (B, T, H)

        # Concatenate sequence (RTG, state, action)
        # [rtg_1, s_1, a_0, rtg_2, s_2, a_1, ...]        
        x = torch.stack((rtg_emb, state_emb, act_emb), dim=2)  # (B, T, 3, H)
        x = x.reshape(batch_size, seq_len * 3, self.hidden_size)  # (B, 3T, H)

        # Position embedding
        pos_emb = self.pos_emb(positions.repeat_interleave(3, dim=1))
        x = x + pos_emb

        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = ~(padding_mask.repeat_interleave(3, dim=1))
            key_padding_mask = key_padding_mask.bool()

        attn_mask = self.causal_mask[: seq_len * 3, : seq_len * 3]
        
        h = self.transformer(
            x,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )  # (B, 3T, H)

        h_state = h[:, 1::3, :]
        act_logits = self.predict_action(h_state)  # (B, T, act_dim)

        return act_logits

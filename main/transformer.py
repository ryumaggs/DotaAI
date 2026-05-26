import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=25, dropout=0.1):  # 24 tokens + CLS
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # sinusoidal encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        # attention projections (linear, no activation) → xavier
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

        # ffn[0]: linear → relu → kaiming
        nn.init.kaiming_uniform_(self.ffn[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.ffn[0].bias)

        # ffn[2]: linear → nothing → xavier
        nn.init.xavier_uniform_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

        # layernorm → identity at init
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x


class DraftTransformer(nn.Module):
    def __init__(self, num_heroes=256, embed_dim=64, num_heads=4,
                 ff_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        # +1 for a PAD token (id=0) to handle missing entries gracefully
        self.hero_embed  = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)
        # learned token type embedding: 0=radiant pick, 1=radiant ban, 2=dire pick, 3=dire ban
        self.pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, 1)   # no sigmoid
        
    def forward(self, hero_ids):
        """
        hero_ids: (batch, 24) — hero IDs in draft order
        """
        x = self.hero_embed(hero_ids)  # (batch, 24, embed_dim)
        x = self.pos_encoder(x)                                     # add positional encoding
        
        # prepend CLS
        cls = self.cls_token.expand(x.size(0), -1, -1)             # (batch, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)                             # (batch, 25, embed_dim)

        for block in self.blocks:
            x = block(x)

        cls_out = x[:, 0, :]                                        # (batch, embed_dim)
        return self.head(cls_out).squeeze(-1)                       # (batch,)
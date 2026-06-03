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
        mask = torch.tril(torch.ones(24,24))
        self.register_buffer('tril',mask.masked_fill(mask==0, float('-inf')))

        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
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
        seq_len = x.size(1)
        attn_out, _ = self.attn(x, x, x, attn_mask=self.tril[:seq_len, :seq_len])
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x

class DraftTransformer(nn.Module):
    def __init__(self, num_heroes=256, embed_dim=64, num_heads=4,
                 ff_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        # +1 for a PAD token (id=0) to handle missing entries gracefully
        self.pos_embed  = nn.Embedding(24, embed_dim)  # 24 = draft sequence length
        self.hero_embed  = nn.Embedding(num_heroes + 1, embed_dim, padding_idx=0)
        # learned token type embedding: 0=radiant pick, 1=radiant ban, 2=dire pick, 3=dire ban
        self.blocks      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(embed_dim, num_heroes)   # no sigmoid
    
    def process_data(self, all_data_info, device=torch.device('cpu')):
        return all_data_info[0].to(device)

    def forward(self, hero_ids):
        """
        hero_ids: (batch, 24) — hero IDs in draft order
        """
        
        # in forward:
        positions = torch.arange(hero_ids.size(1), device=hero_ids.device)  # (24,)
        x = self.hero_embed(hero_ids) + self.pos_embed(positions)      

        for block in self.blocks:
            x = block(x)

        return self.head(x)           
    
def build_availability_mask_all_positions(seq, vocab_size,):
    """
    a core compoenent of both draft and objective transformers.

    it builds a mask such that any vocabulary that already exists in the sequence
    cannot exist again in the sequence

    
    seq: (B, T) — input token ids
    Returns: (B, T, V) BoolTensor, True = available at that position
    """
    B, T = seq.shape
    device = seq.device

    one_hot = torch.zeros(B, T, vocab_size, dtype=torch.long, device=device)
    one_hot.scatter_(2, seq.unsqueeze(2), 1)

    cumsum = one_hot.cumsum(dim=1)

    shifted = torch.cat([
        torch.zeros(B, 1, vocab_size, dtype=torch.long, device=device),
        cumsum[:, :-1, :]
    ], dim=1)

    return shifted == 0  # (B, T, V)
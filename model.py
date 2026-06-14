"""Modelo de linguagem modular com "sequence mixers" plugáveis.

A ideia central do laboratório: o Transformer é (embedding -> N blocos -> head),
e cada bloco é (mixer de sequência + MLP). A pesquisa de arquiteturas eficientes
é, em grande parte, a busca por um mixer melhor que a self-attention.

Aqui o mixer é uma classe plugável. Para testar uma ideia nova, basta
implementar um novo mixer e registrá-lo em MIXERS — todo o resto
(treino, benchmark, medição) é reaproveitado.

Mixers incluídos:
  - attention: self-attention causal completa, O(n^2). O baseline a ser batido.
  - linear:    atenção linear (Katharopoulos et al., 2020), O(n) com estado
               comprimido de tamanho fixo. Espere recall fraco — esse é
               exatamente o problema em aberto que queremos resolver.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 2
    max_seq_len: int = 512
    mixer: str = "attention"


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class CausalSelfAttention(nn.Module):
    """Atenção completa. Custo O(T^2) em tempo e memória de ativação.

    Na inferência exige KV cache que cresce linearmente com o contexto —
    é o custo de hardware que queremos eliminar sem perder recall.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class LinearAttention(nn.Module):
    """Atenção linear com feature map elu+1.

    Equivale a uma RNN cujo estado é a matriz S_t = sum_i phi(k_i) v_i^T
    (tamanho fixo: head_dim x head_dim, independe de T). Na inferência o
    custo por token é O(1) e não há KV cache — eficiência máxima.

    O preço: toda a história do contexto é espremida numa matriz pequena,
    então a recuperação exata (recall) degrada conforme o número de
    associações cresce. Rode o benchmark e veja com seus próprios olhos.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Forma paralela (treino): cumsum dos produtos externos k_t v_t^T.
        # Nota: materializa (B, H, T, D, D) — didático, não otimizado.
        kv = torch.einsum("bhtd,bhte->bhtde", k, v)
        S = kv.cumsum(dim=2)                      # estado acumulado até t
        z = k.cumsum(dim=2)                       # normalizador acumulado
        num = torch.einsum("bhtd,bhtde->bhte", q, S)
        den = torch.einsum("bhtd,bhtd->bht", q, z).unsqueeze(-1).clamp(min=1e-6)
        y = num / den

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


MIXERS = {
    "attention": CausalSelfAttention,
    "linear": LinearAttention,
}


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.mixer = MIXERS[cfg.mixer](cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SequenceModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Nota: sem weight tying — em tarefas sinteticas de recall, amarrar
        # embedding e head cria conflito de gradiente e dificulta o binding k->v.
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

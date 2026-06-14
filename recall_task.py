"""Tarefa de recall associativo multi-query (MQAR, estilo Zoology/HazyResearch).

Por que essa tarefa? Porque ela isola a capacidade que separa o Transformer
das arquiteturas eficientes: recuperar informacao EXATA do contexto.

Formato de cada sequencia (P pares):
    k1 v1 k2 v2 ... kP vP  SEP  q1 a1 q2 a2 ... qP aP

onde (q1..qP) e uma permutacao das chaves e a_i e o valor associado a q_i.
A loss e aplicada APENAS nas posicoes das queries (prever o valor seguinte).
Multiplas queries por sequencia = sinal de aprendizado denso, convergencia
muito mais rapida que a variante de query unica.

Um Transformer resolve isso com ~100% de acuracia (a attention literalmente
"aponta" para a posicao da chave). Modelos de estado comprimido degradam
conforme P cresce alem da capacidade do estado.

Vocabulario: token 0 = SEP, tokens [1, n_keys] = chaves,
tokens [n_keys+1, n_keys+n_vals] = valores.
"""

import torch

IGNORE = -100


def vocab_size(n_keys: int, n_vals: int) -> int:
    return 1 + n_keys + n_vals


def seq_len(n_pairs: int) -> int:
    return 4 * n_pairs + 1  # pares + SEP + (query, resposta) x P


def make_batch(batch_size: int, n_pairs: int, n_keys: int, n_vals: int,
               device: torch.device):
    """Retorna (inputs, targets), ambos (B, T). targets = IGNORE fora das queries."""
    assert n_pairs <= n_keys, "chaves devem ser distintas dentro da sequencia"
    B, P = batch_size, n_pairs

    # Chaves distintas por amostra: argsort de ruido = permutacao aleatoria.
    keys = torch.rand(B, n_keys).argsort(dim=-1)[:, :P] + 1            # (B, P)
    vals = torch.randint(0, n_vals, (B, P)) + 1 + n_keys               # (B, P)

    # Ordem aleatoria das queries (permutacao dos P pares, por amostra).
    perm = torch.rand(B, P).argsort(dim=-1)                            # (B, P)
    q_keys = keys.gather(1, perm)
    q_vals = vals.gather(1, perm)

    T = seq_len(P)
    seq = torch.zeros(B, T, dtype=torch.long)
    seq[:, 0:2 * P:2] = keys
    seq[:, 1:2 * P:2] = vals
    seq[:, 2 * P] = 0                          # SEP
    seq[:, 2 * P + 1::2] = q_keys              # queries
    seq[:, 2 * P + 2::2] = q_vals              # respostas (teacher forcing)

    targets = torch.full((B, T), IGNORE, dtype=torch.long)
    targets[:, 2 * P + 1::2] = q_vals          # prever o valor na posicao da query

    return seq.to(device), targets.to(device)

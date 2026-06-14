"""Treina um modelo na tarefa de recall associativo (MQAR) e mede:
  - acuracia de recall (qualidade de memorizacao)
  - tokens/segundo no treino (custo computacional)
  - parametros do modelo

Uso:
    python train.py --mixer attention --n-pairs 16
    python train.py --mixer linear    --n-pairs 16

Compare os dois e aumente --n-pairs para ver o estado comprimido saturar.
"""

import argparse
import time

import torch
import torch.nn.functional as F

import recall_task
from model import ModelConfig, SequenceModel


def run_eval(model, args, device, n_batches: int = 10) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = recall_task.make_batch(args.batch_size, args.n_pairs,
                                          args.n_keys, args.n_vals, device)
            pred = model(x).argmax(-1)
            mask = y != recall_task.IGNORE
            correct += (pred[mask] == y[mask]).sum().item()
            total += mask.sum().item()
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mixer", choices=["attention", "linear"], default="attention")
    p.add_argument("--n-pairs", type=int, default=16,
                   help="pares chave-valor por sequencia (dificuldade do recall)")
    p.add_argument("--n-keys", type=int, default=64)
    p.add_argument("--n-vals", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = recall_task.seq_len(args.n_pairs)
    cfg = ModelConfig(
        vocab_size=recall_task.vocab_size(args.n_keys, args.n_vals),
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        max_seq_len=T, mixer=args.mixer,
    )
    model = SequenceModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"mixer={args.mixer}  params={model.num_params():,}  "
          f"seq_len={T}  n_pairs={args.n_pairs}  device={device.type}")

    tokens_seen = 0
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = recall_task.make_batch(args.batch_size, args.n_pairs,
                                      args.n_keys, args.n_vals, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                               ignore_index=recall_task.IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tokens_seen += x.numel()

        if step % max(1, args.steps // 8) == 0:
            acc = run_eval(model, args, device, n_batches=3)
            print(f"step {step:5d}  loss {loss.item():.4f}  recall_acc {acc:.3f}")

    elapsed = time.perf_counter() - t0
    final_acc = run_eval(model, args, device)
    print("-" * 60)
    print(f"RESULTADO  mixer={args.mixer}  n_pairs={args.n_pairs}")
    print(f"  recall final : {final_acc:.3f}")
    print(f"  throughput   : {tokens_seen / elapsed:,.0f} tokens/s (treino)")
    print(f"  parametros   : {model.num_params():,}")


if __name__ == "__main__":
    main()

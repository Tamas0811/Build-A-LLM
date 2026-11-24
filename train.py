"""Train a tiny, randomly initialised causal transformer on fineweb_corpus.txt for ~60 s on CPU.

Uses GPT-4o's o200k_base tokenizer. Prints `step | loss | elapsed | tokens/s` live, samples text
from the same beginning and seed before training, every N updates and at the end, and saves the
log, loss plot, samples and checkpoint to the outputs/ folder.
"""
import argparse
import codecs
import csv
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save the plot to a file; no window needed
import matplotlib.pyplot as plt
import tiktoken
import torch

from model import TinyGPT, generate_stream

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, default=HERE / "fineweb_corpus.txt")
    p.add_argument("--tokens", type=Path, default=HERE / "fineweb_tokens.txt")
    p.add_argument("--out", type=Path, default=HERE / "outputs")
    p.add_argument("--seconds", type=float, default=60.0, help="training-time budget (sampling time not counted)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64, help="context length in tokens")
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=1337, help="seed for weight initialisation and batches")
    p.add_argument("--sample-seed", type=int, default=42, help="the same seed is reused for every sample")
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--sample-tokens", type=int, default=40)
    p.add_argument("--prompt", default="If you have set yourself", help="short sentence beginning for samples")
    return p.parse_args()


def load_and_validate(enc, corpus_path, tokens_path, out_dir, log):
    """Return (corpus, token_ids). Uses the file's IDs if they match the corpus, else regenerates."""
    corpus = corpus_path.read_text(encoding="utf-8-sig")
    expected = enc.encode(corpus, disallowed_special=())
    log(f"Corpus: {len(corpus):,} characters -> {len(expected):,} tokens with o200k_base")
    log(f"Round trip decode(encode(corpus)) == corpus: {enc.decode(expected) == corpus}")

    file_ids = None
    if tokens_path.exists():
        file_ids = [int(x) for x in re.findall(r"\d+", tokens_path.read_text(encoding="utf-8-sig"))]
        log(f"{tokens_path.name}: {len(file_ids):,} token IDs")
    else:
        log(f"WARNING: {tokens_path.name} not found.")

    if file_ids == expected:
        log("Validation OK: the token IDs in the file match the corpus exactly.")
        return corpus, file_ids

    if file_ids is not None:
        first = next((i for i, (a, b) in enumerate(zip(file_ids, expected)) if a != b),
                     min(len(file_ids), len(expected)))
        log(f"Validation FAILED: file has {len(file_ids):,} IDs, corpus gives {len(expected):,}; "
            f"first difference at position {first}.")
    regen_path = out_dir / "fineweb_tokens_regenerated.txt"
    regen_path.write_text(", ".join(map(str, expected)), encoding="utf-8")
    log(f"Using token IDs regenerated from the corpus (saved to {regen_path.name}).")
    return corpus, expected


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args.out.mkdir(parents=True, exist_ok=True)

    log_txt = open(args.out / "training_log.txt", "w", encoding="utf-8")

    def log(line=""):
        print(line, flush=True)
        log_txt.write(line + "\n")
        log_txt.flush()

    enc = tiktoken.get_encoding("o200k_base")
    _, ids = load_and_validate(enc, args.corpus, args.tokens, args.out, log)

    # Compact vocabulary: only the token IDs that occur in the corpus, renumbered 0..V-1.
    vocab = sorted(set(ids))                      # internal ID -> original o200k_base ID
    to_internal = {tok: i for i, tok in enumerate(vocab)}
    data = torch.tensor([to_internal[t] for t in ids], dtype=torch.long)
    V = len(vocab)
    log(f"Compact vocabulary: {V:,} distinct tokens (out of {enc.n_vocab:,} in o200k_base)")
    log(f"Loss of a uniform random guess: ln({V}) = {math.log(V):.4f}")

    # Sentence beginning for samples; every token must exist in the compact vocabulary.
    prompt_ids = enc.encode(args.prompt, disallowed_special=())
    if any(t not in to_internal for t in prompt_ids):
        prompt_ids = ids[:6]
        log(f"Prompt {args.prompt!r} contains tokens not in the corpus; using the corpus opening instead.")
    prompt_text = enc.decode(prompt_ids)
    prompt_internal = [to_internal[t] for t in prompt_ids]
    log(f"Sample beginning: {prompt_text!r} | sample seed: {args.sample_seed}")

    torch.manual_seed(args.seed)
    block = min(args.block_size, len(data) - 2)
    model = TinyGPT(V, block_size=block, n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {args.n_layer} layers, {args.n_head} heads, {args.n_embd}-dim, context {block}, {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    samples_file = open(args.out / "samples.txt", "w", encoding="utf-8")

    def sample(label):
        """Generate from the same beginning and seed, streaming tokens as they are produced."""
        model.eval()
        generator = torch.Generator().manual_seed(args.sample_seed)
        idx = torch.tensor([prompt_internal], dtype=torch.long)
        utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
        print(f"\n--- sample: {label} ---", flush=True)
        print(prompt_text, end="", flush=True)
        pieces = []
        for internal_id in generate_stream(model, idx, args.sample_tokens, generator):
            piece = utf8.decode(enc.decode_single_token_bytes(vocab[internal_id]))  # map back to o200k_base
            pieces.append(piece)
            print(piece, end="", flush=True)
        pieces.append(utf8.decode(b"", final=True))
        print("\n", flush=True)
        text = prompt_text + "".join(pieces)
        samples_file.write(f"--- {label} ---\n{text}\n\n")
        samples_file.flush()
        log_txt.write(f"[sample: {label}] {text}\n")
        model.train()

    def get_batch():
        starts = torch.randint(0, len(data) - block, (args.batch_size,))
        x = torch.stack([data[i:i + block] for i in starts])
        y = torch.stack([data[i + 1:i + block + 1] for i in starts])
        return x, y

    sample("before training (step 0)")

    csv_file = open(args.out / "training_log.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["step", "loss", "elapsed_seconds", "tokens_per_second"])

    log(f"Training for about {args.seconds:.0f} seconds (sampling time is not counted)...")
    log("step | loss | elapsed time | tokens/second")
    steps, losses = [], []
    step, train_time, tokens_seen = 0, 0.0, 0
    model.train()
    while train_time < args.seconds:
        t0 = time.perf_counter()
        x, y = get_batch()
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_time += time.perf_counter() - t0

        step += 1
        tokens_seen += x.numel()
        tps = tokens_seen / train_time
        loss_value = loss.item()
        steps.append(step)
        losses.append(loss_value)
        log(f"{step:5d} | {loss_value:.4f} | {train_time:6.2f}s | {tps:,.0f}")
        writer.writerow([step, f"{loss_value:.6f}", f"{train_time:.3f}", f"{tps:.1f}"])
        csv_file.flush()

        if step % args.sample_every == 0:
            sample(f"step {step}")

    sample(f"end of training (step {step})")
    csv_file.close()
    samples_file.close()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, losses, linewidth=1, label="training loss")
    ax.axhline(math.log(V), linestyle="--", color="gray", label=f"uniform random guess: ln({V}) = {math.log(V):.2f}")
    ax.set_xlabel("update step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Tiny transformer on FineWeb sample ({step} steps, {train_time:.0f} s CPU)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "loss_plot.png", dpi=150)

    torch.save({
        "model_state": model.state_dict(),
        "config": model.config,
        "vocab": vocab,                 # internal ID i -> original o200k_base ID vocab[i]
        "tokenizer": "o200k_base",
        "prompt": prompt_text,
        "sample_seed": args.sample_seed,
        "steps": step,
        "train_seconds": train_time,
        "final_loss": losses[-1] if losses else None,
    }, args.out / "model.pt")

    log(f"Done: {step} steps in {train_time:.1f} s of training; final loss {losses[-1]:.4f}")
    log(f"Saved to {args.out}: training_log.txt, training_log.csv, loss_plot.png, samples.txt, model.pt")
    log_txt.close()


if __name__ == "__main__":
    main()

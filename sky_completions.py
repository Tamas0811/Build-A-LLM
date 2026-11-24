"""Load the trained tiny model and complete "The sky is" 100 times -> sky_completions.txt.

If any token of the prompt is missing from the model's small training vocabulary, the model
cannot read the prompt at all. The script then records which tokens are missing and stops.
"""
import argparse
import sys
from pathlib import Path

import tiktoken
import torch

from model import TinyGPT, generate_stream

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=HERE / "outputs" / "model.pt")
    p.add_argument("--out", type=Path, default=HERE / "sky_completions.txt")
    p.add_argument("--prompt", default="The sky is")
    p.add_argument("--n", type=int, default=100, help="number of completions")
    p.add_argument("--tokens", type=int, default=20, help="new tokens per completion")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    enc = tiktoken.get_encoding(ckpt["tokenizer"])
    vocab = ckpt["vocab"]                          # internal ID -> o200k_base ID
    to_internal = {tok: i for i, tok in enumerate(vocab)}

    prompt_ids = enc.encode(args.prompt, disallowed_special=())
    pieces = [enc.decode([t]) for t in prompt_ids]
    missing = [(piece, t) for piece, t in zip(pieces, prompt_ids) if t not in to_internal]

    header = [
        f"Prompt: {args.prompt!r}",
        f"Prompt tokens (o200k_base): " + ", ".join(f"{pc!r}={t}" for pc, t in zip(pieces, prompt_ids)),
        f"Model vocabulary: {len(vocab):,} tokens; checkpoint trained for {ckpt.get('steps')} steps",
    ]

    if missing:
        lines = header + [
            "",
            "FAILED: these prompt tokens never occurred in the training corpus, so the model has no",
            "embedding for them and cannot process the prompt:",
        ] + [f"  {pc!r} (o200k_base ID {t})" for pc, t in missing]
        args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        print(f"\nSaved to {args.out}")
        sys.exit(1)

    generator = torch.Generator().manual_seed(args.seed)
    lines = header + [f"Seed: {args.seed} | {args.n} completions | {args.tokens} new tokens each", ""]
    print("\n".join(lines))
    for k in range(1, args.n + 1):
        idx = torch.tensor([[to_internal[t] for t in prompt_ids]], dtype=torch.long)
        new_ids = [vocab[i] for i in generate_stream(model, idx, args.tokens, generator)]
        text = args.prompt + enc.decode(new_ids)
        line = f"{k:3d}. {text.replace(chr(10), ' ')}"
        lines.append(line)
        print(line, flush=True)

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved {args.n} completions to {args.out}")


if __name__ == "__main__":
    main()

"""Pick a short sentence that appears exactly in fineweb_corpus.txt, use its first few words as
the beginning, check every beginning token is in the trained model's vocabulary, and generate
100 completions -> corpus_completions.txt.
"""
import argparse
import re
import sys
from pathlib import Path

import tiktoken
import torch

from model import TinyGPT, generate_stream

HERE = Path(__file__).resolve().parent
COMMON_OPENINGS = ["The", "It", "This", "If", "In", "We", "I", "You", "A", "But", "And"]


def candidate_sentences(corpus, min_words, max_words):
    """Short sentences from the corpus, common openings first, then shorter, then corpus order."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", corpus)]
    found = []
    for pos, s in enumerate(sentences):
        words = s.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if not s[0].isalpha() or s not in corpus:      # must appear exactly in the corpus
            continue
        opening = words[0]
        rank = COMMON_OPENINGS.index(opening) if opening in COMMON_OPENINGS else len(COMMON_OPENINGS)
        found.append((rank, len(words), pos, s))
    found.sort()
    return [s for *_, s in found]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=HERE / "fineweb_corpus.txt")
    p.add_argument("--checkpoint", type=Path, default=HERE / "outputs" / "model.pt")
    p.add_argument("--out", type=Path, default=HERE / "corpus_completions.txt")
    p.add_argument("--begin-words", type=int, default=3, help="words of the sentence used as the beginning")
    p.add_argument("--min-words", type=int, default=5)
    p.add_argument("--max-words", type=int, default=12)
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
    vocab = ckpt["vocab"]                                # internal ID -> o200k_base ID
    to_internal = {tok: i for i, tok in enumerate(vocab)}

    corpus = args.corpus.read_text(encoding="utf-8-sig")
    chosen = beginning = begin_ids = None
    rejected = []
    for sentence in candidate_sentences(corpus, args.min_words, args.max_words):
        begin = " ".join(sentence.split()[:args.begin_words])
        ids = enc.encode(begin, disallowed_special=())
        missing = [enc.decode([t]) for t in ids if t not in to_internal]
        if missing:
            rejected.append(f"  {begin!r}: missing tokens {missing}")
            continue
        chosen, beginning, begin_ids = sentence, begin, ids
        break

    if chosen is None:
        print("No short corpus sentence has a beginning whose tokens are all in the vocabulary.")
        sys.exit(1)

    header = [
        f"Chosen sentence (appears exactly in {args.corpus.name}): {chosen!r}",
        f"Beginning: {beginning!r}",
        "Beginning tokens (o200k_base): " + ", ".join(f"{enc.decode([t])!r}={t}" for t in begin_ids),
        f"Model vocabulary: {len(vocab):,} tokens; checkpoint trained for {ckpt.get('steps')} steps",
        f"Seed: {args.seed} | {args.n} completions | {args.tokens} new tokens each",
    ]
    if rejected:
        header += ["Sentences skipped because a beginning token was not in the vocabulary:"] + rejected
    lines = header + [""]
    print("\n".join(lines))

    generator = torch.Generator().manual_seed(args.seed)
    start = [to_internal[t] for t in begin_ids]
    exact = 0
    for k in range(1, args.n + 1):
        idx = torch.tensor([start], dtype=torch.long)
        new_ids = [vocab[i] for i in generate_stream(model, idx, args.tokens, generator)]
        text = beginning + enc.decode(new_ids)
        if text.startswith(chosen):
            exact += 1
        line = f"{k:3d}. {text.replace(chr(10), ' ')}"
        lines.append(line)
        print(line, flush=True)

    summary = f"\nCompletions that reproduce the full original sentence: {exact} of {args.n}"
    lines.append(summary)
    print(summary)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

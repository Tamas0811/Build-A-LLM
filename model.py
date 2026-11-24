"""A tiny GPT-style causal transformer, shared by train.py and later sampling scripts."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block: causal self-attention followed by a small MLP."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x, causal_mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Predicts the next token from the previous tokens (randomly initialised)."""

    def __init__(self, vocab_size, block_size=64, n_embd=64, n_head=4, n_layer=2, dropout=0.0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, block_size=block_size, n_embd=n_embd,
                           n_head=n_head, n_layer=n_layer, dropout=dropout)
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # True = "may not attend": each position only sees itself and earlier positions.
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        for block in self.blocks:
            x = block(x, causal_mask)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def generate_stream(model, idx, max_new_tokens, generator, temperature=1.0):
    """Yield one sampled internal token ID at a time, so callers can print as they go."""
    for _ in range(max_new_tokens):
        context = idx[:, -model.block_size:]
        logits, _ = model(context)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat([idx, next_id], dim=1)
        yield next_id.item()


# 07 — Transformers & LLMs (Orientation)

This page connects your overview notes to how modern NLP models operate.

## Architecture sketch
```mermaid
flowchart LR
X[Text] --> T[Tokenizer (BPE/WordPiece)] --> E[Embedding + Positional Encoding]
E --> B1[Self‑Attention Block] --> B2[×N Blocks]
B2 --> H[Task Head\nclassification/generation]
```

- **Self‑attention** lets each token attend to others and build context.
- **Decoder‑only LMs** (GPT‑style) generate text; **Encoder** models (BERT) embed text for classification/extraction; **Encoder‑Decoder** (T5) handle seq2seq like translation.

## Decoding
Greedy, beam search, top‑k, nucleus (top‑p) sampling; temperature controls randomness.

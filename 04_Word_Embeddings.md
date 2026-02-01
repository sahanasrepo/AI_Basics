# 04 — Word Embeddings: Word2Vec, CBOW & Skip-gram

> **Prerequisites:** `03_NLP.md` (One-Hot Encoding, BoW, TF-IDF) and `02_Deep_Learning.md` (neural networks, softmax, backpropagation).  
> **What you'll know after:** Why one-hot encoding fails at scale, how Word2Vec learns dense vector representations, and the two architectures (CBOW & Skip-gram) that power it.

---

## 1. Why One-Hot Encoding Breaks Down

Recall from NLP basics: one-hot encoding gives each word a sparse binary vector.

```
Vocabulary: [Krish, Channel, is, related, to, data, science]   (7 words)

Krish   → [1, 0, 0, 0, 0, 0, 0]
Channel → [0, 1, 0, 0, 0, 0, 0]
is      → [0, 0, 1, 0, 0, 0, 0]
…
```

**Problems at scale:**

| Problem | Why it hurts |
|---------|--------------|
| Dimensionality | Real vocabularies have 50,000–500,000 words → vectors are enormous |
| No semantic info | `king` and `queen` are just as "far apart" as `king` and `table` |
| Memory & compute | Sparse matrices waste resources |

**Solution: Word Embeddings** — represent each word as a **dense, low-dimensional vector** (typically 50–300 dimensions) where **semantic similarity maps to vector similarity**.

---

## 2. Word2Vec — The Big Idea

Word2Vec (Google, 2013) learns embeddings by training a **shallow neural network on a fake prediction task**. We don't actually care about the prediction — we want the **weights learned along the way**, because those weights *are* the embeddings.

**Two architectures, same goal:**

| Architecture | Task | Input | Output |
|---|---|---|---|
| **CBOW** | Predict target word from context | Context words | Target word |
| **Skip-gram** | Predict context words from target | Target word | Context words |

---

## 3. CBOW — Continuous Bag of Words

### 3.1 The Concept

> **Goal:** Given the surrounding context words, predict the word in the middle.

```
Sentence:  "Krish channel is related to data science"

Context window around "is":
  Context words → [Krish, channel, related, to]     (window size = 2 each side)
  Target word   → is

The network learns: context [Krish, channel, related, to] → predicts "is"
```

### 3.2 The Neural Network Architecture

```
  One-hot vectors                  Hidden Layer          Output Layer
  (context words)                  (Embedding)           (Softmax)

  Krish   [1,0,0,0,0,0,0]  ─►┐
                              ├─► Average ─► h ─► W_out ─► Softmax ─► P(word)
  Channel [0,1,0,0,0,0,0]  ─►┘
                                   ↑
  related [0,0,0,1,0,0,0]  ─►─────┘
  to      [0,0,0,0,1,0,0]  ─►─────┘

  Input: C context words (one-hot)
  W_in:  Vocab × Embedding_dim  (input weight matrix)
  h:     Average of the C embedding vectors
  W_out: Embedding_dim × Vocab  (output weight matrix)
  Output: Probability distribution over entire vocabulary
```

### 3.3 Full Worked Example — "I love dogs"

**Setup:**
- Sentence: `"I love dogs"`
- Target word: `dogs`
- Context words: `I`, `love`  (window size = 1 each side)
- Vocabulary: `['I', 'love', 'dogs', 'cats']` → size 4
- Embedding dimension: **2** (tiny for illustration)

**Step 1 — One-Hot Encode:**

```
I    → [1, 0, 0, 0]
love → [0, 1, 0, 0]
dogs → [0, 0, 1, 0]
cats → [0, 0, 0, 1]
```

**Step 2 — Input Weight Matrix W (4 × 2), randomly initialised:**

```
        dim₁   dim₂
W  =  [ 0.2    0.8  ]   ← I
      [ 0.6    0.4  ]   ← love
      [ 0.9    0.1  ]   ← dogs
      [ 0.3    0.7  ]   ← cats
```

**Step 3 — Look up embeddings for context words:**

```
Embedding(I)    = W[0] = [0.2, 0.8]
Embedding(love) = W[1] = [0.6, 0.4]
```

**Step 4 — Average the context embeddings → hidden vector h:**

$$
h = \frac{1}{C} \sum_{i=1}^{C} W \cdot x_i = \frac{1}{2}\big([0.2, 0.8] + [0.6, 0.4]\big) = [0.4, 0.6]
$$

**Step 5 — Compute output scores (h × W_out):**

The output score for each word in the vocabulary:

$$
\text{score} = W^T \cdot h = [0.32,\; 0.46,\; 0.42,\; 0.51]
$$

**Step 6 — Softmax → probabilities:**

$$
\text{Softmax}(u) = \frac{e^{u_i}}{\sum_j e^{u_j}}
$$

```
I:    0.212
love: 0.244
dogs: 0.228   ← we want THIS to be highest
cats: 0.253   ← but cats is highest (model is wrong!)
```

**Step 7 — Loss & Backprop:**

The model predicted `cats` but the answer is `dogs`. The loss is high. Backpropagation adjusts the weights so that next time, `dogs` scores higher given this context. After many iterations over the corpus, the embeddings converge.

### 3.4 What the Embeddings Learn

After training on a large corpus, the embedding matrix `W` captures **semantic relationships**:

```
          Feature 1   Feature 2   …
Boy       0.2         0.8
Girl      0.1         0.9         ← similar to Boy (both are people)
King      0.7         0.5
Queen     0.6         0.6         ← similar to King (both are royals)
Apple     0.9         0.1
Mango     0.8         0.2         ← similar to Apple (both are fruits)
```

The famous analogy: `King − Man + Woman ≈ Queen` works because the embedding space preserves these relationships as **directions**.

---

## 4. Skip-gram — The Flipped Task

### 4.1 The Concept

> **Goal:** Given one target word, predict each of the surrounding context words.

CBOW and Skip-gram are **exactly the same architecture with input and output swapped**.

```
                    Input (target)          Output (context words)
                    ─────────────           ───────────────────────
CBOW:               Context words    →      Target word
Skip-gram:          Target word      →      Context words
```

### 4.2 Example

```
Sentence: "Krish channel is related to data science"
Window size = 2

Target: "is"
Context: [Krish, channel, related, to]

Skip-gram pairs:
  (is, Krish)
  (is, channel)
  (is, related)
  (is, to)

Each pair is a separate training example.
```

### 4.3 CBOW vs Skip-gram — When to Use Which

| | CBOW | Skip-gram |
|---|---|---|
| **Speed** | Faster to train | Slower (more training pairs) |
| **Accuracy** | Better for frequent words | Better for rare words & phrases |
| **Training data** | Needs more data | Works well with less data |
| **Default choice** | When speed matters | When rare-word quality matters |

---

## 5. The Embedding Layer in Deep Learning

In practice, you never manually do the one-hot → matrix multiply step. Instead, you use an **Embedding Layer** — it's just a lookup table that directly returns the dense vector for a given word index.

```
Word index: 325  →  Embedding Layer  →  [0.12, -0.45, 0.78, …]  (300-dim vector)
```

This is exactly what Word2Vec learns — and it's what the Embedding layer in PyTorch/TensorFlow implements.

---

## 6. Code

### PyTorch — Training Word2Vec (CBOW)

```python
import torch
import torch.nn as nn
import numpy as np

class Word2Vec_CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim)   # W_in
        self.output     = nn.Linear(embed_dim, vocab_size)      # W_out

    def forward(self, context_indices):
        # context_indices: (batch, num_context_words)
        embeds = self.embedding(context_indices)                 # (batch, C, embed_dim)
        h      = embeds.mean(dim=1)                              # average → (batch, embed_dim)
        scores = self.output(h)                                  # (batch, vocab_size)
        return scores   # raw logits; use CrossEntropyLoss (includes softmax)

# --- Training loop sketch ---
vocab_size  = 10000
embed_dim   = 128
model       = Word2Vec_CBOW(vocab_size, embed_dim)
loss_fn     = nn.CrossEntropyLoss()
optim       = torch.optim.Adam(model.parameters(), lr=0.001)

for context, target in dataloader:          # context: (batch, C), target: (batch,)
    scores = model(context)
    loss   = loss_fn(scores, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

# After training, the learned embeddings:
embeddings = model.embedding.weight.detach().numpy()   # (vocab_size × embed_dim)
```

### TensorFlow / Keras — Embedding Layer Usage

```python
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential

vocab_size = 10000
embed_dim  = 128

model = Sequential([
    Embedding(vocab_size, embed_dim, input_length=4),   # 4 context words
    GlobalAveragePooling1D(),                            # average the embeddings
    Dense(vocab_size, activation='softmax')              # predict target word
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(X_context_onehot, y_target_onehot, epochs=50)
```

### Using Pre-trained Embeddings with Gensim

```python
from gensim.models import Word2Vec

# Train on your own corpus
sentences = [["I", "love", "dogs"], ["dogs", "are", "great"], …]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get the embedding for a word
print(model.wv["dogs"])           # 100-dim vector

# Find most similar words
print(model.wv.most_similar("dogs", topn=5))

# The famous analogy
result = model.wv.most_similar(positive=["king", "woman"], negative=["man"])
print(result)  # → queen (hopefully!)
```

---

## 7. Summary

| Concept | Key Takeaway |
|---------|-------------|
| One-Hot Encoding | Sparse, high-dimensional, no semantic meaning |
| Word Embeddings | Dense, low-dimensional, captures meaning |
| CBOW | Context → Target. Faster, better for frequent words |
| Skip-gram | Target → Context. Better for rare words |
| Word2Vec trick | We don't care about the prediction task — we want the **weights** |
| Embedding Layer | A learnable lookup table; the standard way to use embeddings in DL |

**Next up → `05_CNN.md`** — Convolutional Neural Networks for image recognition.

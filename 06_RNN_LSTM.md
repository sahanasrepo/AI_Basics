# 06 — Recurrent Neural Networks & LSTM

> **Prerequisites:** `02_Deep_Learning.md` (backpropagation, vanishing gradient), `04_Word_Embeddings.md` (word vectors).  
> **What you'll know after:** How RNNs process sequences, why they fail on long inputs, and exactly how LSTM's three gates solve the problem — gate by gate.

---

## 1. Why We Need Sequence Models

Standard neural networks (ANNs, CNNs) take a **fixed-size input** and produce an output. But many real-world problems involve **sequences** — data where **order matters** and the length can vary.

| Application | Input Sequence | Output |
|---|---|---|
| Sentiment Analysis | "The food is good" | Positive |
| Language Translation | Hindi sentence | English sentence |
| Text Generation | "Once upon a…" | continuation of story |
| Chatbot | User question | Answer |
| Music Generation | First 8 bars | Next 8 bars |
| Stock Prediction | Last 30 days | Tomorrow's price |

All of these need a model that can **remember what came before** while processing each new element. That is what **Recurrent Neural Networks (RNN)** were designed for.

---

## 2. RNN Architecture — The Core Idea

An RNN has a **loop**: the output (hidden state) from one time step is fed back as input to the next.

```
  Standard NN:            RNN (unrolled over time):

  x ──► [NN] ──► y       x₁ ──► [NN] ──► h₁ ──► y₁
                          x₂ ──► [NN] ──► h₂ ──► y₂     h₁ feeds into step 2
                          x₃ ──► [NN] ──► h₃ ──► y₃     h₂ feeds into step 3
                          …                              …
```

At each time step `t`:

$$
h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b)
$$

$$
y_t = W_{hy} \cdot h_t
$$

| Symbol | Meaning |
|--------|---------|
| `x_t` | Input at time t |
| `h_t` | Hidden state at time t (the "memory") |
| `h_{t-1}` | Hidden state from previous time step |
| `W_xh` | Weights: input → hidden |
| `W_hh` | Weights: hidden → hidden (the recurrence) |
| `y_t` | Output at time t |

### 2.1 Types of RNN Architectures

```
① One-to-One          ② One-to-Many         ③ Many-to-One
   ↑ output               ↑ ↑ ↑ outputs         ↑ output
   │                      │ │ │                   │
  [●]                    [●][●][●]              [●][●][●]
   ↑                      ↑                      ↑ ↑ ↑
  input                 input               inputs
  (image classify)    (music gen)        (sentiment analysis)

④ Many-to-Many (synced)     ⑤ Many-to-Many (unsynced)
  ↑ ↑ ↑ ↑ outputs             (encoder)   (decoder)
  │ │ │ │                    [●][●][●] → [●][●][●]
 [●][●][●][●]                ↑ ↑ ↑        ↑ ↑ ↑
  ↑ ↑ ↑ ↑                  inputs       outputs
  inputs                 (language translation)
  (video labeling)
```

| Type | Example |
|------|---------|
| One-to-Many | Music generation, text generation |
| Many-to-One | Sentiment analysis, predict next stock price |
| Many-to-Many (synced) | Video frame labelling |
| Many-to-Many (unsynced) | Language translation, Q&A, chatbots |

---

## 3. Forward Propagation in RNN — Worked Example

> **Task:** Sentiment analysis on "The food is very good" → Positive

**Setup:** Each word is converted to a vector via Word2Vec first, then fed sequentially.

```
Words:   "The"  "food"  "is"  "very"  "good"
Vectors:  x₁     x₂      x₃    x₄     x₅

Time steps:
  t=1: h₁ = tanh(W·x₁ + 0)                    ← h₀ = 0 (no history yet)
  t=2: h₂ = tanh(W·x₂ + W_hh·h₁)
  t=3: h₃ = tanh(W·x₃ + W_hh·h₂)
  t=4: h₄ = tanh(W·x₄ + W_hh·h₃)
  t=5: h₅ = tanh(W·x₅ + W_hh·h₄)

Final output: ŷ = softmax(W_hy · h₅)  →  [P(positive), P(negative)]
```

Each `hₜ` theoretically contains information about **all words seen so far**. In practice, earlier words fade — that is the problem we solve with LSTM.

---

## 4. The Vanishing Gradient Problem in RNNs

During backpropagation through time (BPTT), gradients are multiplied by `W_hh` and by `tanh'()` at every time step going backwards:

```
  ∂L/∂h₁ = ∂L/∂h₂ · ∂h₂/∂h₁ · … 
         = ∂L/∂h₅ · (tanh' · W_hh)⁴

  tanh' is at most 1.0, and W_hh entries are typically < 1
  → multiply together 50 times → gradient ≈ 0
```

**Result:** The RNN has **only short-term memory**. It loses context when the sentence is long or when critical information appeared far back.

> **Example:** *"I am a woman in the city. [long sentence…] I like to travel by bus."*  
> When predicting the pronoun for the subject, the RNN has already "forgotten" that the subject was a woman.

---

## 5. LSTM — Long Short-Term Memory

### 5.1 Why LSTM?

| Problem with RNN | How LSTM fixes it |
|---|---|
| Only short-term memory | Adds a **cell state** (long-term memory highway) |
| Vanishing gradient | Cell state passes through with **minimal modification** — gradient flows freely |
| Can't selectively forget | **Forget gate** decides what to discard |
| Can't selectively remember | **Input gate** decides what to store |

### 5.2 The Big Picture — Cell State as a Conveyor Belt

```
  C_{t-1} ───────────────────────────────────────► C_t
           ─── ×f_t ───  ─── +i_t·C̃_t ───
                ↑                ↑
           (forget gate)    (input gate)
           "how much to     "how much new
            keep from past"  info to add"
```

The cell state (`C`) is the **long-term memory**. Information flows along it with only **pointwise multiplication and addition** — no squashing functions — so gradients don't vanish.

The hidden state (`h`) is the **short-term output** passed to the next layer or used as prediction.

### 5.3 The Three Gates — Step by Step

Each gate is a small neural network that outputs a number between 0 and 1 (via sigmoid). Think of it as a **valve**: 0 = fully closed, 1 = fully open.

---

#### Gate 1: Forget Gate (`f_t`) — "What to throw away"

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

```
  Previous hidden state  h_{t-1} ─►┐
  Current input          x_t    ─►├─► sigmoid ─► f_t  (0 to 1 per cell)
                                   └─► (weights W_f, bias b_f)

  f_t is then multiplied with C_{t-1}:
  If f_t ≈ 0  → forget that piece of old memory
  If f_t ≈ 1  → keep that piece of old memory
```

**Worked intuition:**

> *"Krisha likes pizza but his friend plays the piano."*
>
> - After reading "Krisha likes pizza" → cell state stores {subject: Krisha, likes: pizza}
> - When we read "his friend" → forget gate fires for {subject} because a new subject is coming
> - `f_t ≈ 0` for the subject slot → old subject info is erased

---

#### Gate 2: Input Gate (`i_t`) + Candidate (`C̃_t`) — "What new info to store"

This gate has **two parts**:

**Part A — Input gate (how much to update):**

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
$$

**Part B — Candidate cell state (what the new info looks like):**

$$
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

`tanh` squashes values to [−1, +1], so the candidate can add or subtract from the cell state.

```
  h_{t-1}, x_t ─► sigmoid ─► i_t          (how much?)
  h_{t-1}, x_t ─► tanh    ─► C̃_t         (what value?)
  
  New info added = i_t ⊙ C̃_t             (element-wise multiply)
```

**Intuition:** When we read "plays the piano" → input gate opens for {activity} → candidate carries {activity: piano} → this gets written into the cell state.

---

#### Cell State Update — Combine forget + input

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

```
  C_t  =  (forget gate × old state)  +  (input gate × candidate)
         ─────────────────────────     ─────────────────────────
         "keep what we want"           "add what's new"
```

This is the **key equation**. It's just addition and multiplication — no squashing — so gradients flow through cleanly.

---

#### Gate 3: Output Gate (`o_t`) — "What to output"

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
$$

$$
h_t = o_t \odot \tanh(C_t)
$$

```
  C_t ─► tanh ─► (values in [-1, 1])
                         ↓
  o_t ─────────────────► ⊙ ──► h_t  (the output / next hidden state)
```

The output gate decides **which parts of the cell state to expose** as the hidden state. Not everything in memory needs to be used at every step.

---

### 5.4 Full LSTM — All Equations Together

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) \quad \text{— Forget gate}
$$

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \quad \text{— Input gate}
$$

$$
\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c) \quad \text{— Candidate}
$$

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \quad \text{— Cell state update}
$$

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \quad \text{— Output gate}
$$

$$
h_t = o_t \odot \tanh(C_t) \quad \text{— Hidden state (output)}
$$

### 5.5 Full Architecture Diagram

```
                        ┌──────── C_t ────────────────────►
                        │         ↑
          ┌─── f_t ─►  (×)       (+)  ◄── i_t ⊙ C̃_t
          │             ↑         ↑
  h_{t-1} ─┼── i_t ─►  │    ┌── tanh ── W_c
          │             │    │
          └── o_t ─►   │    └── x_t
                        │
  C_{t-1} ──────────────┘
  
  h_t = o_t ⊙ tanh(C_t)  ──────► output / next step
```

---

## 6. Backpropagation in RNN — The Vanishing Gradient Revisited

In a standard RNN, the weight update for early time steps involves:

$$
\frac{\partial L}{\partial w'} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_{t-1}} \cdot \ldots
$$

The chain rule multiplies through **every time step** — and sigmoid/tanh derivatives shrink the gradient each time. With LSTM, the cell state update is **additive**, so the gradient flows through the addition path without shrinking.

---

## 7. Statistical Distributions — Quick Reference

Your notes covered data distributions used in NLP/ML preprocessing. Here is a quick summary:

| Distribution | Shape | When it appears |
|---|---|---|
| **Normal (Gaussian)** | Bell curve, symmetric | Most natural data after CLT; assumed by StandardScaler |
| **Log-Normal** | Skewed right | Prices, income, reaction times |
| **Pareto** | Heavy right tail | Wealth distribution, file sizes |

**Key stats for Normal distribution:**
- Mean = Median = Mode (all equal — perfectly symmetric)
- ~68% of data within 1 std dev, ~95% within 2, ~99.7% within 3

**Converting Pareto → Normal:** Use **Box-Cox Transformation** (a family of power transforms that makes skewed data approximately normal).

**Checking normality:** Use **Q-Q plots** (quantile-quantile) — if points fall on a straight line, data is approximately normal.

---

## 8. Code — RNN and LSTM

### PyTorch — LSTM Sentiment Classifier

```python
import torch
import torch.nn as nn

class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len) — token indices
        embeds = self.embedding(x)              # (batch, seq_len, embed_dim)
        _, (h_n, _) = self.lstm(embeds)         # h_n: (1, batch, hidden_dim)
        out = self.fc(h_n.squeeze(0))           # (batch, num_classes)
        return out

# --- Training ---
model   = LSTMSentiment(vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=2)
loss_fn = nn.CrossEntropyLoss()
optim   = torch.optim.Adam(model.parameters(), lr=0.001)

for batch_x, batch_y in dataloader:
    logits = model(batch_x)
    loss   = loss_fn(logits, batch_y)
    optim.zero_grad()
    loss.backward()
    optim.step()
```

### TensorFlow / Keras — LSTM Text Classifier

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

vocab_size  = 10000
embed_dim   = 128
hidden_dim  = 64
max_len     = 200      # max sequence length (pad/truncate)

model = Sequential([
    Embedding(vocab_size, embed_dim, input_length=max_len),
    LSTM(hidden_dim),                          # returns only final h_t
    Dropout(0.3),
    Dense(1, activation='sigmoid')             # binary: positive/negative
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

### Key PyTorch vs Keras Differences

| | PyTorch | Keras |
|---|---|---|
| LSTM returns | `output, (h_n, c_n)` | `output` or `(output, h, c)` via `return_state=True` |
| Sequence output | Use `output` (all time steps) | Set `return_sequences=True` |
| Stacking LSTMs | Add multiple `nn.LSTM` layers | Stack `LSTM` layers with `return_sequences=True` |

---

## 9. Summary

| Concept | Key Takeaway |
|---------|-------------|
| RNN | Loops hidden state back — gives sequence memory |
| RNN types | 1-to-1, 1-to-many, many-to-1, many-to-many |
| Vanishing gradient | RNN gradients shrink exponentially over long sequences |
| LSTM Cell State | Long-term memory highway — gradient flows freely |
| Forget Gate | Decides what old info to discard (sigmoid → 0 or 1) |
| Input Gate | Decides what new info to store |
| Candidate | The actual new values (tanh → [−1, +1]) |
| Cell Update | `C_t = f_t⊙C_{t-1} + i_t⊙C̃_t` — the core equation |
| Output Gate | Decides what part of cell state becomes the output |

**Next up → `07_Interview_and_Quick_Reference.md`** — Interview Q&A, model comparison cheat sheet, and the full learning path map.

# 01 — Foundations: Probability, Bayes & Linear Regression

> **Prerequisites:** Basic algebra, understanding of what a function is.  
> **What you'll know after:** How Bayesian reasoning works, how supervised learning is framed mathematically, and how a simple linear model learns from data.

---

## 1. The ML Landscape — What Actually Sits Where

Before diving in, it helps to know what the terms mean and how they nest inside each other.

```
┌─────────────────────────────────────────────────────────┐
│                        AI                               │
│   Apps that can do their own task without human         │
│   intervention (Netflix, Amazon, Chatbots, Self-Drive)  │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │              Machine Learning                   │   │
│   │   Stats tool to analyse data, visualise,        │   │
│   │   predict, forecast. Uses algorithms that       │   │
│   │   learn patterns from data & improve over       │   │
│   │   time without being explicitly programmed.     │   │
│   │                                                 │   │
│   │   ┌───────────────────────────────────────┐     │   │
│   │   │         Deep Learning                 │     │   │
│   │   │   A specialised branch of ML that     │     │   │
│   │   │   uses neural networks to model      │     │   │
│   │   │   complex patterns.                  │     │   │
│   │   │   (It mimics the human brain)        │     │   │
│   │   └───────────────────────────────────────┘     │   │
│   └─────────────────────────────────────────────────┘   │
│                                                         │
│   Data Science: uses stats, programming, domain        │
│   knowledge to extract insights from data.             │
└─────────────────────────────────────────────────────────┘
```

**Why DL became popular — the three catalysts:**

| Year | Event |
|------|-------|
| 2005 | Emergence of social platforms (Orkut, FB, Instagram, WhatsApp) → data grew exponentially |
| 2008 | Big Data era → tools to store data efficiently |
| 2013 | Industry wanted to *use* that data to improve products |

Two enabling resources: **(1) Data** — mountains of it, and **(2) Hardware** — NVIDIA GPUs made training feasible.

---

## 2. Supervised vs Unsupervised Learning

### 2.1 Supervised Learning

The algorithm **learns from labelled examples** — each training sample comes with the correct answer attached.

```
Training Set ──► Features + Targets
       │
       ▼
  Learning Algorithm  ──► Trained Model
       │
       ▼
  Test on unseen data ──► Accuracy Score
```

Supervised learning splits into two tasks:

| | Regression | Classification |
|---|---|---|
| **Goal** | Predict a continuous number | Predict a category / label |
| **Possible outputs** | Infinitely many | Small finite set |
| **Example** | House price from size | Is this email spam? |
| **Output layer** | Single linear neuron | Softmax (multi) / Sigmoid (binary) |

### 2.2 Unsupervised Learning

Find something **interesting in unlabelled data** — the algorithm discovers structure on its own.

```
Raw unlabelled data ──► Algorithm finds patterns ──► Clusters / Groups
```

Common tasks: **Clustering** (group similar data points), **Dimensionality Reduction** (compress data using fewer features), **Anomaly Detection** (find unusual data points).

### 2.3 Key Terminology

| Symbol | Meaning |
|--------|---------|
| `x` | Input variable / feature (e.g. house size) |
| `y` | Output variable / target (e.g. house price) |
| `m` | Number of training examples |
| `(x, y)` | A single training example |
| `(x⁽ⁱ⁾, y⁽ⁱ⁾)` | The iᵗʰ training example |
| `ŷ` (y-hat) | The model's *prediction* |

> **Superscript notation:** `x⁽¹⁾` is the first training example, `x⁽²⁾` the second — it is **not** exponentiation.

---

## 3. Bayes' Theorem — Updating Beliefs with Evidence

This is one of the most important ideas in all of ML. The intuition: **start with a prior belief, then update it when you see new evidence.**

### 3.1 The Formula

$$
P(H|E) = \frac{P(H) \times P(E|H)}{P(E)}
$$

| Term | Name | Meaning |
|------|------|---------|
| `P(H\|E)` | **Posterior** | Probability of hypothesis *after* seeing the evidence |
| `P(H)` | **Prior** | Your belief *before* seeing the evidence |
| `P(E\|H)` | **Likelihood** | How likely would you see this evidence *if* the hypothesis were true |
| `P(E)` | **Evidence** | Total probability of seeing this evidence (normaliser) |

### 3.2 Worked Example — The Librarian vs Farmer Problem

> *Steve is very shy, withdrawn, meets & tidies soul…*  
> **Is Steve a librarian or a farmer?**

**Step 1 — Prior (before reading the description):**

In the US there are roughly **20 farmers for every 1 librarian**.

```
Prior ratio:   P(Librarian) = 1/21 ≈ 0.048
               P(Farmer)    = 20/21 ≈ 0.952
```

Most people *ignore* this and guess "librarian" just because the description sounds librarian-like. That is a **base-rate fallacy**.

**Step 2 — Likelihood (how well does the evidence fit each hypothesis):**

| | Fits the description? |
|---|---|
| `P(E\|Librarian)` | 40% of librarians fit "shy, tidy" |
| `P(E\|Farmer)` | 10% of farmers fit "shy, tidy" |

**Step 3 — Evidence (total probability of seeing this description):**

$$
P(E) = P(E|H_{lib}) \times P(H_{lib}) + P(E|H_{far}) \times P(H_{far})
$$

$$
P(E) = 0.4 \times \frac{1}{21} + 0.1 \times \frac{20}{21} = \frac{0.4 + 2.0}{21} = \frac{2.4}{21} \approx 0.114
$$

**Step 4 — Posterior:**

$$
P(\text{Librarian}|E) = \frac{0.4 \times \frac{1}{21}}{0.114} = \frac{0.019}{0.114} \approx 0.167
$$

$$
P(\text{Farmer}|E) = 1 - 0.167 = 0.833
$$

> **Conclusion:** Despite the "librarian-sounding" description, the math says Steve is **~83% likely a farmer** — because farmers vastly outnumber librarians. The prior dominates when the likelihood ratio isn't extreme enough to flip it.

### 3.3 Visual Intuition

```
ALL people
├── 210 Librarians          ← 4 librarians  fit the desc  (40%)
│       └── 4 fit desc ██
└── 4200 Farmers            ← 420 farmers   fit the desc  (10%)
        └── 420 fit desc ██████████████████████████

Of those who fit the description:
  Librarians: 4 / 424 ≈ 0.94%   ← surprisingly low!
  Farmers:  420 / 424 ≈ 99.06%
```

### 3.4 Key Concept

> **New evidence should always update prior beliefs.**  
> First you make a hypothesis based on assumptions; once you see evidence, the hypothesis *has* to be updated.

---

## 4. Linear Regression — A Model That Learns

### 4.1 What Is It?

Given labelled data (e.g. house sizes and prices), fit a straight line so we can **predict** new values.

```
Price ↑        * *
      |      *   *  *
      |    *   ╱
      |  * ╱  *         ← Best-fit line: ŷ = wx + b
      | ╱  *
      |╱ *
      └──────────────── → Size
```

The model function is:

$$
\hat{y}^{(i)} = f_{w,b}(x^{(i)}) = w \cdot x^{(i)} + b
$$

| Parameter | Role |
|-----------|------|
| `w` | Slope — how much the output changes per unit of input |
| `b` | Bias (intercept) — the output when input is zero |

**Goal:** Find `w` and `b` such that `ŷ` is as close as possible to the true `y` for all training examples.

### 4.2 The Cost Function — How Bad Is Our Guess?

The **cost function** measures how poorly the model is doing across all training data. For regression we use **Mean Squared Error (MSE)**:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2
$$

| Symbol | Meaning |
|--------|---------|
| `m` | Number of training examples |
| The `½` | Makes the derivative cleaner (cosmetic) |
| Squaring | Penalises large errors more than small ones |

> The curve of `J(w)` for a fixed `b` is a **parabola** — it has exactly **one global minimum**. That minimum is where our model is best.

### 4.3 Worked Calculation

Say we have 3 training points: `(0, 0)`, `(1, 1)`, `(2, 3)` and we try `w = 1, b = 0`:

```
x    y    ŷ = 1·x + 0    error = ŷ − y    error²
0    0    0               0                 0
1    1    1               0                 0
2    3    2               −1                1

J(1, 0) = (0 + 0 + 1) / (2 × 3) = 1/6 ≈ 0.167
```

Now try `w = 0.5`:

```
x    y    ŷ = 0.5x        error     error²
0    0    0               0         0
1    1    0.5             −0.5      0.25
2    3    1.0             −2.0      4.0

J(0.5, 0) = (0 + 0.25 + 4) / 6 = 4.25 / 6 ≈ 0.708   ← worse
```

We pick `w = 1` because `J` is lower. The optimiser does this search automatically.

### 4.4 Simplified Form (drop bias for clarity)

When `b = 0`, the model becomes `f(x) = wx` and the cost function simplifies to:

$$
J(w) = \frac{1}{2m} \sum_{i=1}^{m} (wx^{(i)} - y^{(i)})^2
$$

---

## 5. Code — Linear Regression from Scratch

### Python / NumPy

```python
import numpy as np

# Training data: house sizes → prices
X = np.array([0, 1, 2, 3, 4])          # features
y = np.array([0, 1, 3, 2, 4])          # targets
m = len(y)

def predict(X, w, b):
    return w * X + b

def cost(X, y, w, b):
    predictions = predict(X, w, b)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Brute-force search over w (b=0 for simplicity)
best_w, best_cost = 0, float('inf')
for w in np.arange(0, 2, 0.01):
    c = cost(X, y, w, 0)
    if c < best_cost:
        best_w, best_cost = w, c

print(f"Best w = {best_w:.2f}, Cost = {best_cost:.4f}")
```

### PyTorch

```python
import torch
import torch.nn as nn

X = torch.tensor([[0.], [1.], [2.], [3.], [4.]])
y = torch.tensor([[0.], [1.], [3.], [2.], [4.]])

model   = nn.Linear(1, 1)                          # y = wx + b
loss_fn = nn.MSELoss()
optim   = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

print(f"w={model.weight.item():.3f}  b={model.bias.item():.3f}  loss={loss.item():.4f}")
```

### TensorFlow / Keras

```python
import tensorflow as tf
from tensorflow import keras

X = tf.constant([[0.], [1.], [2.], [3.], [4.]])
y = tf.constant([[0.], [1.], [3.], [2.], [4.]])

model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=1000, verbose=0)

w, b = model.layers[0].get_weights()
print(f"w={w[0][0]:.3f}  b={b[0]:.3f}")
```

---

## 6. Summary & What Comes Next

| Concept | Key Takeaway |
|---------|-------------|
| AI / ML / DL | Nested hierarchy — DL is a *subset* of ML, which is a subset of AI |
| Supervised Learning | Learn from labelled data; splits into regression (numbers) and classification (categories) |
| Bayes' Theorem | Always factor in base rates — evidence alone can mislead |
| Linear Regression | Simplest supervised model: `ŷ = wx + b`, optimise `w` and `b` to minimise MSE |
| Cost Function | A single number that tells you how bad your model is right now |

**Next up → `02_Deep_Learning.md`** — Neural networks, activation functions, how gradient descent *actually* updates weights, backpropagation, and loss functions for classification.

# 02 — Deep Learning: Neural Networks, Training & Optimisation

> **Prerequisites:** `01_Foundations.md` — you must be comfortable with cost functions and the idea of minimising error.  
> **What you'll know after:** How neural networks are structured, how they learn through backpropagation, every major activation and loss function, and how modern optimisers work.

---

## 1. From Perceptron to Neural Network

### 1.1 The Perceptron — One Neuron

A perceptron is the simplest possible neural network: **one layer, one neuron**.

```
        w₁
  x₁ ──────►┐
        w₂  │
  x₂ ──────►├──► Σ(xᵢwᵢ) + b ──► Activation ──► Output (ŷ)
        w₃  │
  x₃ ──────►┘
```

**The math inside one neuron:**

$$
z = \sum_{i} x_i \cdot w_i + b = w^T x + b
$$

$$
\hat{y} = \sigma(z)       \quad \text{(activation function applied)}
$$

| Component | Role |
|-----------|------|
| `xᵢ` | Inputs (features) |
| `wᵢ` | Weights — learnable, control how much each input matters |
| `b` | Bias — added so the neuron can activate even when all inputs are 0 |
| `σ()` | Activation function — adds non-linearity |

### 1.2 Worked Example — Binary Classification

> Dataset: Does a student pass the exam?  
> Features: Study hours, Play hours, Sleep hours.

| Study | Play | Sleep | Pass? |
|-------|------|-------|-------|
| 7     | 3    | 7     | 1     |
| 2     | 5    | 8     | 0     |
| 4     | 3    | 7     | 1     |

The perceptron computes `z = w₁·Study + w₂·Play + w₃·Sleep + b`, then applies sigmoid to get a probability between 0 and 1. If the output ≥ 0.5 → predict **Pass**, otherwise **Fail**.

### 1.3 Multi-Layer Neural Network (ANN)

When you stack layers, you get a **fully connected (dense) network**:

```
  Input Layer        Hidden Layer 1     Hidden Layer 2     Output
  (features)

  ● ─────────────►  ●   ●   ●   ●     ●   ●   ●         ●  ──► ŷ
  ● ─────────────►  ●   ●   ●   ●  ►  ●   ●   ●
  ● ─────────────►  ●   ●   ●   ●
                     (each ● connects to every ● in next layer)
```

Every neuron in layer L connects to **every** neuron in layer L+1 — hence "fully connected". Each connection has its own weight.

---

## 2. Activation Functions

Activation functions are applied **after** the weighted sum inside each neuron. Without them, stacking layers would be pointless (a stack of linear functions is still just one linear function).

### 2.1 Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

```
  1.0 ┤                    ___________
      │                 ╱
  0.5 ┤───────────────╱──────────────── (threshold)
      │            ╱
  0.0 ┤__________╱
      └─────────────────────────────── x
     -5   -2    0    2    5
```

| Property | Value |
|----------|-------|
| Range | (0, 1) |
| Derivative | σ(x)·(1 − σ(x)), **max = 0.25** |
| Used for | Binary classification output layer |
| Problem | **Vanishing gradient** — derivative saturates near 0 and 1 |

### 2.2 Tanh (Hyperbolic Tangent)

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

```
  +1 ┤          ___________
     │       ╱
   0 ┤─────╱───────────────── (zero-centred ✓)
     │  ╱
  −1 ┤╱
     └─────────────────────── x
    -5   -2   0   2   5
```

| Property | Value |
|----------|-------|
| Range | (−1, +1) |
| Zero-centred | ✓ Yes — preferred over sigmoid for hidden layers |
| Problem | Still vanishes at extremes, but less severely |

### 2.3 ReLU (Rectified Linear Unit) ⭐ Most Common

$$
\text{ReLU}(x) = \max(0, x)
$$

```
  ↑ f(x)
  │          ╱
  │        ╱
  │      ╱
  │    ╱
  ├──╱────────────── x
  │╱  (flat at 0 for x < 0)
```

| Property | Value |
|----------|-------|
| Range | [0, ∞) |
| Derivative | 0 if x < 0, 1 if x > 0 |
| Speed | Very fast — simple comparison |
| Problem | **Dead ReLU** — neurons that output 0 stay at 0 forever |

### 2.4 Leaky ReLU

$$
f(x) = \max(0.01x,\; x)
$$

```
  ↑ f(x)
  │          ╱
  │        ╱
  ├──────╱──────── x
  │  ╱ (tiny negative slope = 0.01)
  │╱
```

Fixes dead ReLU by allowing a small gradient when x < 0. The 0.01 slope is a hyperparameter.

### 2.5 ELU (Exponential Linear Unit)

$$
f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{otherwise} \end{cases}
$$

- Smoother than Leaky ReLU for x < 0.
- No dead neurons.
- Slightly more computation than ReLU.

### 2.6 Which Activation to Use?

| Situation | Recommended |
|-----------|-------------|
| Hidden layers (general) | **ReLU** (fast, simple) |
| Hidden layers (avoid dead neurons) | **Leaky ReLU** or **ELU** |
| Output — binary classification | **Sigmoid** |
| Output — multiclass classification | **Softmax** |
| Output — regression | **Linear** (no activation) |
| Hidden layers needing zero-centred | **Tanh** |

---

## 3. Forward Propagation

Forward propagation is just **computing the output** by passing data through the network layer by layer.

```
Input x ──► Layer 1 ──► Layer 2 ──► … ──► Output ŷ
              ↓              ↓
         z¹ = W¹x + b¹   z² = W²a¹ + b²
         a¹ = σ(z¹)      a² = σ(z²)
```

Each layer does two things:
1. **Linear combination:** `z = Wa + b`
2. **Activation:** `a = σ(z)`

The output `a` of one layer becomes the input to the next.

---

## 4. Loss Functions

The loss function quantifies **how wrong** the model's predictions are. Different tasks need different loss functions.

### 4.1 Loss Functions for Regression

#### Mean Squared Error (MSE) ⭐ Default for Regression

$$
\text{MSE} = \frac{1}{N} \sum_{i=0}^{N-1} (y_i - \hat{y}_i)^2
$$

```
  J(w) ↑      ╱╲
       │     ╱  ╲
       │    ╱    ╲
       │   ╱      ╲___╱
       │  ╱           ╲
       └──────────────────► w
              ↑
        global minimum (only one — convex!)
```

- Differentiable everywhere → gradient descent works smoothly.
- Single global minimum → guaranteed to converge.
- **Disadvantage:** Sensitive to outliers (squaring amplifies large errors).

#### Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{N} \sum_{i=0}^{N-1} |y_i - \hat{y}_i|
$$

- More robust to outliers than MSE.
- **Disadvantage:** Derivative is not defined at 0; uses subgradient in practice.

#### Huber Loss — Best of Both Worlds

$$
L_\delta = \begin{cases} (y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ 2\delta|y - \hat{y}| - \delta^2 & \text{otherwise} \end{cases}
$$

Acts like MSE for small errors (smooth) and like MAE for large errors (robust). `δ` is a hyperparameter you tune.

### 4.2 Loss Functions for Classification

#### Binary Cross-Entropy (BCE)

$$
L = -\big[y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y})\big]
$$

Simplified form:

$$
L = \begin{cases} -\log(\hat{y}) & \text{if } y = 1 \\ -\log(1 - \hat{y}) & \text{if } y = 0 \end{cases}
$$

Used with **sigmoid** output for binary (yes/no) classification. The `log` makes the penalty shoot to infinity when the model is confidently *wrong*.

#### Categorical Cross-Entropy (CCE)

$$
L(x, y) = -\sum_{j=1}^{C} y_{ij} \cdot \ln(\hat{y}_{ij})
$$

Used with **softmax** output for multiclass classification (dog / cat / bird).

**One-hot encoding** is used for the true label:

```
True label: dog          One-hot: y = [1, 0, 0]
Model logits:            [2.0, 1.0, 0.5]

Softmax step:
  e^2.0  = 7.39          P(dog)  = 7.39 / 11.76 = 0.63  ← highest ✓
  e^1.0  = 2.72          P(cat)  = 2.72 / 11.76 = 0.23
  e^0.5  = 1.65          P(bird) = 1.65 / 11.76 = 0.14
  Σ = 11.76
```

**Quick reference — which loss + which activation:**

| Task | Output Activation | Loss Function |
|------|-------------------|---------------|
| Binary classification | Sigmoid | BCE |
| Multiclass classification | Softmax | CCE |
| Linear regression | Linear (none) | MSE, MAE, or Huber |

---

## 5. Gradient Descent — How the Model Actually Learns

### 5.1 The Core Idea

Imagine you're on a hilly landscape and you want to reach the lowest valley (minimum cost). You can't see the whole map — you can only feel the **slope** under your feet right now. So you take a small step **downhill**. Repeat until you're at the bottom.

```
  Cost ↑     ╱╲
       │    ╱  ╲
       │   ╱    ╲
       │  ╱      ╲
       │ ╱        ╲___╱  ← global minimum
       │╱              
       └──────────────── w
        ↑
   Start here (random w)
   Step ← ← ← downhill
```

### 5.2 The Weight Update Formula

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{dL}{dw_{\text{old}}}
$$

| Symbol | Meaning |
|--------|---------|
| `η` (eta) | **Learning rate** — size of each step. Too big = overshoot. Too small = takes forever. |
| `dL/dw` | **Gradient** (slope) of the loss with respect to weight `w` |
| The minus sign | We go *opposite* to the gradient (downhill) |

The same formula applies to biases:

$$
b_{\text{new}} = b_{\text{old}} - \eta \cdot \frac{dL}{db_{\text{old}}}
$$

### 5.3 Why the Negative Sign?

- If the slope (`dL/dw`) is **positive** → loss increases as `w` increases → we should **decrease** `w` → subtract a positive number ✓
- If the slope is **negative** → loss increases as `w` decreases → we should **increase** `w` → subtract a negative number = add ✓

---

## 6. Backpropagation — Computing Gradients Efficiently

### 6.1 The Problem

To update every weight, we need `dL/dw` for every weight in the network. For a weight deep inside the network, the loss depends on it through many intermediate calculations. **Backpropagation uses the chain rule** to compute these efficiently — starting from the output and working backwards.

### 6.2 The Chain Rule

If `L` depends on `z²`, which depends on `z¹`, which depends on `w¹`:

$$
\frac{dL}{dw^1} = \frac{dL}{dz^2} \cdot \frac{dz^2}{dz^1} \cdot \frac{dz^1}{dw^1}
$$

Each term is easy to compute individually. You multiply them together.

### 6.3 Step-by-Step for a Simple Network

```
  x ──► w¹ ──► z¹ ──► σ ──► a¹ ──► w² ──► z² ──► σ ──► ŷ ──► L (loss)
                                                          ↑
                                               Compare with y
```

**Forward pass** (left to right): compute `z¹ → a¹ → z² → ŷ → L`

**Backward pass** (right to left): compute gradients:

```
∂L/∂ŷ  → ∂L/∂z²  → ∂L/∂w²  (update w²)
                  → ∂L/∂a¹  → ∂L/∂z¹  → ∂L/∂w¹  (update w¹)
```

The key sensitivity calculation for one neuron in the output:

$$
\frac{\partial C}{\partial w^L} = a^{L-1} \cdot \sigma'(z^L) \cdot 2(a^L - y)
$$

Where:
- `a^{L-1}` = activation from previous layer (the input to this neuron)
- `σ'(z^L)` = derivative of the activation function
- `2(a^L − y)` = derivative of the squared-error cost

### 6.4 The Vanishing Gradient Problem

When backpropagating through many layers, each layer multiplies the gradient by the **derivative of the activation function**. For sigmoid, that derivative maxes out at **0.25**:

```
Gradient after 5 layers ≈ 0.25 × 0.25 × 0.25 × 0.25 × 0.25 = 0.001
Gradient after 10 layers ≈ 0.000001  ← essentially zero!
```

The network **stops learning** in the early layers because gradients vanish.

**Solutions:**
- Use **ReLU** (derivative = 1 for positive inputs — no shrinking)
- Use **LSTM** in recurrent networks
- Use **residual connections** (skip connections) in deep networks

---

## 7. Optimisers — Smarter Ways to Descend

### 7.1 Batch Gradient Descent (Vanilla)

Compute the gradient over the **entire dataset**, then update weights once.

```
  pass all 1M data points → calculate gradient → update weights → repeat
```

- **Pro:** Smooth, accurate gradient.
- **Con:** Extremely slow. Uses all RAM at once.

### 7.2 Stochastic Gradient Descent (SGD)

Update weights after **every single training example**.

```
  1st example → update → 2nd example → update → … → 1M examples → (1 epoch done)
```

- **Pro:** Very fast updates, low memory.
- **Con:** Noisy — the path zigzags wildly.

### 7.3 Mini-Batch SGD ⭐ Standard in Practice

Split the dataset into **batches** (e.g. 32 or 64 examples). Compute gradient per batch, update per batch.

```
  Epoch 1:  [Batch 1 → update] [Batch 2 → update] … [Batch N → update]
  Epoch 2:  repeat …
```

**Epoch** = one full pass through the entire dataset.  
If you have 1M data points and batch size = 1000, you do **1000 iterations per epoch**.

- Balances speed and smoothness.
- `Noise: SGD > Mini-Batch > Batch GD`

### 7.4 SGD with Momentum

Momentum smooths out the zigzag by keeping a **running average** of past gradients (Exponentially Weighted Average — EWA):

$$
v_t = \beta \cdot v_{t-1} + (1 - \beta) \cdot \frac{\partial L}{\partial w}
$$

$$
w_t = w_{t-1} - \eta \cdot v_t
$$

- `β` (smoothing factor) is typically 0.9. Higher β = more smoothing, less weight on new data.
- EWA averages out sharp spikes and noise, maintaining direction towards the global minimum.

### 7.5 Adagrad — Adaptive Learning Rate

The idea: learning rate should be **different for each parameter**, based on how much that parameter has been updated before.

$$
\alpha_t = \sum_{i=1}^{t} \left(\frac{\partial L}{\partial w_i}\right)^2
$$

$$
w_{t+1} = w_t - \frac{\eta}{\sqrt{\alpha_t + \epsilon}} \cdot \frac{\partial L}{\partial w_t}
$$

- Parameters updated frequently → `αₜ` grows large → effective learning rate shrinks.
- Parameters updated rarely → effective learning rate stays high.
- `ε` is a tiny number (e.g. 10⁻⁸) to avoid division by zero.
- **Problem:** `αₜ` only grows → learning rate eventually becomes too small → training stops.

### 7.6 Adadelta — Fixes Adagrad's Decay

Instead of accumulating **all** past squared gradients, use an **exponentially decaying window**:

$$
\eta' = \frac{\eta}{\sqrt{S_{dw} + \epsilon}}
$$

$$
S_{dw_t} = \beta \cdot S_{dw_{t-1}} + (1 - \beta) \cdot \left(\frac{\partial L}{\partial w_{t-1}}\right)^2
$$

Now with `β` and `(1−β)` like in SGD with momentum, the `∂L/∂w` can be controlled — the denominator no longer only grows.

### 7.7 RMSProp — Similar Fix, Simpler

$$
\eta' = \frac{\eta}{\sqrt{S_{dw} + \epsilon}}
$$

Same exponential decay idea as Adadelta. Commonly used in practice.

### 7.8 Adam — The Default ⭐ Use This Unless You Have a Reason Not To

**Adam = Momentum + RMSProp** (Adaptive learning rate + smoothing).

$$
w_t = w_{t-1} - \eta' \odot v_{dw}
$$

Where:
- `η'` = adaptive learning rate (from RMSProp)
- `v_dw` = smoothed gradient (from Momentum)

$$
v_{dw_t} = \beta \cdot v_{dw_{t-1}} + (1 - \beta) \cdot \frac{\partial L}{\partial w_t}
$$

Two benefits combined:
- **Smoothing** (momentum) → reduces noise / zigzag
- **Learning rate adaptive** → each parameter learns at its own pace

### 7.9 Optimiser Comparison at a Glance

```
  SGD ─────────────► Noisy but works
  Mini-Batch SGD ──► Practical baseline
  + Momentum ──────► Smooths the path
  + Adagrad ───────► Adaptive per-parameter (but decays too much)
  + Adadelta/RMSProp ► Fixes the decay
  + Adam ──────────► Momentum + RMSProp = best of both  ⭐
```

---

## 8. Feature Scaling — Don't Forget This Step

Before training, **normalise or standardise** your input features so they all live on a similar scale.

**Why?** If one feature ranges 0–1 and another ranges 0–10,000, gradient descent will be dominated by the large-scale feature — convergence becomes very slow or fails entirely.

| Method | Formula | When to use |
|--------|---------|-------------|
| Min-Max Scaling | `x_scaled = (x − min) / (max − min)` | Output in [0, 1] |
| Standardisation | `x_scaled = (x − mean) / std` | Output centred at 0 |

**Critical rule:** `fit()` only on **training data**. `transform()` on both training and test data. This prevents **data leakage** — the model must not "see" test data during training.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # transform only on test
```

---

## 9. Code — Building and Training a Neural Network

### PyTorch — Spam Classifier

```python
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- Load & preprocess ---
msgs = pd.read_csv('trained_data', sep='\t', names=["label", "msg"])
# ... (tokenise, vectorise using TF-IDF or BoW) ...

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Define model ---
class SpamNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, 1),         nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model   = SpamNet(X_train.shape[1])
loss_fn = nn.BCELoss()                          # Binary Cross-Entropy
optim   = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
X_tr = torch.FloatTensor(X_train)
y_tr = torch.FloatTensor(y_train).unsqueeze(1)

for epoch in range(100):
    pred = model(X_tr)
    loss = loss_fn(pred, y_tr)
    optim.zero_grad()
    loss.backward()                             # backpropagation
    optim.step()                                # weight update

# --- Evaluate ---
with torch.no_grad():
    y_pred = (model(torch.FloatTensor(X_test)) > 0.5).numpy().flatten()
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### TensorFlow / Keras — Customer Churn Classifier

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, ReLU

# --- Load data ---
dataset = pd.read_csv("churn.csv")
dataset.head()

# --- Feature engineering ---
X_train = pd.iloc[:, 3:13]
y_train = pd.iloc[:, 13]

# One-hot encode categorical columns (geography, gender)
geography = pd.get_dummies(X_train['Geography'], drop_first=True)
gender    = pd.get_dummies(X_train['Gender'],    drop_first=True)
X_train   = pd.concat([X_train.iloc[:, 3:11], geography, gender], axis=1)

# --- Split & scale ---
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=0)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- Build ANN ---
classifier = Sequential()
classifier.add(Dense(units=11, activation='relu', input_dim=X_train.shape[1]))
classifier.add(Dense(units=7,  activation='relu'))
classifier.add(Dense(units=6,  activation='relu'))
classifier.add(Dense(units=1,  activation='sigmoid'))     # binary output

# --- Compile & Train ---
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = classifier.fit(
    X_train, y_train,
    epochs=100,
    batch_size=10,
    validation_split=0.33
)

# --- Predict & Evaluate ---
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).flatten()
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

---

## 10. Summary

| Concept | One-Line Summary |
|---------|-----------------|
| Perceptron | Single neuron: weighted sum + activation |
| Activation Functions | Add non-linearity; ReLU is the default workhorse |
| Forward Propagation | Pass input through layers left-to-right to get prediction |
| Loss Functions | MSE/MAE/Huber for regression; BCE/CCE for classification |
| Gradient Descent | Walk downhill on the cost landscape using the slope |
| Backpropagation | Chain rule applied backwards to compute every gradient |
| Vanishing Gradient | Deep sigmoid networks stop learning; ReLU fixes it |
| Optimisers | Adam = momentum + adaptive LR; use it by default |
| Feature Scaling | Normalise inputs so all features contribute equally |

**Next up → `03_NLP.md`** — Text preprocessing, vectorisation (BoW, TF-IDF), and similarity metrics.  
**Then → `04_Word_Embeddings.md`** — Word2Vec, CBOW, Skip-gram, and the neural network behind them.

# 07 — Interview Q&A & Quick Reference Cheat Sheet

> **What this file is:** A fast-lookup reference. Use it to revise before interviews or to remind yourself which file covers what.

---

## 1. Common Interview Questions — With Answers

### Q1: What is the difference between Training, Validation, and Test sets?

```
  Full Dataset
  ├── Training Set    (70–80%)
  │     └── The model LEARNS from this data.
  │         Weights and biases are updated here.
  │
  ├── Validation Set  (10–15%)
  │     └── Used to TUNE hyperparameters.
  │         (learning rate, epochs, batch size, dropout rate)
  │         Model does NOT learn from this — just evaluated.
  │
  └── Test Set        (10–15%)
        └── Used ONLY for final accuracy reporting.
            Model has NEVER seen this data at any point.
            This is your "honest" performance number.
```

**Why not just train and test?** Without validation, you have no way to tune hyperparameters without accidentally fitting to the test set. That would make your test accuracy artificially optimistic.

---

### Q2: Why use Random Forest instead of a single Decision Tree?

| | Decision Tree | Random Forest |
|---|---|---|
| Bias | High | Low |
| Variance | High | Low |
| Overfitting | Prone | Resistant |
| How | One tree memorises training data | Ensemble of many trees, each trained on a random subset |

**The bias-variance tradeoff:**
- A single decision tree has **high variance** — it memorises noise in the training data and performs poorly on new data.
- A random forest **averages many trees**, reducing variance while keeping bias low.
- This is why Random Forest almost always outperforms a single Decision Tree.

---

### Q3: Why does Word2Vec matter? Why not just use One-Hot Encoding?

| | One-Hot Encoding | Word2Vec Embeddings |
|---|---|---|
| Dimensionality | Equal to vocabulary size (50,000+) | 50–300 dimensions |
| Semantic info | None ("king" and "table" are equally distant) | Captures meaning ("king" is close to "queen") |
| Memory | Huge sparse matrices | Small dense matrices |
| Analogies | Impossible | `king − man + woman ≈ queen` works! |

Word2Vec learns that words used in **similar contexts** have **similar meanings** — this is the **distributional hypothesis**.

---

### Q4: How do you check if data follows a Normal (Gaussian) distribution?

**Method 1: Q-Q Plot (Quantile-Quantile)**
- Plot your data's quantiles against theoretical normal quantiles.
- If points fall on a **straight diagonal line** → approximately normal.
- Deviations at the tails indicate heavy tails or skew.

**Method 2: Statistical Tests**
- **Shapiro-Wilk test** (best for small samples, n < 5000)
- **D'Agostino-Pearson test** (good for larger samples)
- If p-value > 0.05 → fail to reject normality (data is approximately normal).

**Key visual clue:** Normal distribution has Mean = Median = Mode.

---

### Q5: How do you convert a Pareto (skewed) distribution into a Normal distribution?

**Answer: Box-Cox Transformation**

The Box-Cox transformation is a family of power transforms parameterised by λ:

$$
y^{(\lambda)} = \begin{cases} \frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \ln(y) & \text{if } \lambda = 0 \end{cases}
$$

- The optimal λ is found by maximum likelihood estimation.
- λ = 1 → no transform; λ = 0 → log transform; other values → power transform.
- Works only on **positive** data (y > 0).

```python
from scipy.stats import boxcox

y_transformed, lambda_opt = boxcox(y)     # y must be > 0
print(f"Optimal lambda: {lambda_opt:.3f}")
```

---

### Q6: What is the difference between fit_transform() and transform()?

| Method | What it does | When to use |
|---|---|---|
| `fit()` | Calculates parameters (mean, std, min, max) from the data | — |
| `transform()` | Applies the scaling using already-calculated parameters | Test data |
| `fit_transform()` | Does both: calculates AND applies | Training data only |

**Why this matters:** If you `fit_transform()` on test data, the scaler learns the test set's statistics — the model has indirectly "seen" test data. This is **data leakage** and inflates your accuracy.

---

## 2. The Deep Learning Roadmap — Where Everything Fits

Your notes cover a progression through increasingly powerful sequence models:

```
  ① Standard Neural Network (ANN)
       │  Works for fixed-size inputs (tabular, images via CNN)
       ▼
  ② Recurrent Neural Network (RNN)
       │  Adds memory via hidden state loop
       │  Problem: vanishing gradient on long sequences
       ▼
  ③ LSTM RNN
       │  Adds cell state + 3 gates → solves vanishing gradient
       │  Can remember long-range dependencies
       ▼
  ④ GRU (Gated Recurrent Unit)
       │  Simplified LSTM (2 gates instead of 3)
       │  Fewer parameters, similar performance
       ▼
  ⑤ Bidirectional LSTM
       │  Processes sequence forwards AND backwards
       │  Better context on both sides of each word
       ▼
  ⑥ Encoder-Decoder
       │  Two networks: encoder compresses input, decoder generates output
       │  Used for translation, summarisation
       ▼
  ⑦ Transformers
       │  Replaces recurrence with self-attention
       │  Processes entire sequence in parallel → much faster
       ▼
  ⑧ BERT / GPT
       │  Transformers pre-trained on massive text corpora
       │  Fine-tuned for specific tasks
       │  State of the art for almost everything in NLP
```

---

## 3. Black Box vs White Box — Model Classification

```
  WHITE BOX (Interpretable)          BLACK BOX (Not interpretable)
  ─────────────────────────          ──────────────────────────────
  • Linear Regression                • Neural Networks (ANN)
  • Logistic Regression              • Convolutional NN (CNN)
  • Decision Tree                    • Recurrent NN (RNN / LSTM)
                                     • Random Forest
                                     • XGBoost
                                     • Transformers / BERT / GPT
```

**Explainable AI (XAI)** is the active research area trying to make black-box models interpretable — techniques like LIME, SHAP, and attention visualisation.

---

## 4. Loss Function + Activation Quick Reference

```
  Task                    Output Activation    Loss Function
  ─────────────────────   ──────────────────   ─────────────
  Regression              Linear (none)        MSE / MAE / Huber
  Binary Classification   Sigmoid              Binary Cross-Entropy
  Multiclass              Softmax              Categorical Cross-Entropy
```

---

## 5. Optimiser Quick Reference

```
  SGD          → Basic. Noisy. Slow.
  Mini-Batch   → Standard practice. Balance of speed + accuracy.
  + Momentum   → Smooths the zigzag path.
  Adagrad      → Adaptive LR per parameter. But LR decays too much.
  Adadelta     → Fixes Adagrad's decay with exponential window.
  RMSProp      → Similar fix, simpler.
  Adam         → Momentum + RMSProp. ⭐ USE THIS BY DEFAULT.
```

---

## 6. File-by-File Index — What's Where

| File | Topics Covered |
|------|---------------|
| `01_Foundations.md` | ML landscape, supervised/unsupervised, Bayes' Theorem (librarian/farmer), linear regression, cost function |
| `02_Deep_Learning.md` | Perceptron, activation functions (sigmoid/tanh/ReLU/Leaky ReLU/ELU), forward prop, loss functions, gradient descent, backpropagation, chain rule, vanishing gradient, all optimisers, feature scaling |
| `03_NLP.md` | NLP pipeline, tokenisation, stopwords, stemming vs lemmatisation, One-Hot Encoding, Bag of Words, TF-IDF, cosine similarity, Euclidean distance, N-grams |
| `04_Word_Embeddings.md` | Why OHE fails, Word2Vec, CBOW (full worked example), Skip-gram, embedding layers, code with gensim/PyTorch/Keras |
| `05_CNN.md` | Image representation (grayscale/RGB), convolution operation, filters (edge detection), stride, padding, output size formula, ReLU on feature maps, max pooling, flattening, full CNN pipeline, dropout, black vs white box |
| `06_RNN_LSTM.md` | RNN applications, architecture types (1-to-1 through many-to-many), forward prop in RNN, vanishing gradient in RNNs, LSTM cell state, all 3 gates with equations, worked context-switching example, statistical distributions |
| `07_Interview_and_Quick_Reference.md` | This file — Q&A, roadmap, cheat sheets |

---

## 7. Recommended Learning Order

```
  01_Foundations          ← Start here. Bayes + Linear Regression.
       ↓
  02_Deep_Learning        ← Neural nets, training mechanics, optimisers.
       ↓
  03_NLP                  ← Text preprocessing & vectorisation.
       ↓
  04_Word_Embeddings      ← Word2Vec, CBOW, Skip-gram.
       ↓
  05_CNN                  ← Images. (Can be done after 02 if you prefer.)
       ↓
  06_RNN_LSTM             ← Sequences. Needs 02 + 04.
       ↓
  07_Interview_Quick_Ref  ← Review & consolidate.
```

---

*All code examples use both PyTorch and TensorFlow/Keras where relevant. Formulas use standard ML notation. ASCII diagrams are designed to render in any monospace font.*

# AI Engineering Notes
> A structured, beginner-to-intermediate reference for IT engineers learning AI/ML/DL.  
> Built from hand-written course notes, expanded with theory, diagrams, and code examples in **both PyTorch and TensorFlow/Keras**.

---

## 📘 Learning Path (Read in This Order)

| # | File | What You'll Learn |
|---|------|-------------------|
| 1 | [`01_Foundations.md`](01_Foundations.md) | ML landscape, supervised vs unsupervised, **Bayes' Theorem** (full worked example), linear regression, cost functions |
| 2 | [`02_Deep_Learning.md`](02_Deep_Learning.md) | Perceptrons, activation functions, forward prop, **loss functions**, **gradient descent**, **backpropagation**, vanishing gradient, all major **optimisers** (SGD → Adam), feature scaling |
| 3 | [`03_NLP.md`](03_NLP.md) | NLP pipeline, tokenisation, stopwords, stemming vs lemmatisation, **One-Hot Encoding**, **Bag of Words**, **TF-IDF**, similarity metrics (cosine, Euclidean), N-grams |
| 4 | [`04_Word_Embeddings.md`](04_Word_Embeddings.md) | Why OHE fails at scale, **Word2Vec**, **CBOW** (full worked example with math), **Skip-gram**, embedding layers |
| 5 | [`05_CNN.md`](05_CNN.md) | Image representation, **convolution operation** (edge filters), stride, padding, **max pooling**, flattening, full CNN pipeline, dropout, black/white box models |
| 6 | [`06_RNN_LSTM.md`](06_RNN_LSTM.md) | RNN architecture types, why RNNs forget, **LSTM cell state**, **forget/input/output gates** (all equations + worked example), statistical distributions |
| 7 | [`07_Interview_and_Quick_Reference.md`](07_Interview_and_Quick_Reference.md) | Interview Q&A, DL roadmap (RNN→LSTM→Transformers→BERT), cheat sheets |

---

## 🗺️ The Big Picture — Where AI/ML/DL Sit

```
┌──────────────────────────────────────────────────────────┐
│                          AI                              │
│   ┌──────────────────────────────────────────────────┐   │
│   │              Machine Learning                    │   │
│   │   ┌──────────────────────────────────────────┐   │   │
│   │   │         Deep Learning                    │   │   │
│   │   │   Neural Networks, CNNs, RNNs,           │   │   │
│   │   │   LSTMs, Transformers, BERT, GPT         │   │   │
│   │   └──────────────────────────────────────────┘   │   │
│   │   Also: Random Forest, XGBoost, SVM, KNN         │   │
│   └──────────────────────────────────────────────────┘   │
│   Data Science: Stats + Programming + Domain Knowledge   │
└──────────────────────────────────────────────────────────┘
```

---

## 📊 All 50 Images — What's Where

```
── NLP & Vectorisation (Images 1–11) ──────────────────────────────
Image 01 ──► AI/ML/DL roadmap, tech stack           ──► README + 01_Foundations
Image 02 ──► NLP use-case, preprocessing flow       ──► 03_NLP
Image 03 ──► Stemming vs Lemmatisation              ──► 03_NLP
Image 04 ──► One-Hot Encoding worked example        ──► 03_NLP
Image 05 ──► Cosine similarity, Euclidean distance  ──► 03_NLP
Image 06 ──► N-grams, TF-IDF math                   ──► 03_NLP
Image 07 ──► TF-IDF continued, Word2Vec intro       ──► 03_NLP + 04_Word_Embeddings
Image 08 ──► Word2Vec neural network embedding      ──► 04_Word_Embeddings
Image 09 ──► Embedding layer (one-hot → dense)      ──► 04_Word_Embeddings
Image 10 ──► Word2Vec CBOW architecture             ──► 04_Word_Embeddings
Image 11 ──► Word2Vec Skip-gram                     ──► 04_Word_Embeddings

── Neural Network Fundamentals (Images 12–20) ─────────────────────
Image 12 ──► Spam classifier (perceptron)           ──► 02_Deep_Learning
Image 13 ──► Perceptron fundamentals                ──► 02_Deep_Learning
Image 14 ──► Forward & backward propagation         ──► 02_Deep_Learning
Image 15 ──► Activation functions (all 5)           ──► 02_Deep_Learning
Image 16 ──► Bayes' Theorem (librarian/farmer)      ──► 01_Foundations
Image 17 ──► Neural net image recognition intro     ──► 02_Deep_Learning
Image 18 ──► Gradient descent mechanics             ──► 02_Deep_Learning
Image 19 ──► Backprop chain rule                    ──► 02_Deep_Learning
Image 20 ──► Loss functions (MSE/MAE/Huber/BCE/CCE) ──► 02_Deep_Learning

── Optimisers & Scaling (Images 21–25) ────────────────────────────
Image 21 ──► Vanishing gradient problem             ──► 02_Deep_Learning
Image 22 ──► SGD, Mini-Batch SGD                    ──► 02_Deep_Learning
Image 23 ──► SGD + Momentum, EWA                    ──► 02_Deep_Learning
Image 24 ──► Adagrad, Adadelta, RMSProp             ──► 02_Deep_Learning
Image 25 ──► Adam optimiser                         ──► 02_Deep_Learning

── CNN (Images 26–33) ─────────────────────────────────────────────
Image 26 ──► Feature scaling (normalise/standardise)──► 02_Deep_Learning
Image 27 ──► How images are numbers (grayscale/RGB) ──► 05_CNN
Image 28 ──► Convolution operation, filters         ──► 05_CNN
Image 29 ──► Edge detection filters (horiz/vert)    ──► 05_CNN
Image 30 ──► Stride, padding, output size formula   ──► 05_CNN
Image 31 ──► ReLU on feature maps                   ──► 05_CNN
Image 32 ──► Max pooling worked example             ──► 05_CNN
Image 33 ──► Flattening + full CNN pipeline         ──► 05_CNN

── RNN, LSTM & Wrap-Up (Images 34–50) ─────────────────────────────
Image 34 ──► Train/val/test splits, cross-validation──► 05_CNN
Image 35 ──► Overfitting & dropout                  ──► 05_CNN
Image 36 ──► Black box vs white box models          ──► 05_CNN
Image 37 ──► Why we need sequence models            ──► 06_RNN_LSTM
Image 38 ──► RNN architecture (the loop)            ──► 06_RNN_LSTM
Image 39 ──► RNN types (1-to-1 … many-to-many)     ──► 06_RNN_LSTM
Image 40 ──► RNN forward prop worked example        ──► 06_RNN_LSTM
Image 41 ──► Vanishing gradient in RNNs             ──► 06_RNN_LSTM
Image 42 ──► LSTM — why it was invented             ──► 06_RNN_LSTM
Image 43 ──► LSTM cell state (conveyor belt)        ──► 06_RNN_LSTM
Image 44 ──► Forget gate + Input gate               ──► 06_RNN_LSTM
Image 45 ──► Cell state update equation             ──► 06_RNN_LSTM
Image 46 ──► Output gate + full LSTM equations      ──► 06_RNN_LSTM
Image 47 ──► Statistical distributions (Normal/Log-Normal/Pareto)──► 06_RNN_LSTM
Image 48 ──► Box-Cox transformation                 ──► 07_Interview
Image 49 ──► Interview Q&A (splits, RF, Word2Vec)   ──► 07_Interview
Image 50 ──► DL roadmap (RNN→LSTM→…→BERT/GPT)      ──► 07_Interview
```

---

## ⚙️ Tech Stack Referenced

| Library | Used For |
|---------|----------|
| **NumPy** | Matrix operations, numerical computation |
| **Pandas** | Data loading, feature engineering |
| **Scikit-learn** | Train/test split, scaling, metrics, classical ML |
| **PyTorch** | Neural network definitions, custom training loops |
| **TensorFlow / Keras** | High-level model building, Sequential API |
| **NLTK** | Tokenisation, stopwords, stemming |
| **Gensim** | Pre-trained Word2Vec, embedding similarity |
| **Matplotlib** | Plotting, visualisation |

---

## 📝 Conventions Used Throughout

- **Mathematical formulas** use standard ML notation: `w` = weights, `b` = bias, `η` = learning rate, `σ()` = sigmoid
- **ASCII diagrams** render in any monospace font — designed for GitHub markdown
- **Code blocks** come in pairs where relevant: PyTorch version + TensorFlow/Keras version
- **⭐** marks the recommended default choice (e.g. ⭐ Adam optimiser, ⭐ ReLU activation)

---

*Source: Hand-written notes from AI/ML YouTube courses, expanded and structured for clarity.*

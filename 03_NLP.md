# 03 — Natural Language Processing (NLP)

> **Goal:** Convert human language into structured numerical representations that machine-learning models can learn from.  
> **Prerequisites:** Basic Python, comfort with lists/arrays, high-school-level math.

---

## Table of Contents

1. [What is NLP?](#1-what-is-nlp)
2. [The NLP Pipeline — End to End](#2-the-nlp-pipeline--end-to-end)
3. [Text Preprocessing 1 — Tokenisation, Stopwords, Stemming, Lemmatisation](#3-text-preprocessing-1)
4. [Text Preprocessing 2 — Converting Words to Vectors](#4-text-preprocessing-2--converting-words-to-vectors)
5. [One-Hot Encoding (OHE)](#5-one-hot-encoding-ohe)
6. [Bag of Words (BoW)](#6-bag-of-words-bow)
7. [TF-IDF](#7-tf-idf-term-frequency--inverse-document-frequency)
8. [Similarity Metrics — Euclidean Distance & Cosine Similarity](#8-similarity-metrics)
9. [N-grams — Capturing Semantic Context](#9-n-grams--capturing-semantic-context)
10. [Key Terminology Cheat Sheet](#10-key-terminology-cheat-sheet)
11. [Code Examples](#11-code-examples)

---

## 1. What is NLP?

**Natural Language Processing** is the branch of AI that gives computers the ability to understand, interpret, and generate human language — text or speech.

The core problem: humans communicate in **unstructured text**. Machines work with **numbers**. NLP is the bridge between the two.

### Common NLP Libraries

| Library | Best For |
|---------|----------|
| **NLTK** | Learning / education, classic algorithms |
| **spaCy** | Production pipelines, speed |
| **Hugging Face** | Pre-trained models (BERT, GPT) |
| **TextBlob** | Quick prototyping |
| **Gensim** | Word2Vec, topic models |

### NLP sits inside the Deep Learning stack like this:

```
┌─────────────────────────────────────────────┐
│              DEEP LEARNING                   │
│                                             │
│   ┌─────────┐  ┌──────────┐  ┌──────────┐  │
│   │   NLP   │  │ Computer │  │  Other   │  │
│   │  Text   │  │  Vision  │  │  Domains │  │
│   └────┬────┘  │ Img/Vid  │  └──────────┘  │
│        │       └──────────┘                 │
│        ▼                                    │
│   RNN ──► Encoder/Decoder                   │
│        ──► Transformers ──► LLMs            │
│                                             │
│   Key Models: BERT, GPT, Word2Vec           │
└─────────────────────────────────────────────┘
```

---

## 2. The NLP Pipeline — End to End

Before any model sees your text, it must travel through a **preprocessing pipeline**. This is the single most important concept to internalise before moving to model architecture.

```
┌──────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│          │    │  PRE-PROCESSING │    │  PRE-PROCESSING │    │  CONVERT WORDS   │
│ DATASET  │───►│       1         │───►│       2         │───►│  TO VECTORS      │
│  (raw    │    │                 │    │                 │    │                  │
│  text)   │    │ • Tokenisation  │    │ • Stemming      │    │ • Bag of Words   │
└──────────┘    │ • Lowercasing   │    │ • Lemmatisation │    │ • TF-IDF         │
                │ • Stopword      │    │                 │    │ • Word2Vec       │
                │   removal       │    │                 │    │                  │
                └─────────────────┘    └─────────────────┘    └──────────────────┘
```

### Use-Case Example: Email Spam Detector

```
┌───────────────┬──────────────┬────────────┐
│   Input (i)   │  Input (i2)  │   Output   │
├───────────────┼──────────────┼────────────┤
│  Email Body   │  Email Sub.  │ Spam / Ham │
├───────────────┼──────────────┼────────────┤
│ "You won 1M$$"│ "Millionaire"│   SPAM     │
└───────────────┴──────────────┴────────────┘
                                      │
                                      ▼  Tokenisation
                              [You, won, 1M, $$]
```

---

## 3. Text Preprocessing 1

### 3.1 Tokenisation

**Definition:** Breaking a sentence into individual words (tokens).

```
Input:   "Hey buddy, I want to go to your house."

Output:  ["Hey", "buddy", ",", "I", "want", "to", "go", "to", "your", "house", "."]
```

This is always **Step 1**. Every downstream process depends on having clean tokens.

---

### 3.2 Stopword Removal

**Definition:** Remove common English words that carry almost no meaning for classification tasks.

Common stopwords: `the, is, a, an, to, in, of, and, or, but, ...`

```
BEFORE:  ["Hey", "buddy", "I", "want", "to", "go", "to", "your", "house"]
                                        ^^^^      ^^^^  ^^^^

AFTER:   ["Hey", "buddy", "I", "want", "go", "your", "house"]
```

> **Why?** Stopwords appear in almost every document. They add noise and inflate vector dimensions without helping the model distinguish between classes.

---

### 3.3 Stemming

**Definition:** Reduce a word to its **root form** by chopping off suffixes using rules. Fast but crude — the result may not be a real word.

```
historically  ─┐
historian     ─┼──► "histor"    ⚠️  Not a real word!
history       ─┘

going  ─┐
gone   ─┼──► "go"
goes   ─┘

Pro:  Really fast (rule-based, no dictionary lookup)
Con:  The stem might not have a meaningful meaning
```

**Use cases for stemming:** Spam classification, search engines (speed > accuracy).

**Porter Stemmer** is the most widely used algorithm. It applies a fixed set of suffix-removal rules.

---

### 3.4 Lemmatisation

**Definition:** Reduce a word to its **meaningful root form (lemma)** by understanding grammar and context. Slower but always returns a valid dictionary word.

```
history      ─┐
historical   ─┼──► "history"    ✅  Real word, correct meaning
historically ─┘

Pro:  Accurate — always returns a valid word
Con:  Slower (needs a vocabulary / grammar lookup)
```

**Use cases for lemmatisation:** Summarisation, language translation, chatbots (accuracy > speed).

### Stemming vs Lemmatisation — Side by Side

```
Word          │ Stemming Output │ Lemmatisation Output
──────────────┼─────────────────┼─────────────────────
"running"     │   "run"         │   "run"
"historically"│   "histor"      │   "history"
"better"      │   "better"      │   "good"        ← lemma understands grammar!
"going"       │   "go"          │   "go"
"mice"        │   "mice"        │   "mouse"       ← lemma knows the base noun
```

---

## 4. Text Preprocessing 2 — Converting Words to Vectors

After tokenising, removing stopwords, and stemming/lemmatising, we have clean word tokens. But **models only understand numbers**. We must convert words → vectors (numerical arrays).

Three classical techniques:

```
                    ┌─────────────────────────┐
   Words ─────────► │  1. One-Hot Encoding    │
                    │  2. Bag of Words (BoW)  │
                    │  3. TF-IDF              │
                    └─────────────────────────┘
                              │
                              ▼
                      Numerical Vectors
                   (ready for ML models)
```

---

## 5. One-Hot Encoding (OHE)

### Concept

Each word in the **vocabulary** gets its own unique binary vector. Every position in the vector is `0` except the position corresponding to that word, which is `1`.

### Worked Example

```
Corpus:
  doc1 → "A man eat food"
  doc2 → "Cat eat food"
  doc3 → "People watch Krish YT"

Vocabulary (sorted, size = 9):
  Index:  1    2    3    4    5    6    7    8    9
  Word:   A   man  eat  food cat  people watch krish YT
```

One-hot vectors for each word:

```
  A      → [1, 0, 0, 0, 0, 0, 0, 0, 0]
  man    → [0, 1, 0, 0, 0, 0, 0, 0, 0]
  eat    → [0, 0, 1, 0, 0, 0, 0, 0, 0]
  food   → [0, 0, 0, 1, 0, 0, 0, 0, 0]
  cat    → [0, 0, 0, 0, 1, 0, 0, 0, 0]
  ...
```

A **document** is then a matrix stacking the one-hot vectors of its words.

### Advantages
- Simple to understand and implement
- Intuitive — each word has a unique identity

### Disadvantages

```
┌────────────────────────────────────────────────────────┐
│  ① SPARSE MATRIX                                       │
│     Vocabulary of 50,000 words → each vector has       │
│     49,999 zeros and one 1. Wastes memory & compute.   │
│                                                        │
│  ② OUT OF VOCABULARY (OOV)                             │
│     If a new word appears at inference time that       │
│     wasn't in training, the model has no vector for it.│
│                                                        │
│  ③ INPUT SIZE IS NOT FIXED                             │
│     Different documents have different lengths →       │
│     matrix dimensions change.                          │
│                                                        │
│  ④ NO SEMANTIC MEANING CAPTURED                        │
│     "cat" and "dog" are just as "far apart" as         │
│     "cat" and "spaceship". Relationships between       │
│     words are completely lost.                         │
└────────────────────────────────────────────────────────┘
```

---

## 6. Bag of Words (BoW)

### Concept

Instead of one-hot encoding each word independently, BoW represents an **entire document** as a single vector. Each dimension = a word in the vocabulary, and the value = **how many times** that word appears in the document.

> Word **order is discarded**. Hence the name "bag" — you just throw all words in a bag and count them.

### Worked Example

```
Documents (after stopword removal + lowercasing):
  D1 → "good boy"
  D2 → "good girl"
  D3 → "boy girl good"

Vocabulary:  [good, boy, girl]
              f1    f2   f3
```

Build the BoW matrix — each row is a document, each column is a word:

```
         good   boy   girl
         ────   ───   ────
  Doc 1 │  1  │  1  │  0  │
  Doc 2 │  1  │  0  │  1  │
  Doc 3 │  1  │  1  │  1  │
```

### How to Build It — Step by Step

```
Step 1: Remove meaningless words (stopwords) + lowercase
        D1 → "He is a good boy"   →   "good boy"
        D2 → "He is a good girl"  →   "good girl"
        D3 → "Boys & girls are good" →  "boy girl good"

Step 2: Extract vocabulary (unique words across all docs)
        Vocab = { good, boy, girl }

Step 3: Count occurrences per document → fill the matrix
```

### Advantages
- Simple, fast to compute
- Captures word frequency (unlike OHE which is binary)

### Disadvantages
- Still a **sparse matrix** for large vocabularies
- **No word order** — "dog bites man" and "man bites dog" are identical
- **No semantics** — cannot tell that "happy" and "joyful" are similar

---

## 7. TF-IDF (Term Frequency – Inverse Document Frequency)

### Why TF-IDF?

BoW treats every word equally. But some words (like "the", "is", "a") appear in **every** document and tell us nothing useful. Meanwhile, rare words like "quantum" or "detective" are highly informative for distinguishing documents.

**TF-IDF solves this** by down-weighting common words and up-weighting rare ones.

### The Two Components

#### TF — Term Frequency

> How often does this word appear in **this** document?

$$
\text{TF}(t, d) = \frac{\text{Number of times term } t \text{ appears in document } d}{\text{Total number of words in document } d}
$$

```
Example:  Doc = "the cat sat on the mat"   (6 words total)

  TF("the")  = 2 / 6 = 0.33    ← common word, high TF
  TF("cat")  = 1 / 6 = 0.17
  TF("sat")  = 1 / 6 = 0.17
```

#### IDF — Inverse Document Frequency

> How **rare** is this word across the **entire corpus**?

$$
\text{IDF}(t) = \ln\!\left(\frac{\text{Total number of documents } N}{\text{Number of documents containing term } t}\right)
$$

```
Corpus of 1000 documents:

  IDF("the")      = ln(1000 / 998) ≈ 0.002   ← appears almost everywhere → LOW
  IDF("quantum")  = ln(1000 / 3)   ≈ 5.81    ← appears in very few docs  → HIGH
```

> **Intuition:** If a word appears in every single document, it tells you nothing about what makes *this* document special. IDF penalises such words.

#### TF-IDF — The Final Score

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

```
Word         │  TF    │  IDF   │  TF-IDF
─────────────┼────────┼────────┼────────
"the"        │  0.33  │  0.002 │  0.0007   ← near zero, filtered out
"cat"        │  0.17  │  4.20  │  0.71     ← informative, kept
"quantum"    │  0.10  │  5.81  │  0.58     ← rare & present, kept
```

### Advantages over BoW
- Automatically down-weights common/uninformative words
- Highlights words that are **distinctive** to each document
- Used by search engines (e.g., Google's early ranking)

### Disadvantages
- Still produces **sparse vectors**
- Still **no semantic understanding** ("happy" ≠ "joyful" in TF-IDF space)
- Word order is lost

---

## 8. Similarity Metrics

Once words or documents are represented as vectors, we need ways to **measure how similar** two vectors are. Two dominant metrics in NLP:

```
                          ┌───────────────────┐
   Two Vectors A, B  ───► │  1. Euclidean     │  (straight-line distance)
                          │  2. Cosine Sim.   │  (angle between vectors)
                          └───────────────────┘
```

Imagine placing words in a 2D or 3D map. Similar words cluster together. We measure "closeness" using these metrics.

---

### 8.1 Euclidean Distance

**Definition:** The straight-line (shortest path) distance between two points in space. This is just the Pythagorean theorem generalised to *n* dimensions.

$$
d(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}
$$

For 2D vectors $A = (A_1, A_2)$ and $B = (B_1, B_2)$:

$$
d = \sqrt{(A_1 - B_1)^2 + (A_2 - B_2)^2}
$$

```
        A(3,4)
        •
        |  \
        |    \  ← Euclidean distance
        |      \     = √((3-1)² + (4-1)²)
        |        \   = √(4 + 9)
        |          \ = √13 ≈ 3.61
        •───────────•
     (0,0)       B(1,1)
```

**Limitation:** Sensitive to the **magnitude** (length) of vectors, not just their direction. Two documents with the same topic but different lengths will appear "far apart" even though they mean the same thing.

---

### 8.2 Cosine Similarity

**Definition:** Measures the **angle** between two vectors, ignoring their magnitude. This is the **most commonly used** metric for text embeddings.

$$
\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

Where:
- $A \cdot B$ = dot product (multiply corresponding elements, then sum)
- $\|A\|$ = magnitude of vector A = $\sqrt{\sum A_i^2}$
- $\|B\|$ = magnitude of vector B

### Interpreting the Score

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   cos(θ) =  1   →  vectors point SAME direction         │
│                     Words are SIMILAR                    │
│                                                         │
│   cos(θ) =  0   →  vectors are ORTHOGONAL (90°)         │
│                     Words are UNRELATED                  │
│                                                         │
│   cos(θ) = -1   →  vectors point OPPOSITE direction     │
│                     Words are OPPOSITES                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Visual — Cosine vs Euclidean

```
            A(A1,A2)
           /
          /  θ = 45°
         / ╱
        /╱─────────── B(B1,B2)
       O

  Cosine Similarity = cos(45°) ≈ 0.707
  Cosine Distance    = 1 - cos(θ) = 1 - 0.707 = 0.293
```

### Cosine Distance (derived metric)

$$
\text{Cosine Distance} = 1 - \cos(\theta)
$$

This converts similarity into a distance (higher = more different):

```
  Similarity = 1  →  Distance = 0   (perfect match)
  Similarity = 0  →  Distance = 1   (orthogonal / unrelated)
  Similarity = -1 →  Distance = 2   (opposite)
```

### Reference Table — Angle ↔ Sin ↔ Cos

```
  Angle (θ) │  sin(θ)  │  cos(θ)
  ──────────┼──────────┼─────────
     0°     │   0      │   1
    30°     │   0.5    │   0.866
    45°     │   0.707  │   0.707
    90°     │   1      │   0
   120°     │   0.866  │  -0.5
   180°     │   0      │  -1
```

### Why Cosine over Euclidean for NLP?

| Concern | Euclidean | Cosine |
|---------|-----------|--------|
| Affected by vector length? | ✅ Yes | ❌ No |
| Works well with sparse vectors? | ❌ Poor | ✅ Excellent |
| Captures semantic direction? | Partially | ✅ Yes |
| Standard in word embeddings? | Rarely | ✅ Always |

---

## 9. N-grams — Capturing Semantic Context

### The Problem with OHE and BoW

All the techniques above treat each word **independently**. They capture *what* words are present but not *how* words relate to each other. This means:

```
  "dog bites man"   →  BoW: {dog:1, bites:1, man:1}
  "man bites dog"   →  BoW: {dog:1, bites:1, man:1}   ← IDENTICAL! 😱
```

### Solution: N-grams

An **N-gram** is a contiguous sequence of N words. By treating sequences as single units, we capture word order and local context.

```
Sentence: "The quick brown fox jumps"

  Unigrams  (N=1): ["The", "quick", "brown", "fox", "jumps"]
  Bigrams   (N=2): ["The quick", "quick brown", "brown fox", "fox jumps"]
  Trigrams  (N=3): ["The quick brown", "quick brown fox", "brown fox jumps"]
```

### Why N-grams matter

- **Bigrams** capture common phrases: "New York", "machine learning"
- **Trigrams** capture more context: "out of the"
- Used in language models to **predict the next word**

### Limitation

As N grows, the number of possible N-grams explodes (combinatorial blowup), making the feature space very sparse. This is why modern models (Word2Vec, Transformers) have largely replaced N-gram models — but N-grams remain useful for simple baselines and feature engineering.

---

## 10. Key Terminology Cheat Sheet

```
┌─────────────┬────────────────────────────────────────────────────┐
│  Term       │  Definition                                        │
├─────────────┼────────────────────────────────────────────────────┤
│  CORPUS     │  The entire collection of text documents           │
│             │  (e.g., all emails, all Wikipedia articles)        │
│             │                                                    │
│  DOCUMENT   │  A single unit of text (one email, one sentence,   │
│             │  one article)                                      │
│             │                                                    │
│  VOCABULARY │  The set of all unique words across the corpus     │
│             │  Vocab size directly affects vector dimensions      │
│             │                                                    │
│  TOKEN      │  A single unit produced by tokenisation            │
│             │  (usually a word, sometimes a subword)             │
│             │                                                    │
│  EMBEDDING  │  A dense, low-dimensional vector representation    │
│             │  of a word that captures semantic meaning           │
│             │  (learned, not hand-crafted)                       │
└─────────────┴────────────────────────────────────────────────────┘
```

### Sentiment Analysis — Quick Example

```
Task: Classify movie reviews as Positive or Negative

  Input (text)              │  Output (label)
  ──────────────────────────┼─────────────────
  "The food is good"        │  1  (Positive)
  "The food is bad"         │  0  (Negative)
  "Pizza is OK"             │  1  (Positive)
```

This is a classic **text classification** use case — exactly what BoW and TF-IDF are well-suited for as feature extractors before feeding into a classifier.

---

## 11. Code Examples

### 11.1 Full Preprocessing Pipeline — Python (NLTK)

```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ─── Step 1: Tokenisation ────────────────────────────────────────
text = "Hey buddy, I want to go to your house."
tokens = word_tokenize(text)
print("Tokenised:", tokens)
# Output: ['Hey', 'buddy', ',', 'I', 'want', 'to', 'go', 'to', 'your', 'house', '.']

# ─── Step 2: Lowercasing + Stopword Removal ──────────────────────
stop_words = set(stopwords.words('english'))
tokens_clean = [t.lower() for t in tokens if t.lower() not in stop_words and t.isalpha()]
print("After stopwords:", tokens_clean)
# Output: ['hey', 'buddy', 'want', 'go', 'house']

# ─── Step 3a: Stemming ────────────────────────────────────────────
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in tokens_clean]
print("Stemmed:", stemmed)
# Output: ['hey', 'buddhi', 'want', 'go', 'hous']   ← Note: "buddhi" is not a real word

# ─── Step 3b: Lemmatisation (better accuracy) ────────────────────
lemmatizer = WordNetLemmatizer()
lemmatised = [lemmatizer.lemmatize(w) for w in tokens_clean]
print("Lemmatised:", lemmatised)
# Output: ['hey', 'buddy', 'want', 'go', 'house']   ← All real words ✓
```

---

### 11.2 One-Hot Encoding — Python (scikit-learn)

```python
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Vocabulary
words = np.array([["A"], ["man"], ["eat"], ["food"], ["cat"]])

encoder = OneHotEncoder(sparse_output=False)
ohe_matrix = encoder.fit_transform(words)

print("Vocabulary:", encoder.categories_[0])
print("\nOne-Hot Matrix:")
print(ohe_matrix)

# Output:
# Vocabulary: ['A' 'cat' 'eat' 'food' 'man']
#
# One-Hot Matrix:
# [[1. 0. 0. 0. 0.]   ← "A"
#  [0. 0. 0. 0. 1.]   ← "man"
#  [0. 0. 1. 0. 0.]   ← "eat"
#  [0. 0. 0. 1. 0.]   ← "food"
#  [0. 1. 0. 0. 0.]]  ← "cat"
```

---

### 11.3 Bag of Words — Python (scikit-learn)

```python
from sklearn.feature_extraction.text import CountVectorizer

# Raw documents (already cleaned)
documents = [
    "good boy",
    "good girl",
    "boy girl good"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

print("Vocabulary:", vectorizer.get_feature_names_out())
print("\nBoW Matrix (dense):")
print(bow_matrix.toarray())

# Output:
# Vocabulary: ['boy' 'girl' 'good']
#
# BoW Matrix:
#        boy  girl  good
# Doc1 [  1     0     1 ]
# Doc2 [  0     1     1 ]
# Doc3 [  1     1     1 ]
```

---

### 11.4 TF-IDF — Python (scikit-learn)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "the cat sat on the mat",
    "the dog played in the park",
    "the quantum physicist solved the equation"
]

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(documents)

import pandas as pd
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out(),
    index=["Doc1", "Doc2", "Doc3"]
)
print(df.round(3))

# Output (key columns):
#        cat   dog  quantum  the   ...
# Doc1   0.45  0.0  0.0      0.28  ...   ← "cat" is distinctive here
# Doc2   0.0   0.45 0.0      0.28  ...   ← "dog" is distinctive here
# Doc3   0.0   0.0  0.52     0.25  ...   ← "quantum" is most distinctive
#
# Notice: "the" has a LOW score everywhere (common word, low IDF)
```

---

### 11.5 TF-IDF — PyTorch (manual implementation)

```python
import torch
import math

def compute_tfidf(documents: list[list[str]]) -> torch.Tensor:
    """
    Compute TF-IDF matrix from pre-tokenised documents.
    
    Args:
        documents: List of documents, each document is a list of words.
    Returns:
        TF-IDF matrix of shape (num_docs, vocab_size)
    """
    # ── Build vocabulary ──────────────────────────────────
    vocab = sorted(set(word for doc in documents for word in doc))
    word2idx = {w: i for i, w in enumerate(vocab)}
    N = len(documents)          # total number of documents
    V = len(vocab)              # vocabulary size

    # ── Compute TF matrix ────────────────────────────────
    tf = torch.zeros(N, V)
    for doc_idx, doc in enumerate(documents):
        for word in doc:
            tf[doc_idx, word2idx[word]] += 1
        tf[doc_idx] /= len(doc)          # normalise by doc length

    # ── Compute IDF vector ───────────────────────────────
    df = torch.zeros(V)                  # document frequency per word
    for doc in documents:
        for word in set(doc):            # count each word once per doc
            df[word2idx[word]] += 1

    idf = torch.log(torch.tensor(N, dtype=torch.float) / df)

    # ── TF-IDF = TF × IDF ───────────────────────────────
    tfidf = tf * idf.unsqueeze(0)        # broadcast IDF across all docs
    return tfidf, vocab

# ── Example ──────────────────────────────────────────────
docs = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "played", "in", "the", "park"],
]

matrix, vocab = compute_tfidf(docs)
for i, word in enumerate(vocab):
    print(f"  {word:>8}: Doc1={matrix[0,i]:.3f}  Doc2={matrix[1,i]:.3f}")
```

---

### 11.6 Cosine Similarity — Python (NumPy + PyTorch)

```python
import numpy as np
import torch

# ─── NumPy version ────────────────────────────────────────────────
def cosine_similarity_numpy(a: np.ndarray, b: np.ndarray) -> float:
    dot_product = np.dot(a, b)
    norm_a      = np.linalg.norm(a)
    norm_b      = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

# ─── PyTorch version ──────────────────────────────────────────────
def cosine_similarity_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()

# ─── Example: compare word vectors ───────────────────────────────
vec_cat = np.array([1.0, 2.0, 3.0])
vec_dog = np.array([1.2, 1.8, 3.1])      # similar to cat
vec_car = np.array([5.0, 0.1, -2.0])     # very different

print(f"cat vs dog: {cosine_similarity_numpy(vec_cat, vec_dog):.4f}")  # ≈ 0.99 (similar!)
print(f"cat vs car: {cosine_similarity_numpy(vec_cat, vec_car):.4f}")  # ≈ 0.31 (different)

# PyTorch version
t_cat = torch.tensor(vec_cat)
t_dog = torch.tensor(vec_dog)
print(f"cat vs dog (torch): {cosine_similarity_torch(t_cat, t_dog):.4f}")
```

---

### 11.7 Euclidean Distance — Python (NumPy + PyTorch)

```python
import numpy as np
import torch

# ─── NumPy version ────────────────────────────────────────────────
def euclidean_distance_numpy(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b)

# ─── PyTorch version ──────────────────────────────────────────────
def euclidean_distance_torch(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.norm(a - b).item()

# ─── Example ──────────────────────────────────────────────────────
a = np.array([3.0, 4.0])
b = np.array([1.0, 1.0])

print(f"Euclidean distance: {euclidean_distance_numpy(a, b):.4f}")
# = √((3-1)² + (4-1)²) = √(4+9) = √13 ≈ 3.6056

# PyTorch
print(f"Euclidean (torch):  {euclidean_distance_torch(torch.tensor(a), torch.tensor(b)):.4f}")
```

---

### 11.8 N-grams — Python (NLTK)

```python
from nltk.util import ngrams

sentence = "The quick brown fox jumps over the lazy dog"
tokens = sentence.split()

unigrams  = list(ngrams(tokens, 1))
bigrams   = list(ngrams(tokens, 2))
trigrams  = list(ngrams(tokens, 3))

print("Unigrams: ", unigrams)
# [('The',), ('quick',), ('brown',), ('fox',), ...]

print("Bigrams:  ", bigrams)
# [('The', 'quick'), ('quick', 'brown'), ('brown', 'fox'), ...]

print("Trigrams: ", trigrams)
# [('The', 'quick', 'brown'), ('quick', 'brown', 'fox'), ...]
```

---

*Next up: `04_LLMs.md` — Transformers, Attention Mechanism, BERT, GPT, and Fine-Tuning.*

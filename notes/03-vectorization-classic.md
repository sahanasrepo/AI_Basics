
# 03 — Classic Vectorization: One‑Hot Encoding & Bag‑of‑Words

> **From pages:** your corpus examples ("a man eat food", "cat eat food", "people watch match"), OHE table, disadvantages, BoW TF table after cleaning.

## One‑Hot Encoding (OHE)
**Vocabulary example**: `[a, man, cat, dog, eat, food]`.
- `a   → [1,0,0,0,0,0]`
- `man → [0,1,0,0,0,0]`
- `eat → [0,0,0,0,1,0]`
- `food→ [0,0,0,0,0,1]`

**Sentence matrix**: concatenate per‑token OHE vectors or sum to a BoW vector.

### Disadvantages you listed (expanded)
1. **Sparsity** — long vectors of zeros; memory‑heavy.
2. **Vocabulary growth** — dimension grows with data; OOV words unseen.
3. **No order** — ignores word positions.
4. **No semantics** — "good" vs. "great" are orthogonal.

## Bag‑of‑Words (BoW)
Counts per word per document (still ignores order).

**Your cleaned examples**
- `D1: he good boy`
- `D2: he good girl`
- `D3: boys girls good`

**Vocabulary**: `[good, boy, girl]`

| Doc | good | boy | girl |
|---:|:----:|:---:|:----:|
| D1 |  1   |  1  |  0   |
| D2 |  1   |  0  |  1   |
| D3 |  1   |  1  |  1   |

> Pre‑steps: lowercase, stopwords removal, (optional) lemmatization, and sometimes **n‑grams** to add local order.

### Practical guidance
- Start baselines with **TF‑IDF + unigrams/bigrams**.
- Save the fitted vectorizer to transform test/production data consistently.

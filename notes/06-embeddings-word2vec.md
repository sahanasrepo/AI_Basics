
# 06 — Word Embeddings & word2vec (CBOW/Skip‑gram)

> **From pages:** "word2vec", "CBOW", "Skip‑gram", continuous bag‑of‑words mention, cosine similarity of vectors, tables of feature dimensions, and examples like king/queen etc.

## 1) Why embeddings?
BoW/TF‑IDF vectors are sparse and do not capture semantics. **Embeddings** map words to **dense** vectors so that semantic similarity corresponds to **geometric** closeness.

```mermaid
flowchart LR
TEXT[Tokens] --> E[Embedding Layer\n(dimensionality k)] --> V[(k‑dim vectors)] --> S[Similarity\n(cosine)]
```

## 2) word2vec family
- **CBOW (Continuous Bag‑of‑Words)**: predict the **center** word from surrounding **context** words.
- **Skip‑gram**: predict **context** words from the **center** word.

```mermaid
flowchart TB
subgraph CBOW
C1[w_{t-2}] --- C2[w_{t-1}] --- C3[w_{t+1}] --- C4[w_{t+2}]
C1-->H((Hidden))
C2-->H
C3-->H
C4-->H
H-->Wc[w_t]
end

subgraph Skip-gram
Wc2[w_t] --> H2((Hidden)) --> O1[w_{t-2}]
Wc2 --> H2 --> O2[w_{t-1}]
Wc2 --> H2 --> O3[w_{t+1}]
Wc2 --> H2 --> O4[w_{t+2}]
end
```

**Training details**
- Window size \(m\) controls how many context words on each side.
- Large vocabularies use **negative sampling** or **hierarchical softmax**.
- After training, rows of the input matrix are the **word embeddings**.

## 3) Properties & analogies
- Cosine similarity groups similar words (e.g., *king* close to *queen*, *Paris* close to *France*).
- Linear relationships sometimes emerge: \(\mathrm{vec}(king)-\mathrm{vec}(man)+\mathrm{vec}(woman) \approx \mathrm{vec}(queen)\).

## 4) Worked toy example (conceptual)
Suppose we learn 6‑dimensional vectors for the vocabulary `[boy, girl, king, queen, apple, mango]`.
Your notes show small numeric ranges for dimensions per word. In practice, training assigns each word a k‑dim vector learned from context. **Cosine similarity** evaluates their closeness:

\[ \cos\_\mathrm{sim}(u,v) = \frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert} \]

- If \(\cos\_\mathrm{sim}(u,v)\to 1\): very similar
- If 0: unrelated
- If −1: opposite directions

## 5) From embeddings to models
- Replace BoW/TF‑IDF with embeddings in classifiers (e.g., average embeddings → logistic regression) or feed sequences into RNN/CNN/Transformer.
- For context‑dependent meaning ("bank" river vs. finance), use **contextual embeddings** (BERT‑like) rather than static word2vec.

## 6) Continuous Bag‑of‑Words vs. Bag‑of‑Words
- **BoW**: counts features per document, ignores order.
- **CBOW**: uses context window to predict center word; outputs dense vectors capturing **semantic regularities**.

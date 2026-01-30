
# 05 — TF‑IDF: Mathematics & Worked Examples

> **From pages:** Your multi‑page derivations of TF, IDF, and TF‑IDF with tables and numbers.

## 1) Definitions
- **Term Frequency (TF)** for term *t* in document *d*:
  \[
  \mathrm{TF}(t,d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}
  \]
  Variants include log‑scaled TF \(1+\log(\mathrm{count})\) and binary TF \(\mathbb{1}[t\in d]\).

- **Inverse Document Frequency (IDF)** across a corpus of *N* documents:
  \[
  \mathrm{IDF}(t) = \log\!\left(\frac{N}{1+\mathrm{df}(t)}\right) \quad \text{or} \quad \log\!\left(1+\frac{N}{\mathrm{df}(t)}\right)
  \]
  where \(\mathrm{df}(t)\) is the number of documents containing *t*. (The `+1` variant avoids division by zero.)

- **TF‑IDF** for term *t* in *d*:
  \[ \mathrm{TF\mbox{-}IDF}(t,d) = \mathrm{TF}(t,d) \times \mathrm{IDF}(t). \]

## 2) Tiny corpus (aligned with your notes)
Documents after cleaning:
```
D1: "good boy"
D2: "good girl"
D3: "boys girls good"
```
Vocabulary: `good, boy, girl`

**Raw counts**

| Doc | good | boy | girl | length |
|----:|:----:|:---:|:----:|:------:|
| D1  |  1   |  1  |  0   |   2    |
| D2  |  1   |  0  |  1   |   2    |
| D3  |  1   |  1  |  1   |   3    |

**TF**  \(= count/length\)

| Doc | TF(good) | TF(boy) | TF(girl) |
|----:|:--------:|:-------:|:--------:|
| D1  |  1/2     |  1/2    |   0      |
| D2  |  1/2     |   0     |  1/2     |
| D3  |  1/3     |  1/3    |  1/3     |

**Document frequency**: \(df(good)=3\), \(df(boy)=2\), \(df(girl)=2\); \(N=3\).

**IDF** (using \(\log(1+N/df)\)):
- \(\mathrm{IDF}(good) = \log(1+3/3) = \log(2)\)
- \(\mathrm{IDF}(boy)  = \log(1+3/2)\)
- \(\mathrm{IDF}(girl) = \log(1+3/2)\)

**TF‑IDF examples**
- D1: \(\mathrm{TF\mbox{-}IDF}(good,D1) = (1/2)\log 2\); \(\mathrm{TF\mbox{-}IDF}(boy,D1) = (1/2)\log(1+3/2)\)
- D2: analogous; D3: each term \( (1/3)\times \mathrm{IDF}(t) \)

> Your handwritten numbers follow exactly these steps with slightly different IDF variants; both are common. Libraries like scikit‑learn use `idf = log((1+n)/(1+df)) + 1`.

## 3) Intuition
- **TF** highlights terms central to a **document**.
- **IDF** down‑weights terms that occur in **many documents**.
- **TF‑IDF** balances document salience with corpus rarity.

## 4) Practical notes
- Normalize vectors (L2) before cosine similarity.
- Use **n‑grams** (uni+bi) to capture phrases like *not good*.
- Tune `min_df`/`max_df` to prune extremely rare/common terms.

# 05 — Convolutional Neural Networks (CNN)

> **Prerequisites:** `02_Deep_Learning.md` — neural networks, activation functions, backpropagation.  
> **What you'll know after:** How CNNs process images, what convolution/pooling/flattening actually do, and how to build a CNN in code.

---

## 1. How the Human Brain Sees (and How CNN Copies It)

When you look at an image of a dog, your brain doesn't see "dog" all at once. It processes it in stages:

```
Retina → Visual Cortex (edge detectors) → Higher regions (shapes, curves)
         → Even higher regions (parts like ears, nose) → "Dog!"
```

- Certain neurons in the visual cortex fire for **edges** in specific orientations.
- Deeper neurons combine edges into **curves and shapes**.
- Even deeper neurons recognise **object parts** (eyes, ears).
- The final layer combines parts into a **whole object**.

CNNs mirror this hierarchy exactly.

---

## 2. How Images Are Represented as Numbers

### 2.1 Grayscale Images

A grayscale image is a 2D grid of **pixels**, each with a brightness value from **0 (black) to 255 (white)**.

```
A 5×5 grayscale image:

  ┌─────────────────────┐
  │  0    0    0    0    0  │
  │  0    0    1    1    1  │
  │  0    0    0    1    1  │    ← Each cell = one pixel (0–255)
  │  0    0    0    1    1  │
  │  0    0    0    0    1  │
  └─────────────────────┘
  Single channel (1 layer)
```

### 2.2 Colour (RGB) Images

A colour image has **3 channels** (Red, Green, Blue), each a full 2D grid:

```
  RGB Image (5×5):
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │  Red    │  │  Green  │  │  Blue   │
  │  channel│  │  channel│  │  channel│
  │  5×5    │  │  5×5    │  │  5×5    │
  └─────────┘  └─────────┘  └─────────┘
  Shape: 5 × 5 × 3
```

Before feeding to a CNN, pixel values are typically **normalised to [0, 1]** by dividing by 255 (min-max scaling).

---

## 3. The Convolution Operation — The Core of CNN

### 3.1 What Is a Filter (Kernel)?

A filter is a **small grid of learned weights** (e.g. 3×3) that slides across the image. At each position, it computes a **dot product** with the patch of image it covers, producing one output number.

```
  6×6 Image                3×3 Filter          Output (4×4)
  ┌───────────┐            ┌───────┐           ┌─────────┐
  │ 1 1 0 0 0 0│           │ 1  2  1│           │  ?  ?  ? … │
  │ 1 1 0 0 0 0│    ×      │ 0  0  0│    =     │  ?  ?  ? … │
  │ 0 0 1 1 0 0│           │-1 -2 -1│           │  …       │
  │ 0 0 1 1 0 0│           └───────┘           └─────────┘
  │ 0 0 0 0 1 1│
  │ 0 0 0 0 1 1│
  └───────────┘
```

### 3.2 Worked Example — Horizontal Edge Filter

This 3×3 filter detects **horizontal edges**:

```
Horizontal edge filter:     Applied to image patch:

  [ 1   2   1 ]               [ 1  1  0 ]
  [ 0   0   0 ]     ×         [ 1  1  0 ]  =  (1×1)+(2×1)+(1×0)
  [-1  -2  -1 ]               [ 0  0  1 ]      (0×1)+(0×1)+(0×0)
                                                (-1×0)+(-2×0)+(-1×1)
                              = 1+2+0 + 0+0+0 + 0+0-1 = 2
```

Where there IS a horizontal edge in the image → the output value is **large**.  
Where there is NO edge → the output is **near zero**.

### 3.3 Vertical Edge Filter

```
Vertical edge filter:

  [ 1   0  -1 ]
  [ 2   0  -2 ]       Detects vertical transitions (left↔right contrast)
  [ 1   0  -1 ]
```

### 3.4 How the Filter Slides (Step by Step)

```
Position 1:  Cover top-left 3×3 → compute → output[0,0]
Position 2:  Slide 1 cell RIGHT  → compute → output[0,1]
…
Position N:  Slide to next ROW   → compute → output[1,0]
… continue until bottom-right corner
```

Each position produces **one number** in the output feature map. The collection of all output numbers = one **feature map**.

---

## 4. Stride and Padding

### 4.1 Stride

**Stride** = how many cells the filter moves at each step.

| Stride | Movement |
|--------|----------|
| 1 | Filter moves 1 cell at a time (default) |
| 2 | Filter moves 2 cells — output is smaller, faster |

### 4.2 Output Size Formula

$$
\text{Output size} = \frac{n + 2p - f}{s} + 1
$$

| Symbol | Meaning |
|--------|---------|
| `n` | Input size (one dimension) |
| `f` | Filter size |
| `p` | Padding |
| `s` | Stride |

**Example:** Input 6×6, filter 3×3, stride 1, no padding:

$$
\frac{6 + 0 - 3}{1} + 1 = 4 \quad → \quad \text{Output is } 4 \times 4
$$

### 4.3 Padding

Without padding, the output shrinks with every convolution layer. **Padding** adds zeros around the border to preserve spatial dimensions.

```
Original 3×3 image      With padding (p=1) → 5×5:

  ┌───────┐             ┌─────────────┐
  │ a b c │             │ 0  0  0  0  0│
  │ d e f │     →       │ 0  a  b  c  0│
  │ g h i │             │ 0  d  e  f  0│
  └───────┘             │ 0  g  h  i  0│
                        │ 0  0  0  0  0│
                        └─────────────┘
```

With p=1, f=3, s=1: output = (3 + 2 − 3)/1 + 1 = **3×3** — same as input ✓

---

## 5. After Convolution: ReLU Activation

After computing the feature map, **ReLU is applied element-wise** to every value:

```
Feature map (raw):        After ReLU:
  [ 2  -1   0 ]             [ 2   0   0 ]
  [-3   4  -2 ]     →       [ 0   4   0 ]
  [ 1  -5   3 ]             [ 1   0   3 ]
```

Negative values → 0. This adds non-linearity between conv layers.

---

## 6. Max Pooling — Shrink Without Losing Key Info

Max pooling **downsamples** the feature map by taking the **maximum value** in each non-overlapping window.

### Why?

- Reduces spatial dimensions → fewer parameters → faster training.
- Makes the network **location invariant** — a feature detected slightly shifted still triggers.

### Worked Example (2×2 pooling, stride 2):

```
  Input feature map (4×4):        Max Pool output (2×2):

  ┌─────────────┐                 ┌───────┐
  │ 1  2 │ 3  4 │                 │ 2   7 │   max(1,2,5,6)=6 ? 
  │ 5  6 │ 7  8 │     →           │ 6   8 │   Actually:
  ├─────────────┤                 └───────┘   Top-left 2×2: max(1,2,5,6) = 6
  │ 0  3 │ 5  1 │                             Top-right 2×2: max(3,4,7,8) = 8
  │ 4  2 │ 6  0 │                             Bot-left 2×2: max(0,3,4,2) = 4
  └─────────────┘                             Bot-right 2×2: max(5,1,6,0) = 6

  Correct output:
  ┌───────┐
  │ 6   8 │
  │ 4   6 │
  └───────┘
```

> There is also **Average Pooling** and **Min Pooling**, but max pooling is by far the most common.

---

## 7. Flattening — Bridge to the Dense Layers

After all convolution + pooling layers, the output is still a 2D (or 3D with channels) feature map. Dense layers expect a **1D vector**. So we **flatten**:

```
  Feature map (4×4):         Flattened:
  ┌───────────┐
  │ 5  7  3  2│
  │ 1  8  4  6│     →    [5, 7, 3, 2, 1, 8, 4, 6, 9, 0, 2, 1, 3, 5, 7, 4]
  │ 9  0  2  1│
  │ 3  5  7  4│
  └───────────┘
  
  Then feed into Dense layers → final classification output
```

---

## 8. Full CNN Pipeline

```
  Input Image (28×28×1)
         │
         ▼
  ┌─────────────┐
  │ Conv Layer 1 │  (e.g. 32 filters, 3×3) → 32 feature maps
  │ + ReLU       │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Max Pool     │  (2×2) → halves spatial dims
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Conv Layer 2 │  (e.g. 64 filters, 3×3) → 64 feature maps
  │ + ReLU       │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Max Pool     │  (2×2)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Flatten      │  → 1D vector
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Dense (ANN)  │  → fully connected layers
  │ + ReLU       │
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Output       │  (Softmax → class probabilities)
  └─────────────┘
```

**Parameter count example (MNIST digit recognition):**

| Layer | Shape | Parameters |
|-------|-------|------------|
| Input | 28×28 = 784 pixels | 0 |
| Hidden 1 | 16 neurons | 784×16 + 16 bias = 12,560 |
| Hidden 2 | 16 neurons | 16×16 + 16 = 272 |
| Output | 10 neurons (digits 0–9) | 16×10 + 10 = 170 |
| **Total** | | **~13,002** |

---

## 9. Training & Evaluation Concepts

### 9.1 Dataset Split

```
  Full Dataset
  ├── Training Set    (80%)  → model learns from this
  ├── Validation Set  (10%)  → tune hyperparameters (learning rate, epochs)
  └── Test Set        (10%)  → final accuracy (model NEVER sees this during training)
```

> **Key rule:** The test set must **never** influence any training decision — not even hyperparameter choices. That's what validation is for.

### 9.2 Cross-Validation

When data is limited, use **K-Fold Cross-Validation**:

```
  1000 samples, CV = 5:
    Fold 1: [Train: 800] [Val: 200]
    Fold 2: [Train: 800] [Val: 200]  ← different 200 each time
    …
  Average the 5 accuracy scores for a robust estimate.
```

### 9.3 Overfitting & Dropout

**Overfitting** = model memorises training data, performs terribly on new data.

```
  Training accuracy:  99%   ← looks great!
  Test accuracy:      60%   ← actually terrible
```

**Dropout** is the fix: randomly **deactivate** a fraction of neurons during training.

```
  If Dropout = 0.3 → 30% of neurons are turned off each batch
  This forces the network to learn redundant representations
  → generalises better to unseen data
```

### 9.4 Black Box vs White Box Models

| Model | Interpretability |
|-------|-----------------|
| Linear Regression, Decision Trees | **White box** — you can see why it decided what it did |
| Random Forest | **Black box** — ensemble of many trees, hard to interpret |
| ANN / CNN / RNN | **Black box** — millions of weights, no simple explanation |

**Explainable AI (XAI)** is the field trying to make black-box models interpretable.

---

## 10. Code — CNN Image Classifier

### TensorFlow / Keras — MNIST Digit Recognition

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU
)

# --- Load data ---
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0   # normalise + add channel dim
X_test  = X_test.reshape(-1, 28, 28, 1)  / 255.0

# --- Build CNN ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),                                   # prevent overfitting
    Dense(10, activation='softmax')                 # 10 digit classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# One-hot encode labels
y_train_oh = tf.keras.utils.to_categorical(y_train, 10)

model.fit(X_train, y_train_oh, epochs=10, batch_size=32, validation_split=0.2)

# --- Evaluate ---
y_pred = model.predict(X_test).argmax(axis=1)
from sklearn.metrics import accuracy_score
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

### PyTorch — CNN from Scratch

```python
import torch
import torch.nn as nn

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28×32
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 14×14×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → 14×14×64
            nn.ReLU(),
            nn.MaxPool2d(2),                              # → 7×7×64
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # → 7*7*64 = 3136
            nn.Linear(3136, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10)                             # 10 classes
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model   = DigitCNN()
loss_fn = nn.CrossEntropyLoss()                           # includes softmax
optim   = torch.optim.Adam(model.parameters(), lr=0.001)
```

---

## 11. Summary

| Concept | One-Line Summary |
|---------|-----------------|
| Convolution | Slide a small filter across the image; each position → one output value |
| Filters | Learned weight grids that detect edges, textures, patterns |
| Stride | Step size of the filter; larger = smaller output |
| Padding | Add zeros around borders to control output size |
| ReLU | Applied after conv; adds non-linearity |
| Max Pooling | Keep only the max in each window; shrinks spatial dims |
| Flattening | Reshape 2D feature maps into 1D for dense layers |
| CNN Pipeline | Conv → ReLU → Pool → … → Flatten → Dense → Output |
| Dropout | Randomly disable neurons during training to prevent overfitting |

**Next up → `06_RNN_LSTM.md`** — Recurrent Neural Networks, why they struggle with long sequences, and how LSTM fixes it.

# Building a Neural Network from First Principles  
### Comfort vs Temperature (PyTorch)

This repository documents my hands-on work while studying *The StatQuest Illustrated Guide to Neural Networks and AI* by Josh Starmer.  
Rather than treating neural networks as black boxes, the goal here is to **understand how simple networks behave step by step**, both before and after training.

The examples focus on intuition, visualization, and failure modes as much as success.

---

## 1. Problem Setup

We model a very simple idea:

> **Human comfort as a function of room temperature**

The assumptions are intentionally naive:
- Comfort starts increasing after ~15°C  
- Comfort peaks around ~25°C  
- Comfort decreases beyond that  

This shape is useful because it is **not linear**, but can still be approximated using a small neural network built from basic components.

---

## 2. Network Architecture

The neural network is deliberately minimal:

- Two linear transformations  
- Two ReLU activations  
- One linear output combination  

Conceptually:
<p align="center">
  <img src="https://github.com/user-attachments/assets/ea528a0d-839d-4a19-94bf-adf62d55009a" 
       alt="Two-ReLU neural network architecture" 
       width="600">
</p>


This setup produces **piecewise linear functions**, which makes it easy to see how:
- ReLU gates turn regions on and off
- Linear pieces combine to form more complex shapes

---

## 3. Chapter 1: Manual Weights and Shape Intuition

In the first phase, **all weights and biases are chosen manually**.

This stage is not about training — it’s about understanding.

By plotting:
- pre-activation values,
- ReLU outputs,
- and the final combined output,

it becomes clear that:
- ReLU does not “curve” the function,
- it **bends straight lines into new straight segments**,
- and combining multiple ReLUs creates structured shapes.

This removes much of the mystery behind how neural networks approximate functions.

---

## 4. Chapter 2: Training with Backpropagation

The same network is then trained using backpropagation on a small dataset:

Inputs:  [10, 17, 20, 22, 25, 30]
Labels:  [ 0,  4, 10, 14, 20, 15]

The model parameters are now trainable, and optimization is introduced.

---

## 5. Issue Encountered: Dead ReLU with SGD

When training with **Stochastic Gradient Descent (SGD)**, the model repeatedly failed.

### What happened?

- SGD updates pushed bias terms into a region where **all ReLU inputs were negative**
- ReLU outputs became exactly zero
- Gradients through ReLU vanished
- Learning stopped completely

This resulted in a **flat output** regardless of input.

This is a classic **dead-ReLU problem**.

---

## 6. Solution: Switching to Adam

Replacing SGD with **Adam** resolved the issue.

Why this worked:
- Adam adapts learning rates per parameter
- When gradients became small, Adam effectively increased step sizes
- Biases moved back into regions where ReLUs activated
- Gradients flowed again
- Training resumed normally

This demonstrated how **optimizer choice directly affects model behavior**, especially in small networks with ReLU activations.

---

## 7. Open Question: Why One Data Point Changed Everything

After training successfully, one more data point was added:

(35.0, 2.5)

This single addition caused a major change:
- The learned function became closer to a simple inclined line
- The original “peak then drop” shape was no longer captured

At this stage, the reason is **not fully understood**.

Possible contributing factors:
- Extremely small dataset
- Limited model capacity
- Strong influence of an edge (high-leverage) point
- Optimization favoring a simpler global fit

This behavior is expected to become clearer with later topics such as:
- model capacity,
- loss landscapes,
- and regularization.

For now, it is left as an **explicit open question**, not a bug to hide.

---

## 8. Key Takeaways So Far

- Neural networks are combinations of very simple parts  
- ReLU creates structure by turning regions on and off  
- Optimizers matter as much as architecture  
- Small datasets can produce unintuitive behavior  
- “Failure cases” are often the most educational  

This repository is meant to **document understanding in progress**, not present a polished solution.

---

## 9. Feedback and Discussion

If you have insights into:
- the shape collapse after adding the extra data point, or
- anything technically incorrect or incomplete here,

feel free to open an issue or reach out.  
This repo is part of a learning journey, not the final word.


## Chapter - 3

# End-to-End Learning Notes: Wine Dataset → PyTorch Model

This repository documents the implementation of a neural network trained on the Wine dataset, based on *The StatQuest Illustrated Guide to Neural Networks and AI* (Chapter 3). The primary goal of this project is to understand data flow, indexing semantics, and training behavior using PyTorch and PyTorch Lightning.

---

## 1. Dataset Overview

The project uses a simplified version of the standard Wine dataset to make the model's behavior easier to reason about.

* **Total Rows:** 178
* **Total Features:** 14 (Only **Alcohol** and **Color Intensity** are used as inputs)
* **Target:** 3 distinct wine classes

---

## 2. Model Architecture and Setup

The network is built using PyTorch Lightning (`L.LightningModule`) for structured training. It is intentionally kept small to focus on the learning dynamics rather than raw predictive power.

* **Input Layer:** 2 features (Alcohol, Color Intensity)
* **Hidden Layer:** 2 units (ReLU activation)
* **Output Layer:** 3 units (One for each wine class)
* **Loss Function:** Mean Squared Error (`nn.MSELoss(reduction='sum')`) with one-hot encoded targets
* **Optimizer:** Adam (Learning Rate = 0.001)

---

## 3. Roadblocks and Resolutions

Many apparent "model bugs" encountered during this build were actually data preparation bugs.

### Issue: `KeyError` and Shape Errors During Tensor Conversion

**The Problem:** Loading the dataset with `pd.read_table(url, sep=',', header=None)` caused pandas to assign integer column names (`0, 1, 2...`) while treating the actual header row as data. Attempting to isolate the labels (`df['Wine']`) or convert them to a tensor using `F.one_hot(torch.tensor(label_train))` resulted in `KeyError: 7`.

**The Root Cause:**
Pandas and PyTorch handle indexing in fundamentally different ways.

* **Pandas:** Treats integers as *index labels*. If you ask for `Series[7]`, it looks for the literal label `7`.
* **PyTorch:** Uses *positional indexing*. It iterates through elements at position `0, 1, 2...`.
Because the pandas Series was passed directly into PyTorch without standardizing the index, PyTorch's positional requests triggered pandas label lookups, resulting in a `KeyError`.

**The Fix:**
Remove pandas indexing semantics entirely before passing data to PyTorch.

```python
# Converts labels to 0-based integers and strips pandas indexing
label_train = label_train.factorize()[0]

```

> **Rule of Thumb:** pandas → encode / numpy → torch. Always strip pandas indexing semantics before passing data into a PyTorch model pipeline.

---

## 4. Understanding Accuracy Calculation

In PyTorch, accuracy is computed by bridging the gap between raw model logits and one-hot encoded labels using `torch.argmax`.

1. **Extract Predicted Class:** `preds = torch.argmax(outputs, dim=1)`
Converts raw scores (e.g., `[2.4, 0.8, -1.1]`) into a class index (e.g., `Class 0`).
2. **Extract Target Class:** `targets = torch.argmax(labels, dim=1)`
Converts one-hot encoded labels (e.g., `[0, 1, 0]`) back to a class index (e.g., `Class 1`).
3. **Compute Metric:** `acc = (preds == targets).float().mean()`
Compares the tensors to create a boolean mask (`[True, False, True]`), converts them to floats (`[1.0, 0.0, 1.0]`), and averages them to find the fraction of correct predictions.

*Note: In PyTorch Lightning, adding `prog_bar=True` to `self.log()` simply displays this metric in the console during training; it does not impact the gradient calculations.*

---

## 5. Training Behavior Across Epochs

The model was trained entirely from scratch (no checkpoint loading) across three separate runs to observe how epoch count affects test accuracy.

* **10 Epochs:** ~48.89% Accuracy
* **100 Epochs:** ~80.00% Accuracy
* **200 Epochs:** ~77.78% Accuracy

**Interpretation:**
The network learns the most obvious structural patterns in the data during the early epochs, leading to rapid improvement. However, more epochs do not linearly guarantee better performance. The slight dip at 200 epochs (from 80% to ~77.8%) is standard behavior for simple models on small feature sets, indicating a plateau where the model might be oscillating around a minimum or beginning to overfit.

---

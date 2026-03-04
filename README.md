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

## Neural Network Architecture

<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/300866f9-9d06-4e4c-b47a-d09e9dd36c7b"
    alt="Neural Network Architecture"
    width="600"
  />
</p>

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

## Training Metrics Visualization

### Accuracy vs Epoch

<p align="center">
  <img width="680" src="https://github.com/user-attachments/assets/134982ee-e23d-4606-83f6-bd72a5cde568" />
  <img width="680" src="https://github.com/user-attachments/assets/bb94c691-63cc-4ff2-af9d-d41e59d75a8f" />
  <img width="680" src="https://github.com/user-attachments/assets/ffd3a5f6-ddcc-4023-b771-920dde0e9db8" />
</p>

<p align="center">
  <em>Training accuracy progression across different epoch counts.</em>
</p>

---

### Loss vs Epoch

<p align="center">
  <img width="680" src="https://github.com/user-attachments/assets/96c7a54b-1085-4e00-b472-72641eb61602" />
  <img width="680" src="https://github.com/user-attachments/assets/d725d684-1dce-46d3-a834-7c8c33c4a11f" />
</p>

<p align="center">
  <em>Loss curves showing optimization behavior over increasing epochs.</em>
</p>

**Interpretation:**
The network learns the most obvious structural patterns in the data during the early epochs, leading to rapid improvement. However, more epochs do not linearly guarantee better performance. The slight dip at 200 epochs (from 80% to ~77.8%) is standard behavior for simple models on small feature sets, indicating a plateau where the model might be oscillating around a minimum or beginning to overfit.

---



# Neural Network Experiment: MSE Loss vs Cross Entropy Loss  
### Based on Chapter 3 & 4 of *The StatQuest Illustrated Guide to Neural Networks and AI* – Joshua Starmer

This experiment reproduces and extends the neural network examples from **The StatQuest Illustrated Guide to Neural Networks and AI** using **PyTorch**.  

The book demonstrates the concepts using the **Iris dataset**, but for this implementation the **Wine dataset** was used instead to explore the same ideas in a slightly different setting.

The goal was to observe how **Mean Squared Error (MSE) Loss** and **Cross Entropy Loss** behave during training for a **multi-class classification problem**.

---

# Dataset

The experiment uses the **Wine dataset**.

Dataset characteristics:

- **178 samples**
- **14 features**
- **3 wine classes**

To keep the neural network simple and aligned with the book's educational approach, only two features were selected:

- **Alcohol**
- **Color Intensity**

These two features are used as the **input layer of the neural network**.

The model predicts **one of three wine types**.

---

# Neural Network Setup

The neural network architecture was intentionally kept small to match the conceptual examples used in the book.

Input Layer: 2 neurons
Hidden Layer: small dense layer
Output Layer: 3 neurons (one per wine class)

The output layer produces **logits**, which are raw scores representing how strongly the model predicts each class.

---

# Chapter 3 Implementation — MSE Loss

In **Chapter 3**, the neural network is trained using **Mean Squared Error (MSE) Loss**.

MSE is traditionally used for **regression problems**, where the objective is to minimize the squared difference between predicted values and target values.

For classification, the target labels are typically converted into **one-hot encoded vectors**.

Example:

Wine Class 1 → [1, 0, 0]
Wine Class 2 → [0, 1, 0]
Wine Class 3 → [0, 0, 1]

The loss measures the squared difference between the predicted outputs and these target vectors.

---

## Training Results (MSE Loss)

| Epochs | Test Accuracy |
|------|------|
| 10 | ~48.89% |
| 100 | ~80% |
| 200 | ~78% |

The model quickly improved during early training and stabilized around **150 epochs** according to the training accuracy graph.

---

# Chapter 4 Implementation — Cross Entropy Loss

Chapter 4 introduces three key concepts used in modern classification models:

- **Softmax**
- **Argmax**
- **Cross Entropy Loss**

Only **1–2 lines of code needed to be changed** to move from MSELoss to CrossEntropyLoss in PyTorch.

---

# Softmax

During training, the neural network produces **logits** (raw output scores).

Example:

[2.3, 1.1, 0.4]

These values are not probabilities.

The **Softmax function** converts logits into probabilities that sum to 1.

Example:

Softmax([2.3, 1.1, 0.4]) → [0.66, 0.24, 0.10]

Now the output can be interpreted as:

- 66% probability → Class 1
- 24% probability → Class 2
- 10% probability → Class 3

---

# Cross Entropy Loss

Cross Entropy Loss measures how well the predicted probability distribution matches the **true class label**.

Instead of measuring numeric distance like MSE, it evaluates **how much probability the model assigns to the correct class**.

If the model assigns high probability to the correct class → **low loss**  
If the model assigns low probability to the correct class → **high loss**

Cross entropy is therefore designed specifically for **classification tasks**.

---

# Argmax (Inference)

During prediction, we do not need the entire probability distribution.

We simply select the class with the highest probability.

Prediction = argmax(probabilities)

Example:

[0.66, 0.24, 0.10] → Class 1

Argmax returns the index of the largest value.

---

# Training Results (Cross Entropy Loss)

| Epochs | Test Accuracy |
|------|------|
| 10 | ~42.22% |
| 100 | ~80% |

Initially the accuracy was slightly lower than MSE.

However, after sufficient training both approaches reached **similar performance**.

---

# Training Behavior Comparison

The most noticeable difference appears in the **training accuracy vs epoch graphs**.

| Loss Function | Training Stabilization |
|------|------|
| MSE Loss | ~150 epochs |
| Cross Entropy Loss | ~350 epochs |

---

# Why Training Behavior Differs

**MSE Loss**

MSE treats classification like a regression problem.  
It minimizes the squared difference between predicted values and targets.

As predictions approach the target values, gradients shrink rapidly, which can make training appear to stabilize earlier.

---

**Cross Entropy Loss**

Cross entropy maximizes the **log-likelihood of the correct class**.

Because it operates on probabilities and log values, it maintains stronger gradients during training.

This allows the model to continue adjusting weights longer to improve class separation.

---

# Observations

1. Both loss functions eventually reached **~80% test accuracy**.
2. **MSE converged faster**, but this does not necessarily mean it is better for classification.
3. **Cross entropy is theoretically more appropriate** for classification problems because it directly models class probabilities.
4. The difference becomes more significant in **larger datasets and deeper networks**.

---

# Key Takeaways

- Neural networks output **logits**, not probabilities.
- **Softmax** converts logits into probabilities during training.
- **Cross Entropy Loss** measures how well predicted probabilities match the true class.
- **Argmax** selects the predicted class during inference.
- While MSE can work for classification, **Cross Entropy Loss is the standard choice** in modern deep learning models.

---

# Training Graphs

The following graphs show training behavior for both loss functions:

- ![Training Loss vs Epoch](https://github.com/user-attachments/assets/627cee7f-f72e-4c59-8ac0-5d44a0b8dbb0)

- ![Training Accuracy for 200 epoch vs Epoch](https://github.com/user-attachments/assets/e18a5ccb-af89-4cbc-a580-02328100296a)

- ![Training Accuracy for 300 epoch vs Epoch](https://github.com/user-attachments/assets/954a08d1-4c24-4d6d-8517-d873a4a9575e)

- ![Cross Entropy Loss Example](https://github.com/user-attachments/assets/61a6ac1c-3f16-4042-a7c5-12ad956f883d)

(See images above.)

---


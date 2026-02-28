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

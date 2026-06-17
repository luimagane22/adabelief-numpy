# AdaBelief: Optimizer Implementation and Comparative Analysis

A from-scratch implementation of the **AdaBelief optimizer** and an experimental comparison against Adam and SGD with Momentum, evaluated on both classical mathematical benchmark functions and multilayer perceptron (MLP) neural networks trained on three real-world datasets.

## Overview

AdaBelief adapts its step size according to the optimizer's "belief" in the current gradient. When the observed gradient closely matches its exponential moving average (*mₜ*), larger updates are taken; when the gradient deviates significantly, the update size is reduced. This behavior makes AdaBelief more stable than Adam in noisy-gradient regions while often providing better generalization performance.

This project reproduces and analyzes the methodology proposed in the original paper by **Zhuang et al. (NeurIPS 2020)**, reimplementing all optimizers entirely in pure NumPy without relying on deep learning frameworks.

## Implemented Optimizers

| Optimizer          | Features                                                                   |
| ------------------ | -------------------------------------------------------------------------- |
| **AdaBelief**      | Gradient clipping, cosine annealing, decoupled weight decay, *sₜ* tracking |
| **Adam**           | Gradient clipping, *vₜ* tracking                                           |
| **SGD + Momentum** | Weight decay, momentum buffer                                              |

---

## Experiments

### 1. Two-Dimensional Benchmark Functions

Convergence trajectory comparison on three classical optimization benchmarks:

| Function   | Known Minimum                                     | Challenge                              |
| ---------- | ------------------------------------------------- | -------------------------------------- |
| Rosenbrock | f(1,1) = 0                                        | Curved valley with small gradients     |
| Beale      | f(3, 0.5) = 0                                     | Flat plateaus with near-zero gradients |
| Himmelblau | f(3, 2) = 0 (plus three additional global minima) | Multiple global optima                 |

---

### 2. MLP Neural Network — Fashion-MNIST

* Architecture: **784 → 256 → 128 → 10** (ReLU + Softmax)
* He initialization
* Manual backpropagation implementation
* 15 training epochs using mini-batches

| Optimizer      | Test Accuracy |
| -------------- | ------------- |
| AdaBelief      | **0.8944**    |
| Adam           | 0.8946        |
| SGD + Momentum | 0.8907        |

---

### 3. MLP Neural Network — Digits 8×8 Dataset (scikit-learn)

* 30 training epochs

| Optimizer      | Test Accuracy |
| -------------- | ------------- |
| AdaBelief      | **0.9778**    |
| Adam           | 0.9778        |
| SGD + Momentum | 0.9722        |

---

### 4. Neural Network — Wisconsin Breast Cancer Dataset

* Binary classification problem
* Sigmoid output layer
* 50 training epochs
* Per-patient probability visualization, including benign/malignant prediction confidence and false-negative identification

| Optimizer      | Test Accuracy |
| -------------- | ------------- |
| AdaBelief      | 0.9474        |
| Adam           | **0.9561**    |
| SGD + Momentum | 0.9474        |

---

## Key Technical Features

* **Backpropagation implemented entirely from scratch in NumPy** (without PyTorch or TensorFlow)
* Bias correction for first- and second-order moments (*mₜ* and *vₜ/sₜ*)
* Integrated cosine annealing learning-rate schedule in AdaBelief
* Historical tracking of *sₜ* (AdaBelief) and *vₜ* (Adam) for comparative analysis
* Visualization of loss curves, accuracy evolution, and confusion matrices

---

## Technologies

* Python 3 / Jupyter Notebook
* `numpy` — matrix computations and backpropagation
* `matplotlib`, `seaborn` — data visualization
* `scikit-learn` — datasets and preprocessing

---

## Running the Project

```bash
pip install numpy matplotlib seaborn scikit-learn
jupyter notebook AdaBelief_modif.ipynb
```

The notebook automatically downloads the Fashion-MNIST dataset through `fetch_openml`.

---

## Reference

> Zhuang, J., Tang, T., Ding, Y., Tatikonda, S., Dvornek, N., Papademetris, X., & Duncan, J.
> *"AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients."*
> Advances in Neural Information Processing Systems (NeurIPS), 2020.
> https://arxiv.org/abs/2010.07468

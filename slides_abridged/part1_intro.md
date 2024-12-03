---
marp: true
theme: beam
math: mathjax
style: |
    img[alt~="center"] {
      display: block;
      margin: 0 auto;
    }
---

# Introduction

## Latent Gaussian Compression

James Zhao, Blaine Arihara, Emily Tang, Terry Weber

<!-- 
Presentation:
-  Introduction to the problem: compression
- Dataset compression basics
- Problem setup
- Autoencoders
- Variational Autoencoders
- Contrastive Autoencoders
- Gaussian Mixture Models
- Finding Optimal k
- Methods comparison
  - Our Method
  - Submodular Maximization
  - Combined
-->

---


# Problem Setup

Suppose we have a dataset $D = \{Cat, Dog\}$ with two classes and we want to train a classifier.

- The Problem:
  - Cannot store or transmit full dataset $D$ because of
    - Network bandwidth constraints
    - Space constraints
    - Privacy constraints
- Can we share compressed dataset $D'$ (equivalent to coreset $S_k$) instead?

--- 
# Problem Assumptions

![w:700 center](../diagrams/network_diagram.png)

---
# Workflow

![w:1200 center](../diagrams/flow_diagram.png)

---

# Sequence Diagram

![w:900 center](../diagrams/usage.png)


---

# Gaussian Mixture Modeling

- Map original data in $R^n (A, B)$ to simpler latent space $R^l (A', B')$ where $l << n$.
- We can approximate the class distributions using Gaussian Mixture Models (GMMs):
  - Represent each class distribution $C' \in (A', B')$ as linear combinations of $k$ Gaussian distributions:

$$
P(z) = \sum_{i=1}^k \pi_i \mathcal{N}(\mu_{k_{C'}}, \Sigma_{k_{C'}}), \quad z \in R^l
$$
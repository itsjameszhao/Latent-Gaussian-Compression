---
marp: true
theme: beam
math: mathjax

---

# **Introduction**

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


# **Problem Setup**

Suppose we have a dataset $D = \{A, B\}$ with two classes $A$, $B$ and we want to train a classifier.

- **The Problem**:
  - Cannot store or transmit full dataset $D$ because of
    - Network bandwidth constraints.
    - Space constraints
    - Privacy constraints.
- Can we share compressed dataset $D'$ instead?

--- 
# **Problem Assumptions**

![centered w:800](../diagrams/network_diagram.png)

---
# **Dataset Assumptions**

![Centered Image](../diagrams/flow_diagram.png)

---

# **Gaussian Mixture Modeling**

- In the reduced space $R^k (A', B')$, the data looks smoother than in $R^n$.
- We can approximate the class distributions using **Gaussian Mixture Models (GMMs)**:
  - Represent the class distributions as linear combinations of Gaussian distributions.

$$
N(\mu_A, \Sigma_A), \quad N(\mu_B, \Sigma_B)
$$
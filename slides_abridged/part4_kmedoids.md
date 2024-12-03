---
marp: true
theme: beam
math: mathjax

---
# **Baseline Comparison**
As a baseline comparison for the performance, subsets of size equal to the compressed model were extracted from the MNIST dataset
- Gradient-Based Clustering
- Random Subset Selection

Each model was evaluated using a CNN classifier
<!-- ![bg right height:3in](../pics/submodular_maximization/example_size.png) -->

---

# **Gradient-Based Exemplar Clustering**

![bg right height:4in](../pics/submodular_maximization/GradientClusters.png)



Optimization problem:

$$
\arg \min_{S, \gamma_j \geq 0} |S| \quad \text{s.t.} \quad \max_{w \in W} ||\nabla_w F(w, V) - \nabla_w F(w, S)|| \leq \epsilon
$$

---
# **Baseline Results**
Test Accuracy of GBEC on the 10000 test images: 85.68%
![left height:5in](../pics/submodular_maximization/sm_confusion_matrix.png)



---

# **GMM Compression Results**

![left height:5in](../pics/AE/ae_confusion_matrix.png)

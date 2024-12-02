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

# **Visualizing GMM Distribution Learning**

![w:600 center](../pics/gmm/gmm.png)

- The image shows the learning of a Gaussian Mixture Model (GMMs) with two components ($k=2$).
- The distribution is a linear combination of the two components:
$$p(z) = \pi_1 \mathcal{N}(\mu_1, \Sigma_1) + \pi_2 \mathcal{N}(\mu_2, \Sigma_2)$$

---
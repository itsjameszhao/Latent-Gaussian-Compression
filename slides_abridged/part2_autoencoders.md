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

## Visualizing GMM Distribution Learning

![w:600 center](../pics/gmm/gmm.png)

- The image shows the learning of a Gaussian Mixture Model (GMMs) with two components ($k=2$).
- The distribution is a linear combination of the two components, but can be any integer number of components:
$$p(z) = \pi_1 \mathcal{N}(\mu_1, \Sigma_1) + \pi_2 \mathcal{N}(\mu_2, \Sigma_2)$$

---

## Compression with Autoencoders and GMMs

![bg right:50% w:240](../diagrams/autoencoder.png)
- I want to put this text to the left of the 
---

## VAE Loss Function


The VAE loss combines two terms:
1. **Reconstruction Loss**:

$$
L_{\text{recon}}(x, \hat{x}) = ||x - \hat{x}||^2
$$

2. **KL Divergence (regularizer)**:

$$
L_{\text{KL}} = D_{\text{KL}}(q(z|x) || p(z))
$$

---

## BIC Curve

![Centered Image](../pics/AE/bic_curves.png)

$$ \text{BIC} = k \ln(n) - 2 \ln(\widehat{L}) $$

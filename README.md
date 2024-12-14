# Latent Gaussian Compression (LGC): Dataset Compression and Reconstruction with Autoencoders and Gaussian Mixture Models
Authors: James Zhao, Blaine Arihara, Emily Tang, and Terry Weber

## Overview

This project explores a machine learning-based approach to compress and reconstruct large-scale image datasets, such as MNIST or CIFAR-10. By combining autoencoders and Gaussian Mixture Models (GMMs), we aim to create a compact, efficient representation of the dataset that maintains key features for classification.

### Project Goals

1. **Compress**: Use an autoencoder to encode images into a low-dimensional latent space.
2. **Model**: Capture the distribution of the compressed latent space using a GMM.
3. **Reconstruct**: Sample from the GMM and decode to recreate images.
4. **Evaluate**: Train a classifier on the reconstructed dataset and compare performance to the original.

### Motivation

Traditional compression methods may not preserve essential features for machine learning tasks. This approach leverages learned representations and generative models to balance storage efficiency with classification performance.

## Source Code

Implementations for this project can be found in the \code\ folder.

The Final Project Colab is the bulk of the implementation for this project.


# Story outline for presentation (script outline)

## Part 1 (2 min)
- Problem Set
    - We aim to compress and decompress image datasets for bandwidth-limited environments where storage and bandwidth are limited, similar to approaches in class for subset learning
- Problem Assumptions
    - Give example of problem: researcher 1 sends dataset D' to researcher 2, 
    - size(D') << size(D)
    - Save time and preserve privacy
- Dataset Assumptions
    - Describe the pipeline 
- GMM slide
    - Give an overview of R^k and how it 

## Part 2 (2 min)
- Visualizing GMM Distribution Learning
    - Cite here: https://mpatacchiola.github.io/blog/2020/07/31/gaussian-mixture-models.html?fbclid=IwZXh0bgNhZW0CMTEAAR2u5ikAd845MIqWwmxjkahZpWb8zk6bggZlxJHr87nC6Kpt9Hp5B6mpKTw_aem_e-ZHozaQ0ZUubu6iqBXXAg
    - Tie it in with our MNIST classes (2 vs 5) and how it better approximates the data in the latent space
- Compression with Autoencoders (part 1)
- Compression Step
- Add BIC slide equation
- BIC Curve
- VAE Loss Function (different loss function)
    - Tie it with SpuCo


## Parts 4 and 5 (2 min) (middle)
- Clusters
    - Explain training method
    - Explain VAE gradient extraction
- Gradient-Based Clustering
- Explain accuracies (82-85%, 97.85%, random subset %tage)

## Part 3 (2 min) (now last)
- Compression method is vulnerable to spurious correlations
- Show "no upsampling" slide images
- Tie it in to how it would compress the spurious correlations and decompress to ONLY give back the spurious correlations, removing the true data
- Explain that to ensure that the decompressed representation has the original data, you would need to do some prepreocessing to remove the spurious correlations
- TODO add pics from notebook
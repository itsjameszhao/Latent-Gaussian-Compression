# Story outline for presentation (script outline)

Slide 1: Hi, and welcome to our final project presentation, where we introduce a compression algorithm called Latent Gaussian Compression for compressing large image datasets.

Slide 2: Suppose we have a large dataset D consisting of two classes, Cat and Dog. However, we can't train on the full dataset because it is too big and the network bandwidth is limited. Is there a good way to compress dataset D to D' such that it becomes smaller while retaining essential information about the original dataset D so that the model can achieve a similar performace after training.

Slide 3: A use case for such a dataset compression algorithm is if researchers want to share training datasets between each other while preserving the privacy of the individual data points and reducing storage. In this workflow, researcher 1 trains model M on dataset D, but can't release the full dataset D due to privacy and storage constraints. Using our approach, researcher 1 can compress D to D' send D' to researcher 2, who then decompresses it to train model M' on D'.

Slide 4: An existing approach, coreset learning, where we select a coreset S* from V, can also be used for dataset compression. This approach provides both V / S speedup and compression. We implement this approach as a baseline to validate our Latent Gaussian Compression algorithm.

Slide 5: Here is a high-level overview of our algorithm. First, user 1 provides a raw image dataset, say MNIST, that they want to compress. An autoencoder learns a good latent representation from this data. The encoder part of the autoencoder then maps the dataset to the low-dimensional latent space, where several Gaussian Mixture Models are fitted, one per class, to approximate each class's latent distribution. The GMM mean, covariance, and component weight parameters are then extracted along with the decoder. This forms the compressed dataset. The dataset is then transmitted to user 2, who does the process in reverse. They sample points from the GMM's latent distribution and pass them through the decoder to reconstruct an approximation to the original images, which are then sued to train the model.


Slide 5: To fit each class distribution with a Gaussian Mixture Model, we map our original data to a lower dimensional latent space. Then we approximate each class distribution's probability density function with a weighted llinear combination of several multivariate Gaussian distributions. Next up I'll hand it over to Blaine to explain the specifics of the GMM, GMM k-component selection, and autoencoders.
 

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
- Baseline comparisons
    - CNN Classifier
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

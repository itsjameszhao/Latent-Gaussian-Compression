from __future__ import print_function
import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F

def hello_VAE():
    print("Hello from VAE.py!")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_size):
        super(Encoder, self).__init__()
        self.mu_layer = None
        self.logvar_layer = None

        # init encoder using the defined architecture
        # input image [128,1,28,28]->
        self.encoder = nn.Sequential(
            nn.Flatten(), #[128,1,28,28]->[128,784]
            nn.Linear(input_size, hidden_dim), #[128,784]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU() #[128,400]->[128,400]
            )

        # init mu layer
        self.mu_layer = nn.Linear(hidden_dim, latent_size)  #[128,400]->[128,15]
        
        # init logvar layer
        self.logvar_layer = nn.Linear(hidden_dim, latent_size)  #[128,400]->[128,15]

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder and reparametrize trick
        
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        mu = None
        logvar = None
    
        # input image [128,1,28,28]->
        h = self.encoder(x) #[128,1,28,28]->[128,400]
        # Convert encoder output to posterior mu and posterior log-variance
        mu = self.mu_layer(h) #[128,400]->[128,15]
        logvar = self.logvar_layer(h) #[128,400]->[128,15]
        
        # (2) Reparametrize to compute  the latent vector z
        z = reparametrize(mu, logvar) #[128,15]->[128,15]
        return z, mu, logvar

class Decoder(nn.Module):
    """
        Performs backward pass through FC-VAE model by estimated latent vectors
        through decoder
        
        Inputs:
        - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
        
        Returns:
        - x_hat: Reconstructed input data of shape (N,1,H,W)
        """
    def __init__(self, input_size, hidden_dim, latent_size):
        super(Decoder, self).__init__()
        # (3) Pass z through the decoder to resconstruct x  
        # latent space [128,15]->
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_dim), #[128,15]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, hidden_dim), #[128,400]->[128,400]
            nn.ReLU(), #[128,400]->[128,400]
            nn.Linear(hidden_dim, input_size), #[128,400]->[128,784]
            nn.Sigmoid(), #->[128,784]
            nn.Unflatten(1, (1, 28, 28)) #[128,784]->[128,1,28,28]      
            )

    def forward(self, z):
        x_hat = self.decoder(z)
        return x_hat

class VAE(nn.Module):
    def __init__(self, input_size=28*28, hidden_dim=400, latent_size=15,):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_size, hidden_dim, latent_size)
        self.decoder = Decoder(input_size, hidden_dim, latent_size)

    def forward(self, x):
        z, mu, logvar = self.encoder(x) #[128,15]->[128,1,28,28]
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns: 
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    
    # init epsilon as a normal distribution
    eps = torch.randn_like(logvar)
    
    # convert logvar back to standard deviation
    sigma = torch.exp(0.5 * logvar)
    
    # scale eps by mu and sigma (std dev)
    z = sigma * eps + mu

    return z


def loss_function(x_hat, x, mu, logvar, beta=1):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    loss = None
    
    # determine samples in minibatch to average the loss
    batch_size = x.size(0)
    
    reconstruction_loss = F.binary_cross_entropy(x_hat,x,reduction="sum")
    divergence_loss = -1/2 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    loss = reconstruction_loss + beta * divergence_loss
    loss /= batch_size
    return loss

"""
Spring 2023 EC ENGR 239AS Lecture 4 Notes:
For the encoder, the AE takes us from our image x, to our latent z. For the AE this was just a NN that learned that z = f(x). For the VAE, everything that was previously a deterministic NN is going to now turn into a probabilistic distribution. For VAE, we are going to sample z_s by sampling from a learned distribution, a learned distribution that is z_s ~ q(z|x), which gives us a distribution of the latents over the image and we can sample a latent from it.
For the decoder, the AE we had x_hat = g(z) that reconstructs x_hat from the low dimensional latent. For the VAE, we are going to be able to grab a sample x_hat_s ~ p(x|z) from a distribution, and that distribution is given by p(x|z). This is saying that given our latents, (instead of our autoencoder mapping one latent to one output x_hat) we're going to map our latent into a distribution over what the images look like, and we can sample from that distribution to get x_hat_s.

So we can think of the VAE as a probabalistic extension of the AE, where we replace these determinstic functions f and g, with probability distributions q(z|x) and p(x|z).

Training our autoencoder we have no gaurantee what our data will look like in latent space, we can't even say that it would exhibit something gaussian, and even if it could be represented as something gaussian it could have a mean that is centered at 100 in one of its latent dimensions. z in the VAE case is going to be mean zero and unit variance, because we're constraining it to be that, but in the AE it's just finding a z that captures information about your data, and that z can have a mean of 57. It could have a distribution anywhere in latent space.

With our VAE, were going to start with our image x \in \doubleR^n and we're going to have our latent z \in \doubleR^d. The first thing that the VAE does is learn a distribution q(z|x), which is our continuous probability distribution because our latents z are continuous and can take on any value in the latent space, and we're going to model q(z|x) to be a normal distribution N(mu(x), Sigma(x)), which is going to be parameterized by two things, once I know the mean and covariance matrix of a normal distribution then I know the distribution entirely. So the way that our x is going to affect our distribution q(z|x) is that x is going to get passed into two NNs, one that tells us the mean and one that tells us the covariance. So this is going to be q(z|x), a distribution over the latent z, where the mean is given by mu (its parameters are going to be the variable phi) and its covariance is given by Sigma. So mu_phi(x) is going to be a NN that takes our original image x, and outputs to me the mean of a d-dimensional distribution... It's going to be a NN that goes from \doubleR^n to \doubleR^d. So I take my image x, I pass it through a NN and I get the mean of this normal distribution. I take that same image x and pass it through another NN and I get the covariance of the normal distribution, d by d output which is my covariance matrix. Then I can draw a sample z from this distribution q(z|x) that has that mean and covariance determined by the NNs applied to my input image.
Then... after that the way that I generate back my original image is through another set of NNs that are parameterized by p(x|z). So p(x|z) is going to be a normal distribution where we just specify the mean and the covariance. So we just take our latent, or our random noise, and we pass that latent through the first NN which will take the d-dimensional latent into an n-dimensional mean of the distribution, and similarly for the covariance. So the VAE learns four NNs.       

q(z|x) = N(mu(x), Sigma(x)) is a normal distribution that has a mean and a covariance that changes as a function of the noise z. Every sampled z will go through the neural networks and produce its own mean and its own covariance matrix. These distributions for every sample are very expressive because there are infinitely many normal distributions where the normal distributions changes based off of what latent noise was sampled. Why a normal distribution? because in the loss function, we need to be able to compute gradients and back propagate,

We have these sampling operations that we don't know how to backpropagate through. How do you backpropagate through drawing a sample? We usually optimize mean squared error, maximize likelihood and maximizing gaussian likelihood is equivalent to minimizing mean squared error. So we thought lets go ahead and get to calculating the likelihood of the data, but when we got to that we saw that we can't mathematically do that, because p(z)*p(x|z)*dx is not a distribution that we can easily write down and integrate, its intractable. Because of this, because we can't write down the likelihood, to optimize the neural network we still need a loss function that we can compute a gradient with respect to, so we need to get a loss function out of this p(x). If we can't optimize p(x) directly, because we can't write down p(x), we're going to write a lower bound on p(x) that we can compute and optimize or maximize that lower bound. This lower bound is called the Evidence Lower Bound (ELBO) to use the evidence lower bound, we're going to bring in an approximating distribution, q(z|x), that approximates an intractable distribution p(z|x), then we'll calculate the expectation of the log likelihood under this approximating distribution. In this log likelihood we had an intractable distribution, p(z|x) that we are going to approximate with q(z|x), and we can never write down what p(z|x) because its too complicated. So the way that we're going to get rid of it is to simplify the expressions in the ELBO such that we can define a KL divergence term, a term expected value of log q(z|x) over p(z|x), which is by defintion a KL divergence. And even though we can't compute p(z|x), we know a really good property of KL divergence, which is that KL divergences are always greater than or equal to 0. So if the term is bigger than or equal to 0, then we can say the KL divergence is bigger than or equal to 0 such that the rest of the ELBO is less than or equal to the log p(x). When the ELBO becomes tight, it gets gets closer to the likelihood. The ELBO becomes tight when the KL divergence term log q(z|x) over p(z|x) equals 0 exactly. So the ELBO becomes tight when the approximating q(z|x) is actually equal to the true p(z|x). So the closer that we tighten the ELBO by making q(z|x) closer to p(z|x), the closer the ELBO approximates the log likelihood.
This allows us to derive and write a loss function or a function to optimize, L_vae, which we can take and backpropagate through our NNs and compute gradients. But how do we backprop through random sampling? In the formulation of the VAE, we are sampling z from q(z|x), the question is how do we backprop through this? We'll use a reparameterization trick. If I want to draw a sample z from q(z|x), and I know that q(z|x) is a normal distribution with mean mu_phi and covariance Sigma_phi, drawing from that distributino can be done in several ways. And one of the ways that we can do this is drawing noise from a unit gaussian distribution. So we can define an epsilon which can be drawn from a normal mean 0 identity covariance distribution. And then we know that if we do y = A*epsilon + B, then we know that y is going to be normal N(b, AA^T). So to sample from N(mu_phi, Sigma_phi), we first draw epsilon, and set B equal to mu_phi, and then set A equal to the square root of Sigma_phi (which is the Cholesky decomposition of Sigma_phi). So we have mu_phi and Sigma_phi, and we can compute z = mu_phi + sqrt(Sigma_phi)*epsilon, where epsilon is unit gaussian noise, which gives us a sample z, where z is drawn from this distribution defined by mu_phi and Sigma_phi.
The reason that this is cool is that when we draw the computational graph, sampling z in this way allows us to add the arithmetical steps needed to compute gradients. We can sample epsilon from a unit gaussian, the I take that unit gaussian sample that I get, multiply it by the square root of the covariance matrix, add to it mu_phi, and that gives me my sample, z. But what it does for me us take the sampling operation out of the backpropagation flow from the computational graph, and we can backpropagate through the plus sign and the multiplication signs.

**Variational Autoencoder (VAE)**
  - The variational autoencoder (VAE) is a probabilistic extension of the vanilla autoencoder (AE) that replaces the determinstic functionality of the encoder and decoder of an AE with probability distributions. The AE compresses a single image into a latent representation, and then reconstructs an image from that latent. The VAE takes a single image and learns a continuous probability distribution for its latent representation. This probability distribution is modeled as a normal distribution characterized by means and covariances with dimensions that of the latent dimension. Two neural networks comprise the encoder, where one learns how to represent the means of input images and the other learns how to represent the covariances of the input images. Knowing the mean and covariance of a normal distribution allows the distribution to be known entirely, so when an image is encoded by its means and covariances, it is encoded as a latent space probability distribution. To decode the image from its latent, a random sample can be drawn from the distribution that has mean and covariance determined by the neural networks applied to that image. The sample will then get passed through two more sets of neural networks that will map the latent into normal distributions of what the images look like. Similar to the encoder, the decoder is comprised of two neural networks, one which takes the sampled latent representation and determines its means in image space, and the other which determines its covariances in image space. These determined means and covariances describe the image space probability distribution of the latent, from which a random sample can be drawn to return a reconstructed image.
  - The loss function for the VAE involves defining an Evidence Lower Bound (ELBO) that allows for the probabilistic terms of the image space distribution to be defined as a Kullback-Leibler (KL) divergence. Because the image space distribution is intractable and unable to be written down, it needs to be approximated by the the latent space distribution, such that minimizing the KL divergence term tightens the ELBO, which allows it to more closely approximate the log likelihood, which is equivalent to minimizing the mean squared error of the reconstructed image. A trick is needed to back propagate through the sampling operation, called the Reparameterization trick, which allows for a random sampling of noise from a unit gaussian to be multiplied by the covariance and added to the mean of the determined latent probability distribution. This allows for the sampling operation to be performed while introducing the arithmetic operations into the computational graph needed to compute gradients through the neural networks. Below is the KL divergence term that measures the difference between the approximate posterior $q(z|x)$, which is the conditional distribution of the latents z given an image x, and the prior $p(z)$, which is the assumed unit Gaussian distribution over latents z prior to seeing any data. $$KL(q(z|x)||p(z)) = \int{q(z|x)log\frac{q(z|x)}{p(z)}dz}$$ When the approximate posterior $q(z|x)$ is parameterized as a Guassian distribution, $\mathcal{N}(\mu_\phi, \Sigma_\phi)$ with the mean and covariances determined by the encoder, the KL divergence can be represented by the closed-form solution: $$KL(q(z|x)||p(z)) = \frac{1}{2}\sum_{i=1}^d(\Sigma_i + \mu_i^2 - 1 - log\Sigma_i)$$
  - Below is the tSNE plot that shows the encoder representing the images of every class as a normal gaussian distribution, where every image is encoded to have its own unique latent z values. The z values for any image are continuous and can take on any value in latent space, and are defined by the mean and covariances learned for that image by the encoder. The scattered nature of the latents of different classes is the result of the KL divergence term pulling all the latents given an image, $q(z|x)$, closer to the assumed unit Gaussian distribution, $p(z)$.
"""
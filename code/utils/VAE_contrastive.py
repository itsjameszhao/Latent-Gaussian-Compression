from __future__ import print_function
import torch
from torch import nn
import torch.utils.data
from torch.nn import functional as F

def hello_VAE_contrastive():
    print("Hello from VAE_contrastive.py!")

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

class VAE_contrastive(nn.Module):
    def __init__(self, input_size=28*28, hidden_dim=400, latent_size=15,):
        super(VAE_contrastive, self).__init__()
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


def reconstruction_loss(x_hat, x):
    """
    Computes the reconstruction loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    
    Returns:
    - loss: Tensor containing the scalar loss for the reconstruction loss
    """
    # reconstruction_loss = F.mse_loss(x_hat, x, reduction='mean')
    # reconstruction_loss = F.binary_cross_entropy(x_hat,x,reduction="mean")
    reconstruction_loss = F.binary_cross_entropy(x_hat,x,reduction="sum")
    return reconstruction_loss

def kl_divergence_loss(mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to formulation in notebook).

    Inputs:
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z latent space dimension
    
    Returns:
    - loss: Tensor containing the scalar loss for the negative variational lowerbound
    """
    divergence_loss = -1/2 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return divergence_loss

def supervised_contrastive_loss(z, labels, temp=0.5):
    """
    Supervised contrastive loss for a batch of latent vectors.

    Inputs:
    - z: Tensor, shape (batch_size, latent_dim), latent representations.
    - labels: Tensor, shape (batch_size,), ground truth labels.
    - temp: Float, scaling factor for the similarity scores.

    Returns:
    - loss: Tensor containing the supervised contrastive loss
    """
    # normalize latent vectors
    device = z.device
    z = F.normalize(z, dim=1)
    batch_size = z.size(0)
    
    # compute pairwise cosine similarities
    similarity_matrix = torch.matmul(z, z.T) / temp
    
    # create a mask for positive pairs (examples of the same class)
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T).float().to(device) #positive_mask.shape: (batch_size, batch_size)
    
    # exclude diagonal elements (self-similarity)
    mask = torch.eye(batch_size).bool().to(device)
    positive_mask = positive_mask.masked_fill(mask, 0) #set all diagonal elements to 0
    
    # compute positive and negative similiarity components
    exp_sim = torch.exp(similarity_matrix)
    numerator = exp_sim * positive_mask
    denominator = exp_sim.sum(dim=1, keepdim=True) - exp_sim.diagonal().unsqueeze(1)
    
    # compute supervised contrastive loss
    eps = 1e-10
    contrastive_loss = -torch.log((numerator.sum(dim=1) + eps) / (denominator + eps))
    return contrastive_loss.mean()


"""
**Variational Autoencoder with Contrastive Learning:**
  - Taking the VAE previously described and adding a contrastive loss term produces the tSNE plot below. This VAE with contrastive learning now has three loss terms that are guiding the gradient computations: one for cross entropy image recontrustion, a second for KL divergence, and a third for contrastive learning. It can be seen the interplay of the KL divergence term and the contrastive loss term, as the KL divergence term pulls all examples closer to a unit Gaussian distribution, while the contrastive loss term pulls examples within classes towards each other.
"""
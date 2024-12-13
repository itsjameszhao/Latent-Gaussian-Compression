from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


def hello_vae():
    print("Hello from vae.py!")

class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size # H*W
        self.latent_size = latent_size # Z
        self.hidden_dim = 400 # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code
        layers = []
        layers.append(nn.Flatten())
        layers.append(nn.Linear(int(self.input_size), int(self.hidden_dim)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(self.hidden_dim), int(self.hidden_dim)))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(int(self.hidden_dim), int(self.hidden_dim)))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        self.mu_layer = nn.Linear(self.hidden_dim, self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, self.latent_size)
        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        
        layers = []
        layers.append(nn.Linear(self.latent_size, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append( nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, self.input_size));
        layers.append(nn.Sigmoid())
        layers.append(nn.Unflatten(dim=1, unflattened_size=(3, 28, 28)))

        self.decoder = nn.Sequential(*layers)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################


    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        encoder_out = self.encoder(x)
        mu = self.mu_layer(encoder_out)
        logvar = self.logvar_layer(encoder_out)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar

class VAEConv(nn.Module):
    def __init__(self):
        super(VAEConv, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0)
        )
        
        # Latent space: Mean and Log-variance
        self.fc_mu = nn.Linear(64 * 7 * 7, 128)
        self.fc_logvar = nn.Linear(64 * 7 * 7, 128)
        
        # Decoder
        self.fc_decode = nn.Linear(128, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation (exp(0.5 * logvar))
        eps = torch.randn_like(std)  # Random noise
        z = mu + eps * std  # Sample from the distribution
        return z
    
    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 64, 7, 7)  # Reshape to match decoder input
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

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
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code

    sigma = torch.sqrt(torch.exp(logvar))
    z = sigma * torch.randn_like(mu) + mu

    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
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
    ################################################################################################
    # TODO: Compute negative variational lowerbound loss as described in the notebook              #
    ################################################################################################
    # Replace "pass" statement with your code
    N = mu.shape[0]
    rec_term = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    

    kldiv_term = 1 + logvar - mu**2 - torch.exp(logvar)
    kldiv_term = -.5 * kldiv_term.sum()

    # print(f'rec_term: {rec_term}')
    # print(f'kldivterm: {kldiv_term}')
    loss = rec_term + kldiv_term

    loss /= N
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss

def loss_function_eval(x_hat, x, mu, logvar):
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
    N = mu.shape[0]
    rec_term = nn.functional.binary_cross_entropy(x_hat, x, reduction='mean')
    # kldiv_term = 1 + logvar - mu**2 - torch.exp(logvar)
    # kldiv_term = -.5 * kldiv_term.sum()

    # print(f'rec_term: {rec_term}')
    # print(f'kldivterm: {kldiv_term}')
    loss = rec_term# + kldiv_term

    loss /= N
    ################################################################################################
    #                            END OF YOUR CODE                                                  #
    ################################################################################################
    return loss

def upsampleGroups(group_partition):
  len_max_group = max([len(group_partition[key]) for key in group_partition.keys()])
  indices = []
  for key in group_partition.keys():
    group_reordered_indices = [group_partition[key][i] for i in torch.randperm(len(group_partition[key])).tolist()]
    group_indices = []
    while len(group_indices) < len_max_group:
        group_indices.extend(group_reordered_indices)
    indices.extend(group_indices[:len_max_group])
  return indices

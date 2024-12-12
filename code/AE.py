# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def hello_AE():
    print("Hello from AE.py!")

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_dim)  # Example layer
        self.fc2 = nn.Linear(hidden_dim, latent_size) # Example layer for latent space

    def forward(self, x):
        x = x.view(-1, self.input_size)  # Flatten input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_dim, latent_size):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Output in range [0, 1]
        x_hat = x.view(-1, 1, 28, 28)  # Reshape to image size
        return x_hat

# Combine Encoder and Decoder into Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, input_size=28*28, hidden_dim=400, latent_size=15):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_dim, latent_size)
        self.decoder = Decoder(input_size, hidden_dim, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x_hat = self.decoder(x)
        return x_hat
    
def loss_function(x_hat, x):
    loss_function = nn.MSELoss()
    loss = loss_function(x_hat, x)
    return loss

# Confirm model import to drive
print("Successfully imported AE")

"""
**Vanilla Autoencoder:**
  - The vanilla autoencoder is the baseline version of autoencoder that consists of the basic encoder and decoder components that respectively transform the higher-dimensional input images into lower-dimensional representations. Starting with the MNIST dataset, which are 28 by 28 greyscale (single channel dimension) images, the encoder flattens the input into a 784 dimensional vector and then runs the vector through two dimensional reducing linear layers with rectified linear unit (ReLU) activation functions to transform the image into a 64 dimension latent space vector. The decoder reverses the process by taking in the 64 dimensional latent space vectors and dimensionally increases the image back to its 784 dimensional vector before being reshaped back to its 28 by 28 pixel image.
  - The mean squared error (MSE) reconstruction loss is used to train the model such that the averaged squared errors between the reconstructed image and the original image is minimized.
  $$MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y_i})^2$$
  - t-Distributed Stochastic Neighbor Embedding (tSNE) is a dimensional reduction technique that attempts to preserve higher dimensional structure in a lower dimension typically for visualization. The tSNE plot below is a 2-dimensional visualization of the 64-dimensional latent vectors of the 10 classes of the MNIST dataset color-coded by class. It can seen that the encoder is transforming examples of the same class in roughly similar locations in latent space, while placing examples from different classes in different locations in latent space. This plot gives the intution of the latent encodings of different classes that the GMM is attempting to decompose into multiple gaussian distributions.
"""
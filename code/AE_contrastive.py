# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def hello_AE_contrastive():
    print("Hello from AE_contrastive.py!")

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
class AE_contrastive(nn.Module):
    def __init__(self, input_size=28*28, hidden_dim=400, latent_size=15):
        super(AE_contrastive, self).__init__()
        self.encoder = Encoder(input_size, hidden_dim, latent_size)
        self.decoder = Decoder(input_size, hidden_dim, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x_hat = self.decoder(x)
        return x_hat
    
def reconstruction_loss(x_hat, x):
    loss_function = nn.MSELoss()
    recon_loss = loss_function(x_hat, x)
    return recon_loss

def supervised_contrastive_loss(latent, labels, temp=0.5):
    """
    Supervised contrastive loss for a batch of latent vectors.

    Inputs:
    - latent: Tensor, shape (batch_size, latent_dim), latent representations.
    - labels: Tensor, shape (batch_size,), ground truth labels.
    - temp: Float, scaling factor for the similarity scores.

    Returns:
    - loss: Tensor containing the supervised contrastive loss
    """
    # normalize latent vectors
    device = latent.device
    latent = F.normalize(latent, dim=1)
    batch_size = latent.size(0)

    # compute pairwise cosine similarities
    similarity_matrix = torch.matmul(latent, latent.T) / temp
    
    # create mask for positive pairs
    labels = labels.unsqueeze(1)
    positive_mask = positive_mask = (labels == labels.T).float().to(device)
    
    # exclude diagonal elements
    mask = torch.eye(len(latent)).bool().to(device)
    positive_mask.masked_fill_(mask, 0)
    
    # compute positive and negative similarity components
    exp_sim = torch.exp(similarity_matrix)
    numerator = exp_sim * positive_mask
    denominator = exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim)
    
    # compute supervised contrastive loss
    eps = 1e-10
    contrastive_loss = -torch.log((numerator.sum(dim=1) + eps) / (denominator + eps))
    return contrastive_loss.mean()

"""
**Vanilla Autoencoder with Contrastive Learning:**
  - Contrastive learning is a technique that can be applied to supervised or self-supervised machine learning. In the self-supervised case, contrastive learning works to pull augmentations of the same example closer together, as well as similar examples with similar augmentations closer together, while pushing dissimilar examples with dissimilar augmentations away, such that different clusterings of examples can be labeled as classes. Supervised contrastive learning has the advantage of knowing the labels of the examples so that it can effectively pull examples of the same class closer together, while pushing examples of other classes away. The supervised contrastive loss term is defined as below: $$\mathcal{L}_{SCL} = \frac{1}{|I|} \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$
    - $I$: Set of all indices in the batch.
    - $P(i) $: Set of indices of positives for the anchor $i$ (samples from the same class as $i$).
    - $A(i)$: Set of all indices except $i$ itself.
    - $\mathbf{z}_i$: Normalized embedding of sample $i$.
    - $\tau$: Temperature scaling parameter.

  - When taking the vanilla autoencoder architecture and adding a contrastive loss term, the tSNE plot below shows the effect that the contrastive loss has on the training of the autoencoder. It is clear that the model is learning to place examples of the same class close together in latent space, while pushing examples in other classes away from each other.
  """
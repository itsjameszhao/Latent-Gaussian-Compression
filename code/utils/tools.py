import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import chi2
from scipy.stats import entropy
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.manifold import TSNE  # Import t-SNE

def show_images(images):
    images = torch.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))
    return 

def latent_space_interp(model, S=12, latent_size=15):
    device = next(model.parameters()).device
    z0 = torch.randn(S,latent_size , device=device)
    z1 = torch.randn(S, latent_size, device=device)
    w = torch.linspace(0, 1, S, device=device).view(S, 1, 1)
    z = (w * z0 + (1 - w) * z1).transpose(0, 1).reshape(S * S, latent_size)
    x = model.decoder(z)
    show_images(x.data.cpu())

def sample_VAE_image(model, num_images=5, latent_size=64, conditional=False):
    device = next(model.parameters()).device
    num_classes = 10  #hard code 10 for MNIST
    z = torch.randn(num_images, latent_size, device=device)
    if conditional:
        c = torch.zeros(num_images, num_classes, device=device)
        for i in torch.arange(num_images):
            c[i, i%num_classes] = 1
        z = torch.cat((z, c), dim=1)
    x = model.decoder(z)
    images = x.data.cpu()

    img_dim = int(images.shape[-1]) 
    img_scale = 1.5
    width = num_classes if (num_images // num_classes > 0) else num_images % num_classes
    height = num_images // num_classes + 1
    fig = plt.figure(figsize=(width*img_scale, height*img_scale))
    gs = gridspec.GridSpec(height, width)
    gs.update(wspace=0.05, hspace=0.05)
    for i, img in enumerate(images):
        h,w = divmod(i,num_classes)
        ax = plt.subplot(gs[h,w])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape(img_dim,img_dim), cmap='gray')

    return

def one_hot(labels, class_size):
    # Create one hot label matrix of size (N, C)
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets

def calc_mean_cov(samples):
    mu = np.mean(samples, axis=0)
    sigma = np.cov(samples, rowvar=False)
    return mu, sigma

def calc_mahalanobis(samples, mu, sigma):
    mahal_dists = np.zeros(samples.shape[0])
    inv_sigma = np.linalg.inv(sigma)
    for p in range(samples.shape[0]):
        diff = samples[p,:] - mu
        mahal_dists[p] = diff @ inv_sigma @ diff.T
    return mahal_dists

def is_gaussian(mahal_dists, dof, thresh):
    # cutoff of dof degree chi squared inverse CDF
    cutoff = chi2.ppf(0.95, dof)
    mahal_dists_sort = np.sort(mahal_dists)
    keep = np.sum(mahal_dists_sort < cutoff)
    percent_discard = (len(mahal_dists_sort) - keep) / len(mahal_dists_sort)
    is_gaussian = percent_discard < thresh
    return is_gaussian

def plot_mahal_hist(mahal_dists, dof):
    plt.figure()
    plt.hist(mahal_dists, bins=30, density=True, alpha=0.6, label='Mahalnobis hist')
    x = np.linspace(0, np.max(mahal_dists), 100)
    y = chi2.pdf(x, dof)
    plt.plot(x, y, 'b-', linewidth=2, label=f'{dof} DoF Chi-squared PDF')
    plt.grid(True)
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Probability Density')
    plt.title(f'Empirical vs. Theoretical Chi-squared PDF')
    plt.legend()
    plt.show()

def z_score_threshold(data, z_thresh=4):
    # median absolute deviation rejection
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    mask = np.abs(z_scores) <= z_thresh
    chopped_data = data[mask]
    return chopped_data, mask

def calc_mahal_from_gmm(latent_representations, labels, means, covariances, plot_flag=False):
    # grab number of classes and GMMs per class
    num_classes = len(means)
    
    mahal_dists = {}
    # calculate the Mahalanobis distances of the latent space vectors for every GMM
    # for every class using the calculated means and covariances for every GMM
    for n in range(num_classes):
        # grab only the latent space encodings for class n
        num_gmms = means[n].shape[0]
        class_encodings = latent_representations[labels == n]
        for k in range(num_gmms):
            # unpack the means and covariances from the gmm model
            mu = means[n][k,:]
            sigma = covariances[n][k,:]
            # calculate the mahalanobis distances and plot
            mahal_dists[(n,k)] = calc_mahalanobis(class_encodings, mu, sigma)
    
    if plot_flag:
        dof = len(means[0][0])
        # plot grid with number of gmms along columns, and classes along rows
        max_gmms = max(len(sublist) for sublist in means.values())
        fig = plt.figure(figsize=(max_gmms*1.5, num_classes))
        gs = gridspec.GridSpec(num_classes, max_gmms)
        gs.update(wspace=0.1, hspace=0.2)
        
        # create plots for every gmm distribution for every class
        for n in range(num_classes):
            num_gmms = means[n].shape[0]
            for k in range(num_gmms):
                ax = plt.subplot(gs[n,k])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_aspect('equal')

                # grab the mahalanobis distances for the GMM for the class
                data = mahal_dists[(n,k)]
                _, mask = z_score_threshold(data, z_thresh=4) #mask out outliers

                plt.hist(data[mask], bins=30, density=True, alpha=0.6, label='Mahalnobis hist')
                # plot the theoretical chi squared pdf of same dof
                x = np.linspace(0, np.max(data[mask]), 100)
                y = chi2.pdf(x, dof)
                plt.plot(x, y, 'b-', linewidth=1, label=f'{dof} DoF Chi-squared PDF')
                plt.grid(True)
                ax.set_title(f'Class {n + 1}, GMM {k + 1}', fontsize=6,pad=-20)
                # ax.legend(fontsize=8)
                ax.set_xlabel('Mahalanobis Distance')
                ax.set_ylabel('Probability Density')
    
        plt.tight_layout()  # Reduce padding between the figure and subplots
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)  # Adjust borders
        plt.show()
        return mahal_dists

def calc_kl_div(mahal_dists, dof, thresh=1, plot_flag=False):
    """
    Small KL Divergence ( <0.1 ):
        Indicates that the two distributions are very similar.
        This range is typically considered a strong match between the empirical and theoretical distributions.
        
    Moderate KL Divergence ( 0.1 - 1.0 ):
        Suggests some discrepancy between the distributions, but they are still reasonably similar.
        For many machine learning or statistical modeling tasks, a KL divergence in this range might be acceptable.
        
    Large KL Divergence ( >1.0 ):
        Indicates a significant difference between the distributions.
        This suggests the empirical data does not closely follow the theoretical distribution.
    """
    num_classes = max(key[0] for key in mahal_dists.keys()) + 1
    kl_divergence = {}
    is_gaussian = {}

    if plot_flag:
        # plot grid with number of gmms along columns, and classes along rows
        max_gmms = max(key[1] for key in mahal_dists.keys()) + 1
        fig = plt.figure(figsize=(max_gmms, num_classes))
        gs = gridspec.GridSpec(num_classes+1, max_gmms+1)
        gs.update(wspace=0.4, hspace=0.6)

    for n in range(num_classes):
        num_gmms = max([key[1] for key in mahal_dists.keys() if key[0] == n]) + 1
        for k in range(num_gmms):
            # unpack the mahalanobis distances for each GMM for every class
            data = mahal_dists[(n,k)]
            _, mask = z_score_threshold(data, z_thresh=4) #mask out outliers
            # TEST KL DIVERGENCE CALCULATION. KEEP COMMENTED
            # data = np.random.chisquare(df=dof, size=1000)

            # normalize the empirical histogram (P)
            num_bins = 100
            hist, bin_edges = np.histogram(data[mask], bins=num_bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  #bin centers
            p = hist

            # compute the theoretical chi-squared pdf (Q) over the same bins
            q = chi2.pdf(bin_centers, df=dof)

            # ensure no zeros in either distribution
            epsilon = 1e-16
            p = np.clip(p, epsilon, None)
            q = np.clip(q, epsilon, None)

            # calculate KL divergence using scipy's entropy function which computes KL divergence
            # or the Shannon entropy or relative entropy of given distributions (P and Q)
            kl_divergence[(n,k)] = entropy(p, q)
            is_gaussian[(n,k)] = entropy(p, q) <= 1
            
            if plot_flag:
                ax = plt.subplot(gs[n+1,k+1])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_aspect('equal')
                # plot the distributions for comparison
                plt.plot(bin_centers, p, label="Empirical PDF (Histogram)", marker=".")
                plt.plot(bin_centers, q, label="Theoretical Chi-squared PDF", linestyle="--")
                ax.set_title(f"Class {n+1}, GMM {k+1}", fontsize=6, pad=5)
                ax.set_xlabel("Mahalanobis Distance")
                ax.set_ylabel("Density")
                plt.grid(True)
                # plt.legend()
                plt.show()
    if plot_flag:
        fig.subplots_adjust(left=0.01, right=0.95, top=0.99, bottom=0.05)  # Adjust borders
        plt.tight_layout()  # Adjust layout to fit the figure
        plt.show()

    return kl_divergence, is_gaussian

def plot_bic_scores(bic_scores, max_cols=5):
    num_classes = len(bic_scores)
    max_rows = (num_classes + max_cols - 1) // max_cols
    fig = plt.figure(figsize=(max_cols*4, max_rows*3))
    gs = gridspec.GridSpec(max_rows, max_cols, wspace=0.4,hspace=0.3)

    # plot the BIC scores over k classes
    for ind, (digit, scores) in enumerate(bic_scores.items()):
        # (ind//max_cols, mod(ind, max_cols)) = divmod(ind, max_cols)
        row, col = divmod(ind, max_cols)
        ax = fig.add_subplot(gs[row, col])

        ax.plot(range(1, len(scores)+1), scores, marker='.', label=f'Class {digit}')
        ax.set_title(f'Class {digit}', fontsize=10)
        ax.set_xlabel('Number of components (K)', fontsize=8)
        ax.set_ylabel('BIC score', fontsize=8)
        ax.grid(True)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_tsne(samples, labels, plt_dim=2):
    
    colors = [
            '#C71585',  # MediumVioletRed
            '#4682B4',  # SteelBlue
            '#008080',  # Teal
            '#DAA520',  # Goldenrod
            '#8B0000',  # DarkRed
            '#556B2F',  # DarkOliveGreen
            '#483D8B',  # DarkSlateBlue
            '#2E8B57',  # SeaGreen
            '#FF8C00',  # DarkOrange
            '#9932CC',  # DarkOrchid
            '#8B4513',  # SaddleBrown
            '#708090',  # SlateGray
            '#6B8E23',  # OliveDrab
            '#FF6347',  # Tomato
            '#40E0D0',  # Turquoise
            '#B22222',  # FireBrick
            '#5F9EA0'   # CadetBlue
        ]
    
    if plt_dim == 2:
        # Visualize clusters in latent space using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        # Then apply t-SNE on the sample
        latent_2d = tsne.fit_transform(samples)
        # Prepare data in a DataFrame for Plotly
        data = pd.DataFrame({
            't-SNE Dimension 1': latent_2d[:, 0],
            't-SNE Dimension 2': latent_2d[:, 1],
            'Digit': labels
        })
        # Create a Plotly scatter plot with distinct palette and distinct markers
        fig = go.Figure()
        for name, group in data.groupby('Digit'):
            fig.add_trace(go.Scatter(
                x=group['t-SNE Dimension 1'],
                y=group['t-SNE Dimension 2'],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=7,
                    color=colors[name],
                    opacity=0.8
                ),
                text=f'Cluster {name}'
            ))

    elif plt_dim == 3:
        tsne = TSNE(n_components=3, random_state=42)
        latent_3d = tsne.fit_transform(samples)
        data = pd.DataFrame({
            't-SNE Dimension 1': latent_3d[:, 0],
            't-SNE Dimension 2': latent_3d[:, 1],
            't-SNE Dimension 3': latent_3d[:, 2],
            'Digit': labels,  # Replace labels with your digit labels
        })

        fig  = go.Figure()
        for name, group in data.groupby('Digit'):
            fig.add_trace(go.Scatter3d(
                x=group['t-SNE Dimension 1'],
                y=group['t-SNE Dimension 2'],
                z=group['t-SNE Dimension 3'],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=4,
                    color=colors[name],
                    opacity=0.8
                ),
                text=f'Cluster {name}'
            ))
    else:
        print(f'Invalid dim: {plt_dim}')

    # Update layout for better readability
    fig.update_layout(
        title="Latent Space Distribution by Class (t-SNE)",
        title_font_size=18,
        legend_title_text='Digit',
        legend_title_font_size=14,
        legend_font_size=12
    )

    # Show the interactive plot
    fig.show()
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from fastcore.transform import Transform
from fastaudio.core.signal import AudioTensor
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

class AudioNormalize(Transform):
    """Normalize a single `AudioTensor`."""
    def encodes(self, x:AudioTensor): return (x - x.mean()) / x.std()

    
class SpectrogramToFakeRGB(Transform):
    """Transform single-channel spectrograms into fake RGB images
    
    Args:
        as_batch_tfm: True if used as batch transform
    """
    def __init__(self, as_batch_tfm=True):
        n_dims = 4 if as_batch_tfm else 3
        channel_dim_idx = 1 if as_batch_tfm else 0
        self.sizes = [3 if dim == channel_dim_idx else 1 for dim in range(n_dims)]

    def encodes(self, x: AudioSpectrogram):
        return x.repeat(*self.sizes)
    

def print_top_results(test_sample_idx, preds, labels, vocab, show_max=5):
    """Print top results for a single test sample"""
    idx_sort = preds[test_sample_idx].argsort(descending=True)
    print(f'Top {show_max} results for sample \'{labels[test_sample_idx]}\':')
    for rank,i in enumerate(idx_sort[:show_max]):
        print(f'    [#{rank+1}] {vocab[i]} ({preds[test_sample_idx][i]:.2f})')


def plot_top_results (preds_softmax, vocab, title=None, show_max=5, figsize=None):
    """Plot horizontal bar plot showing top classification results."""
    figsize = figsize or (5, 2)
    fig, ax = plt.subplots(figsize=figsize)
    
    show_max = show_max or len(preds_softmax)
    idx_sort = preds_softmax.argsort(descending=True)
    
    x = np.arange(show_max)
    if title is not None:
        ax.set_title(title)
    ax.barh(x[::-1], preds_softmax[idx_sort[:show_max]])
    ax.set_xlabel('p')
    ax.set_ylabel('Category')
    ax.set_xlim([0, 1])
    ax.set_yticks(x)
    ax.set_yticklabels(vocab[idx_sort[:show_max]][::-1])
    
    return fig, ax

def plot_loss_landscape_3d(a, b, model, loss_fn, xx, yy, plot_initial_params=True, param_trajectory=None, seed=42):
    torch.manual_seed(seed)
    model[0].reset_parameters()
    
    n_mesh = 50
    xv, yv = torch.meshgrid(torch.linspace(a - 2, a + 2, n_mesh), torch.linspace(b - 2, b + 2, n_mesh))
    zv = torch.zeros_like(xv)
    
    with torch.no_grad():
        initial_params = torch.tensor([*[p for p in model.parameters()], loss_fn(model(xx), yy)])
        for i, (x, y) in enumerate(zip(xv.flatten(), yv.flatten())):
            pa, pb = model.parameters()
            pa.copy_(x.reshape(pa.shape))
            pb.copy_(y.reshape(pb.shape))
            loss = loss_fn(model(xx), yy)
            zv.flatten()[i] = loss
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_loss = zv.min().item()
    max_loss = zv.max().item()
    ax.plot_surface(
        xv.numpy(), yv.numpy(), zv.numpy(),
        cmap=mpl.cm.inferno,
        norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2),
        antialiased=False, zorder=0)
    if plot_initial_params:
        ax.text3D(*initial_params, 'x', zorder=100)
    if param_trajectory is not None:
        ax.plot(*param_trajectory.T, zorder=20, color='white')

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('loss')
    ax.view_init(elev=30, azim=-110)
    fig.tight_layout()
    return fig, ax

def plot_loss_landscape_2d(a, b, model, loss_fn, xx, yy, plot_initial_params=True, param_trajectory=None, seed=42):
    torch.manual_seed(seed)
    model[0].reset_parameters()
    
    n_mesh = 50
    xv, yv = torch.meshgrid(torch.linspace(a - 2, a + 2, n_mesh), torch.linspace(b - 2, b + 2, n_mesh))
    zv = torch.zeros_like(xv)
    
    with torch.no_grad():
        initial_params = torch.tensor([*[p for p in model.parameters()], loss_fn(model(xx), yy)])
        for i, (x, y) in enumerate(zip(xv.flatten(), yv.flatten())):
            pa, pb = model.parameters()
            pa.copy_(x.reshape(pa.shape))
            pb.copy_(y.reshape(pb.shape))
            loss = loss_fn(model(xx), yy)
            zv.flatten()[i] = loss
        
    min_loss = zv.min().item()
    max_loss = zv.max().item()
    fig, ax = plt.subplots()
    cs = ax.contourf(
        xv.numpy(), yv.numpy(), zv.numpy(), 
        levels=np.logspace(np.log10(min_loss), np.log10(max_loss), n_mesh),
        norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2),
        cmap=mpl.cm.inferno)
    if plot_initial_params and param_trajectory is None:
        ax.text(*initial_params[:2], 'x')
    if param_trajectory is not None:
        ax.plot(*param_trajectory.T[:2], color='white', label='model params')
        ax.scatter(a, b, color='red', label='target')
        ax.text(*(param_trajectory[0, :2] + 0.01), 'initial', color='white')
        ax.text(*(param_trajectory[-1, :2] - 0.01), 'final', color='white', ha='right')
        ax.legend();

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    fig.tight_layout()
    return fig, ax
from fastcore.transform import Transform
from fastaudio.core.signal import AudioTensor
import numpy as np
import matplotlib.pyplot as plt

class AudioNormalize(Transform):
    "Normalizes a single `AudioTensor`."
    def encodes(self, x:AudioTensor): return (x - x.mean()) / x.std()


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
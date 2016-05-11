# -*- coding:utf-8 -*-

# This demo is based on O'reilly's website of Introduction to the t-SNE Algorithm

__author__ = "JOHNKYON"

# numpy and scipy imports
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# sklearn(Everyone is using this)
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# Code in sklearn 0.15.2
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, _kl_divergence)
from sklearn.utils.extmath import _ravel

# Random state
RS = 20160510

# matplotlib for graphics
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# Import seaborn to make nice plots
import seaborn as sns

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context('notebook', font_scale=1.5, rc={"line.linewidth": 2.5})

# Generate an animation with matplotkib and moviepy
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy

# load digits
digits = load_digits()
digits.data.shape

# print(digits['DESCR'])

nrows, ncols = 2, 5
plt.figure(figsize=(6, 3))
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i, ...])
    plt.xticks([]);
    plt.yticks([])
    plt.title(digits.target[i])
plt.savefig('images/digits_generated.png', dpi=150)

# reorder the data points according to labels
X = np.vstack([digits.data[digits.target == i]
               for i in range(10)])
y = np.hstack([digits.target[digits.target == i]
               for i in range(10)])

digits_proj = TSNE(random_state=RS).fit_transform(X)


def scatter(x, colors):
    # color palette with seaborn
    palette = np.array(sns.color_palette("hls", 10))

    # Create a scatter plot
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # Add labels for each digit
    txts = []
    for i in range(10):
        # Position of each label
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])
        txts.append(txt)

    return f, ax, sc, txts


scatter(digits_proj, y)
plt.savefig('images/digits_tsne-generated.png', dpi=120)


# compute similarity matrix
def _joint_probabilities_constant_sigma(D, sigma):
    # print D
    P = np.exp(-D ** 2 / 2 * sigma ** 2)
    P /= np.sum(P, axis=1)
    return P


# Pairwise distances between all data points
D = pairwise_distances(X, squared=True)
# Similarity with constant sigma
P_constant = _joint_probabilities_constant_sigma(D, .002)
# Similarity with variable sigma
P_binary = _joint_probabilities(D, 30., False)
# The output of this function needs to be reshaped to a square matrix
P_binary_s = squareform(P_binary)

plt.figure(figsize=(12, 4))
pal = sns.light_palette("blue", as_cmap=True)

plt.subplot(131)
plt.imshow(D[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title("Distance matrix", fontdict={'fontsize': 16})

plt.subplot(133)
plt.imshow(P_binary_s[::10, ::10], interpolation='none', cmap=pal)
plt.axis('off')
plt.title('$p_{j|i}$ (variable $\sigma$)', fontdict={'fontsize': 16})
plt.savefig('images/similarity-generated.png', dpi=120)

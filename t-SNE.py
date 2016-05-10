# -*- coding:utf-8 -*-

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



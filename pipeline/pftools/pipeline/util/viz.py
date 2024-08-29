

import matplotlib.colors as mcolors
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from skimage.color import rgb2lab

import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt

def calculate_view_limits(cells):
    min_x = np.min([np.min(x) for x,y in cells.geometry.apply(lambda x: x.exterior.xy)])
    max_x = np.max([np.max(x) for x,y in cells.geometry.apply(lambda x: x.exterior.xy)])
    min_y = np.min([np.min(y) for x,y in cells.geometry.apply(lambda x: x.exterior.xy)])
    max_y = np.max([np.max(y) for x,y in cells.geometry.apply(lambda x: x.exterior.xy)])
    return min_x, max_x, min_y, max_y

def plot_cell_polygons_with_values(cells, values, ax=None, alpha=0.5, linewidth=1, edgecolor='blank', cmap='viridis', vmin=None, vmax=None):
    """
    Plot the cell polygons on the given axis.
    Plot using add_patch, filling with a particular color.
    """ 
    color_map = plt.cm.get_cmap(cmap)
    min_x, max_x, min_y, max_y = calculate_view_limits(cells)
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(cells.shape[0]):
        x, y = cells.iloc[i].geometry.exterior.xy
        vertices = np.array([x,y]).T
        #print(np.min(x), np.max(x), np.min(y), np.max(y))
        ax.add_patch(plt.Polygon(vertices, color=mcolors.to_hex(color_map(values[i])), alpha=alpha, linewidth=linewidth, edgecolor=edgecolor))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    return ax

def plot_cell_polygons(cells, ax=None, color='gray', alpha=0.5, linewidth=1, edgecolor='black'):
    """
    Plot the cell polygons on the given axis.
    Plot using add_patch, filling with a particular color.
    """ 
    min_x, max_x, min_y, max_y = calculate_view_limits(cells)
    if ax is None:
        fig, ax = plt.subplots()
    for i in range(cells.shape[0]):
        x, y = cells.iloc[i].geometry.exterior.xy
        vertices = np.array([x,y]).T
        ax.add_patch(plt.Polygon(vertices, color=color, alpha=alpha, linewidth=linewidth, edgecolor=edgecolor))
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y) 
    return ax

def random_colormap(N):
    colormap = np.random.rand(N, 3)
    return plt.cm.colors.ListedColormap(colormap)

def perceptually_distinct_colormap(N, candidate_count=10000, seed=42):
    # Generate candidate colors in RGB
    np.random.seed(seed)
    rgb_candidates = np.random.rand(candidate_count, 3)
    
    # Convert candidates to LAB color space
    lab_candidates = rgb2lab(rgb_candidates)

    # Compute pairwise Euclidean distances between candidates in LAB space
    pairwise_distances = pdist(lab_candidates)

    # Convert the pairwise distances to a square matrix
    distance_matrix = squareform(pairwise_distances)

    # Optimization step to maximize the minimum distance
    # Here, we're trying to find a subset of N colors that have the greatest minimum pairwise distance
    _, selected_indices = linear_sum_assignment(-distance_matrix, maximize=True)
    selected_indices = selected_indices[:N]

    # Return the corresponding RGB colors
    return plt.cm.colors.ListedColormap(rgb_candidates[selected_indices])


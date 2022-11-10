"""
This script is used to analyze the extracted data. The point is to make the
kymograph, curvature time lapse plot, and the plot of the height/length ratio
over time
"""

import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import copy

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--data_path", help="path of folder containing data", type=str
)

args = parser.parse_args()
data_path = args.data_path
plot_path = data_path + "/plots"


folders = os.listdir(data_path)
folders = [f for f in folders if "." not in f]  # ignore .DS_Store
folders.sort(key=int)

interpolation_points = []

# populate the arrays
for f in folders:
    arr = np.load(data_path + "/" + f + "/arr.npz")
    interpolation_points.append(arr["arr_0"])

stacked_points = np.hstack(interpolation_points)
stacked_points.sort()
inte_points = np.linspace(stacked_points.min(), stacked_points.max(), 1000)

curve_preds = []
for fold in folders:
    with open(data_path + "/" + fold + "/interpolator.pkl", "rb") as f:
        spl_loaded = pickle.load(f)
        curve_pred = spl_loaded(inte_points)
        curve_preds.append(curve_pred)
curve_mat = np.vstack(curve_preds)

plt.figure(figsize=(10, 10))
skip = int(curve_mat.shape[0] * 0.1)
skppd = curve_mat[skip:]


def replace_with_nan_flip(curve_mat: np.ndarray):
    """
    Find where the interpolation went out of range and just predicted all zeros.
    Replace the zero values with Nans. Also flip the matrix before we get to
    that point so the base is always the leftmost part of the frame.
    """
    i = 0
    for row in curve_mat:
        try:  # assuming there are no interpolated zero values. might fail with negative curvature
            zero_idx = np.argwhere(row == 89)[0][0]
            row[zero_idx:] = np.nan
            pre_zero = copy.deepcopy(row[:zero_idx])
            flip_pre_zero = np.flip(pre_zero)
            row[:zero_idx] = flip_pre_zero
        except:  # deal with the case wehre there are no zeros
            curve_mat[i] = np.flip(curve_mat[i])
        i += 1
    return curve_mat


skppd = replace_with_nan_flip(skppd)

plt.imshow(
    skppd,
    interpolation=None,
    cmap="viridis",
    aspect="auto",
    origin="lower",
    extent=[0, 1, 0, 1],
)
plt.clim(vmax=0.2)
current_cmap = matplotlib.cm.get_cmap()
current_cmap.set_bad(color="white")
plt.colorbar()
plt.xlabel(r"$\frac{l}{l_{final}}$", fontsize=16)
plt.ylabel(r"$\frac{t}{t_{final}}$", fontsize=16)
plt.title("Curvature (1/cm) along snake over time", fontsize=20)
# plt.show()
os.mkdir(plot_path)
plt.savefig(plot_path + "/new_kymograph.png")

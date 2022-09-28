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
plt.imshow(
    skppd,
    interpolation=None,
    cmap="viridis",
    aspect="auto",
    origin="lower",
    extent=[0, 100, 0, skppd.shape[0]],
)
plt.clim(vmin=0, vmax=0.2)
plt.colorbar()
plt.xlabel("Percent along length of snake")
plt.ylabel("Frame #")
plt.title("Curvature (1/cm) along snake over time", fontsize=20)
# plt.show()
os.mkdir(plot_path)
plt.savefig(plot_path + "/new_kymograph.png")

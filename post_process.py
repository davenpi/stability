"""
This script is used to analyze the extracted data. The point is to make the
kymograph, curvature time lapse plot, and the plot of the height/length ratio
over time
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--data_path", help="path of folder containing data", type=str
)

args = parser.parse_args()
data_path = args.data_path
plot_path = data_path + "/plots"

os.mkdir(plot_path)

files = []
for filename in os.listdir(data_path):
    if filename.endswith(".npz"):
        im_file = os.path.join(data_path, filename)
        files.append(im_file)


def extract_num_from_npy_string(string):
    numbers = re.findall("[0-9]+", string[-9:])  # list of strings
    numbers = int(numbers[0])  # convert num string to int
    return numbers


fnames = sorted(files, key=extract_num_from_npy_string)

# format is image, interpolated line, interpolation points, curvature, and
# height/length values.
cropped_images = []
interpolated_lines = []
interpolation_points = []
curvatures = []
height_length_ratios = []

# populate the arrays
for f in fnames:
    arr = np.load(f)
    cropped_images.append(arr["arr_0"])
    interpolated_lines.append(arr["arr_1"])
    interpolation_points.append(arr["arr_2"])
    curvatures.append(arr["arr_3"])
    height_length_ratios.append(arr["arr_4"][0])

n_ims = len(interpolation_points)
frms_list = []
for i in range(n_ims):
    frmi = list(zip(interpolation_points[i], curvatures[i]))
    frms_list.append(frmi)

# DON'T THINK THIS ACTUALLY WORKS
def deal_with_zero_dup(tup_list: list, arr_idx: int, n_ims: int) -> list:
    """
    Drop the tuples which duplicate zero starting elements

    Just gets rid of duplicates in the list and makes interpolation easier.
    In general this is a dirty method but it's better than nothing.
    """
    zero_start_list = [i for i in tup_list if i[0] == 0.0]
    tup_list = tup_list[n_ims:]  # ignore the first n_ims things
    tup = zero_start_list[arr_idx]
    tup_list.insert(0, tup)
    return tup_list


# Do the padding.
zeros_list = np.zeros(100)
padded_frames = []
for i in range(n_ims):
    frmi = frms_list[i]
    idxs = list(np.linspace(0, n_ims - 1, n_ims, dtype=int))
    idxs.remove(i)
    for idx in idxs:
        zipped = list(zip(interpolation_points[idx], zeros_list))
        frmi.extend(zipped)
    frmi.sort(key=lambda el: el[0])
    # frmi = deal_with_zero_dup(frmi, i, n_ims)
    padded_frames.append(frmi)

# extract the extended curvatures
ext_curvatures = []
for lst in padded_frames:
    ext_curve = [tup[1] for tup in lst]
    ext_curvatures.append(ext_curve)

curve_mat = np.vstack(ext_curvatures)

# plot the results
plt.figure(figsize=(10, 10))
plt.imshow(
    curve_mat[::], interpolation=None, cmap="viridis", aspect="auto", origin="lower"
)
plt.clim(vmin=0, vmax=0.1)
plt.colorbar()
plt.xlabel("Length of snake")
plt.ylabel("Frame #")
plt.title("Curvature (1/cm) along snake over time", fontsize=20)
plt.savefig(plot_path + "/kymograph")


# TO DO
# - add in curvature time lapse plot
# - add in height/length vs time plot

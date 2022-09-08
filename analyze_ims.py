"""
This file is for loading in a folder of images, analyzing all of them, and
then outputting the results of each image analysis into another folder for
later processing.
"""

import os
import argparse
import numpy as np
import vid_helpers as vh

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--ims_path", help="path of folder containing images", type=str
)
parser.add_argument(
    "-ppcm", "--px_per_cm", help="pixels per centimeter in the images", type=float
)

# cal_dict will hold the pixel per cm value from a downsampled video for each
# calibration. The video must be rotated and then downsampled to 540:-1 for
# the calibration number to make sense.
cal_dict = {8: 8.5758}
args = parser.parse_args()
ims_path = args.ims_path
px_per_cm = args.px_per_cm
save_folder = ims_path + "/extracted_data"
os.mkdir(
    save_folder,
)

print(px_per_cm)


files = []
for filename in os.listdir(ims_path):
    if filename.endswith(".png"):
        im_file = os.path.join(ims_path, filename)
        files.append(im_file)

# sort the list of files


def extract_nums_from_str(string: str) -> int:
    """
    Extract numbers from string of the form 'stringxxxx.png'
    """
    return int(string[-8:-4])


files = sorted(files, key=extract_nums_from_str)

i = 0
for im_file in files:
    # print("Image path is: " + im_file + "\n")
    # print("Save path is: " + save_folder + "\n")
    img = vh.load_img(img_path=im_file)
    front = vh.extract_front(img)
    bottom = vh.extract_bottom(img)
    hybrid = vh.hybridize(front, bottom)
    distance = vh.get_distance_parameter(outline=hybrid, px_per_cm=px_per_cm)
    interpolated_line, interpolator = vh.get_interpolation(
        distance, hybrid, num_points=15
    )
    interpolation_points = vh.get_interp_points(distance=distance, num_points=15)
    smooth_kappa, smooth_points = vh.compute_curvature(
        interpolated_line, interpolation_points
    )
    np.savez(
        save_folder + f"/im{i}",
        img,
        interpolated_line,
        smooth_points,
        smooth_kappa,
    )
    i += 1

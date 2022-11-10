"""
This file is for loading in a folder of images, analyzing all of them, and
then outputting the results of each image analysis into another folder for
later processing.
"""
import argparse
import os
import pickle
import numpy as np
import vid_helpers as vh

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--ims_path", help="path of folder containing images", type=str
)
parser.add_argument(
    "-ppcm", "--px_per_cm", help="pixels per centimeter in the images", type=float
)

parser.add_argument(
    "-n",
    "--reaching_num",
    help="how many pixels counts as reaching high in an image",
    type=int,
)

args = parser.parse_args()
ims_path = args.ims_path
px_per_cm = args.px_per_cm
reaching_px = args.reaching_num
save_folder = ims_path + "/extracted_data_signed"  # add in signed curvature
os.mkdir(
    save_folder,
)

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
    os.mkdir(ims_path + f"/extracted_data_signed/{i}")
    img = vh.load_img(img_path=im_file)
    front = vh.extract_front(img)
    bottom = vh.extract_bottom(img)
    hybrid = vh.hybridize(img, front, bottom, reaching_num=reaching_px)
    distance = vh.get_distance_parameter(outline=hybrid, px_per_cm=px_per_cm)
    interpolated_line, interpolator = vh.get_interpolation(
        distance, hybrid, num_points=15
    )
    interpolation_points = vh.get_interp_points(distance=distance, num_points=15)
    smooth_kappa, smooth_points, smooth_interp = vh.compute_curvature(
        interpolated_line, interpolation_points, px_per_cm=px_per_cm
    )
    with open(save_folder + f"/{i}/interpolator.pkl", "wb") as f:
        pickle.dump(smooth_interp, f)
    np.savez(
        save_folder + f"/{i}/arr",
        smooth_points,
    )
    i += 1

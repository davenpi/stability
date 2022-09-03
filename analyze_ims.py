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

parser.add_argument("ims_path", help="path of folder containing images")
args = parser.parse_args()
ims_path = args.ims_path
save_folder = ims_path + "/extracted_data"
os.mkdir(
    save_folder,
)

i = 0
for filename in os.listdir(ims_path):
    if filename.endswith(".png"):
        im_file = os.path.join(ims_path, filename)
        # print("Image path is: " + im_file + "\n")
        # print("Save path is: " + save_path + "\n")
        img = vh.load_img(img_path=im_file)
        front = vh.extract_front(img)
        distance = vh.get_distance_parameter(outline=front)
        interpolated_line, interpolator = vh.get_interpolation(
            distance, front, num_points=15
        )
        interpolation_points = vh.get_interp_points(distance=distance, num_points=15)
        smooth_kappa, smooth_points = vh.compute_curvature(
            interpolated_line, interpolation_points
        )
        # vh.plot_interpolated_line(
        #     cropped=img, outline=front, interpolated_line=interpolated_line, size=6
        # )

        # I need to do better here on the ordering of the files. idk why
        # they os thing is looping through in such a weird order.
        np.savez(
            save_folder + f"/{i}",
            img,
            interpolated_line,
            smooth_points,
            smooth_kappa,
        )
        i += 1

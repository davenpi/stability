import os
import re
import subprocess


# First find all of the directories with videos.
rootdir = "vids"
snake_dirs = []
for file in os.listdir(rootdir):
    d = os.path.join(rootdir, file)
    if os.path.isdir(d):
        snake_dirs.append(d)

print(snake_dirs)

print("\n Now printing sub directories \n")
vid_dirs = []
# Now for each of these directories loop and get all video directories
for directory in snake_dirs:
    for f in os.listdir(directory):
        d = os.path.join(directory, f)
        if os.path.isdir(d):
            vid_dirs.append(d)

vid_dirs = [v for v in vid_dirs if "." not in v]


def get_calibration(directory):
    cal = int(directory[-2:])
    return cal


python_cal = {8: 8.7, 9: 9.9}
bi_cal = {6: 18, 8: 17}

# print(vid_dirs)
# now loop throught the tree snake vids and run the analyze_ims code on it.
for vid_dir in vid_dirs:
    if "python" in vid_dir:
        cal_dict = python_cal
    else:
        cal_dict = bi_cal
    cal = get_calibration(directory=vid_dir)
    px_per_cm = cal_dict[cal]
    with open(vid_dir + "/reaching_num.txt") as f:
        reaching_num = f.readlines()[0]

    subprocess.run(
        [
            "python3",
            "new_analyze_ims.py",
            "-p",
            vid_dir + "/full_processed",
            "-ppcm",
            f"{px_per_cm}",
            "-n",
            f"{reaching_num}",
        ]
    )
    subprocess.run(
        [
            "python3",
            "new_post_process.py",
            "-p",
            vid_dir + "/full_processed/extracted_data_signed",
        ]
    )

    print(f"Made plots for {vid_dir}")

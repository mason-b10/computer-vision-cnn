import glob
import os
from statistics import variance
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """

    numPixels = 0
    for dir in os.listdir(dir_name):
        if dir == ".DS_Store":
            continue
        for subject in os.listdir(os.path.join(dir_name, dir)):
            if subject == ".DS_Store":
                continue
            for image in os.listdir(os.path.join(dir_name, dir, subject)):
                path = os.path.join(dir_name, dir, subject, image)
                curr_img = Image.open(path).convert("L")
                curr_img = np.array(curr_img).flatten()
                curr_img = curr_img / 255.0
                numPixels += curr_img.size
                if mean == None:
                    mean = np.sum(curr_img)
                else:
                    mean += np.sum(curr_img)

    mean = (1 / numPixels) * mean

    for dir in os.listdir(dir_name):
        if dir == ".DS_Store":
            continue
        for subject in os.listdir(os.path.join(dir_name, dir)):
            if subject == ".DS_Store":
                continue
            for image in os.listdir(os.path.join(dir_name, dir, subject)):
                path = os.path.join(dir_name, dir, subject, image)
                curr_img = Image.open(path).convert("L")
                curr_img = np.array(curr_img).flatten()
                curr_img = curr_img / 255.0
                if std == None:
                    std = np.sum(np.power(curr_img - mean, 2))
                else:
                    std += np.sum(np.power(curr_img - mean, 2))

    std = (1 / (numPixels - 1)) * std
    std = np.sqrt(std)

    return mean, std

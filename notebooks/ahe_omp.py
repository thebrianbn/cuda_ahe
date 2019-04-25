# Author: Brian Nguyen, Daniel Marzec

import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
import numpy as np
from copy import copy
from datetime import datetime
from itertools import repeat
from helper import window_hist, rgb2gray


def adaptive_hist_eq_omp(img, slider_len, worker):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """
    
    # make a copy of original to replace pixels
    final_img = copy(img)
    n = len(img)
    m = len(img[0])

    gap = int(slider_len[0]// 2)  # left and right shifts
    
    # top worker handles top border
    if worker=="top":
        for i in range(gap):
            for j in range(gap, m-gap):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[:i+gap,j-gap:j+gap], center_pixel_val, None)
        for i in range(gap):
            for j in range(gap):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[:i+gap,:j+gap], center_pixel_val, None)
        for i in range(gap):
            for j in range(m-gap, m):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[:i+gap,j-gap:], center_pixel_val, None)
            
    # bottom worker handles bottom border
    elif worker=="bottom":
        for i in range(n-gap, n):
            for j in range(gap, m-gap):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[i-gap:n,j-gap:j+gap], center_pixel_val, None)
        for i in range(n-gap, n):
            for j in range(m-gap, m):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[i-gap:,j-gap:], center_pixel_val, None)
        for i in range(n-gap, n):
            for j in range(gap):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[i-gap:,:j+gap], center_pixel_val, None)

    # every worker handles left and right borders
    for i in range(gap, n-gap):
        for j in range(gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap,:j+gap], center_pixel_val, None)
    for i in range(gap, n-gap):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap,j-gap:m], center_pixel_val, None)

    # for each pixel in the center of the image, apply adaptive histogram equalization
    for i in range(gap, n - gap):
        for j in range(gap, m - gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap, j-gap:j+gap], center_pixel_val, slider_len)
    
    # only return areas the worker worked on
    if worker == "top":
        final_img = final_img[:n-gap, :].astype(int)
    elif worker == "bottom":
        final_img = final_img[gap:, :].astype(int)
    else:
        final_img = final_img[gap:n-gap, :].astype(int)

    return final_img


if __name__ == "__main__":

    # read in the image
    img = plt.imread("test_image1.jpg")

    # convert image to grayscale and round pixel values
    gray = rgb2gray(img)
    clean_image = np.matrix.round(gray).astype(int)

    n = len(clean_image)
    m = len(clean_image[0])

    # parameters for parallelization and AHE
    window_len = (15,15)
    gap = window_len[0] // 2
    n_processes = 100
    data_per = n//n_processes

    # list for worker types
    worker_type = ["top"]
    for i in range(n_processes-2):
        worker_type.append("middle")
    worker_type.append("bottom")

    # data for each worker to compute
    splits = [clean_image[:data_per+gap,:]]
    for i in range(1,n_processes-1):
        splits.append(copy(clean_image[(i*data_per)-gap:((i+1)*data_per)+gap, :]))
    splits.append(clean_image[((n_processes-1)*data_per)-gap:, :])

    start = datetime.now()
    # assign each worker a job and start them
    with Pool(processes = n_processes) as pool:
        results = pool.starmap(adaptive_hist_eq_omp, zip(splits, repeat(window_len), worker_type))
    end = datetime.now()

    # concatenate output of all workers
    final_img = np.concatenate(results, axis=0)
    print("Time taken for OMP implementation: ", end-start)

    # display output image
    plt.imshow(final_img, cmap=plt.get_cmap('gray'))
    plt.show()

    # save the image matrix for comparison
    np.savetxt("final_image_omp.txt", final_img)

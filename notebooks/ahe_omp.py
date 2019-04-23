from collections import OrderedDict
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
import numpy as np
from copy import copy
from datetime import datetime
from itertools import product, repeat

def rgb2gray(rgb):
    """ Convert an RGB image to grayscale. """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def window_hist(img, center_pixel_val, slider_len):
    """ Calculate new pixel value for the center pixel
    in an image window for adaptive histogram equalization. """
    
    pixel_freq = {}
    pdf = {}
    cdf = {}
    
    if slider_len is not None:
        pixel_count = slider_len[0] * slider_len[1]
        slider_len = (slider_len[0]-1, slider_len[1]-1)
    else:
        pixel_count = len(img) * len(img[0])
        slider_len = (len(img[0]), len(img))
    
    # for each pixel in the window update pixel frequency
    for i in range(slider_len[1]):
        for j in range(slider_len[0]):
            pixel_val = img[i, j]
            if pixel_val in pixel_freq:
                pixel_freq[pixel_val] += 1
            else:
                pixel_freq[pixel_val] = 1
                
    # for each pixel value, calculate its probability
    for pixel_val, freq in pixel_freq.items():
        pdf[pixel_val] = freq / pixel_count
    
    # order the pdf in order to calculate cdf
    pdf = OrderedDict(sorted(pdf.items(), key=lambda t: t[0]))
    
    # for each pixel value, update cdf
    prev = 0
    for pixel_val, prob in pdf.items():
        cdf[pixel_val] = prev + pdf[pixel_val]
        prev = cdf[pixel_val]
        cdf[pixel_val] = round(cdf[pixel_val] * 250)
        
        # once the cdf reaches the target pixel, no need to continue
        if pixel_val == center_pixel_val:
            break
        
    return cdf[center_pixel_val]


def adaptive_hist_eq_omp(img, slider_len, worker):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """
    
    # make a copy of original to replace pixels
    final_img = copy(img)
    n = len(img)
    m = len(img[0])

    gap = int(slider_len[0]// 2)  # left and right shifts
    
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
    window_len = (31,31)
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
    print("Time taken for OMP implementation: %f" % int(end-start))

    # display output image
    plt.imshow(final_img, cmap=plt.get_cmap('gray'))
    plt.show()

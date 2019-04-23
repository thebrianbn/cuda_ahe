from collections import OrderedDict
import matplotlib.pyplot as plt
from multiprocessing import Process
import numpy as np
from copy import copy
from datetime import datetime
from helper import window_hist, rgb2gray


def hist_eq(img, intensity_val):
    """ Normal image histogram equalization for improved contrast. """
    
    n = len(img)
    m = len(img[0])
    
    final_img = copy(img)
    pixel_freq = {}
    pdf = {}
    cdf = {}
    pixel_count = n * m    
    
    # for each pixel in the image update pixel frequency
    for i in range(n):
        for j in range(m):
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
        cdf[pixel_val] = round(cdf[pixel_val] * intensity_val)
    
    # update all pixels
    for i in range(n):
        for j in range(m):
            final_img[i, j] = cdf[img[i, j]]
    
    return final_img.astype(int)


def adaptive_hist_eq(img, slider_len):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """
    
    # make a copy of original to replace pixels
    final_img = copy(img)
    n = len(img)
    m = len(img[0])

    gap = int(slider_len[0]// 2)  # left and right shifts
    
    for i in range(gap):
        for j in range(gap, m-gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[:i+gap,j-gap:j+gap], center_pixel_val, None)
            
    for i in range(n-gap, n):
        for j in range(gap, m-gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:n,j-gap:j+gap], center_pixel_val, None)
            
    for i in range(gap, n-gap):
        for j in range(gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap,:j+gap], center_pixel_val, None)
            
    for i in range(gap, n-gap):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap,j-gap:m], center_pixel_val, None)
            
    for i in range(gap):
        for j in range(gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[:i+gap,:j+gap], center_pixel_val, None)
    
    for i in range(n-gap, n):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:,j-gap:], center_pixel_val, None)
            
    for i in range(n-gap, n):
        for j in range(gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:,:j+gap], center_pixel_val, None)
            
    for i in range(gap):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[:i+gap,j-gap:], center_pixel_val, None)
    
    # for each pixel in the center of the image, apply adaptive histogram equalization
    for i in range(gap, n - gap):
        for j in range(gap, m - gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap, j-gap:j+gap], center_pixel_val, slider_len)

    return final_img.astype(int)


if __name__ == "__main__":

    # read in the image
    img = plt.imread("test_image3.jpeg")

    # convert image to grayscale and round pixel values
    gray = rgb2gray(img)
    clean_image = np.matrix.round(gray).astype(int)

    n = len(clean_image)
    m = len(clean_image[0])

    # parameters for parallelization and AHE
    window_len = (31,31)
    gap = window_len[0] // 2

    # run the algorithm and time it
    start = datetime.now()
    final_img = adaptive_hist_eq(clean_image, (31,31))
    end = datetime.now()

    print("Time taken for serial implementation: ", end-start)

    # display output image
    plt.imshow(final_img, cmap=plt.get_cmap('gray'))
    plt.show()

    # save the image matrix for comparison
    np.savetxt("final_image_serial.txt", final_img)

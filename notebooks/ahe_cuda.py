import matplotlib.pyplot as plt
import numba
from numba import cuda
import numpy as np
from copy import copy
from datetime import datetime
from itertools import repeat

def rgb2gray(rgb):
    """ Convert an RGB image to grayscale. """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

@cuda.jit
def window_hist(img, center_pixel_val, slider_len, final_img, index):
    """ Calculate new pixel value for the center pixel
    in an image window for adaptive histogram equalization. """
    
    pixel_freq = cuda.local.array(shape=(255), dtype=numba.int32)
    pdf = cuda.local.array(shape=(255), dtype=numba.int32)
    cdf = cuda.local.array(shape=(255), dtype=numba.int32)
    
    if slider_len != 0:
        pixel_count = slider_len[0] * slider_len[1]
        slider_len_x = slider_len[0]
        slider_len_y = slider_len[1]
    else:
        pixel_count = len(img) * len(img[0])
        slider_len_x = len(img)
        slider_len_y = len(img[0])
    
    # for each pixel in the window update pixel frequency
    for i in range(slider_len_x):
        for j in range(slider_len_y):
            pixel_val = img[i, j]
            pixel_freq[pixel_val] += 1

    # for each pixel value, calculate its probability
    for pixel_val in range(255):
        pdf[pixel_val] = pixel_freq[pixel_val] / pixel_count
    
    # for each pixel value, update cdf
    prev = 0
    for pixel_val in range(255):
        prob = pdf[pixel_val]
        cdf[pixel_val] = prev + prob
        prev = cdf[pixel_val]
        cdf[pixel_val] = round(cdf[pixel_val] * 250)
        
        # once the cdf reaches the target pixel, no need to continue
        if pixel_val == center_pixel_val:
            break


def adaptive_hist_eq_cuda(img, slider_len):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """
    
    # make a copy of original to replace pixels
    final_img = copy(img)
    n = len(img)
    m = len(img[0])

    print("test")

    gap = int(slider_len[0]// 2)  # left and right shifts

    for i in range(gap):
        for j in range(gap, m-gap):
            center_pixel_val = img[i, j]
            window_hist(final_img, center_pixel_val, 0, final_img, [i,j])
            
    for i in range(n-gap, n):
        for j in range(gap, m-gap):
            center_pixel_val = img[i, j]
            window_hist(img[i-gap:n,j-gap:j+gap], center_pixel_val, None, final_img, [i,j])
            
    for i in range(gap, n-gap):
        for j in range(gap):
            center_pixel_val = img[i, j]
            window_hist(img[i-gap:i+gap,:j+gap], center_pixel_val, None, final_img, (i,j))
            
    for i in range(gap, n-gap):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            window_hist(img[i-gap:i+gap,j-gap:m], center_pixel_val, None, final_img, (i,j))
            
    for i in range(gap):
        for j in range(gap):
            center_pixel_val = img[i, j]
            window_hist(img[:i+gap,:j+gap], center_pixel_val, None, final_img, (i,j))
    
    for i in range(n-gap, n):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            window_hist(img[i-gap:,j-gap:], center_pixel_val, None, final_img, (i,j))
            
    for i in range(n-gap, n):
        for j in range(gap):
            center_pixel_val = img[i, j]
            window_hist(img[i-gap:,:j+gap], center_pixel_val, None, final_img, (i,j))
            
    for i in range(gap):
        for j in range(m-gap, m):
            center_pixel_val = img[i, j]
            window_hist(img[:i+gap,j-gap:], center_pixel_val, None, final_img, (i,j))
    
    # for each pixel in the center of the image, apply adaptive histogram equalization
    for i in range(gap, n - gap):
        for j in range(gap, m - gap):
            center_pixel_val = img[i, j]
            window_hist(img[i-gap:i+gap, j-gap:j+gap], center_pixel_val, slider_len, final_img, (i,j))

    return final_img.astype(int)


if __name__ == "__main__":

	# original image
	img = plt.imread("test_image3.jpeg")
	gray = rgb2gray(img)
	clean_image = np.ascontiguousarray(np.matrix.round(gray).astype(int), dtype=np.int32)
	final_image = adaptive_hist_eq_cuda(clean_image, (3, 3))
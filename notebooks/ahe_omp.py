from collections import OrderedDict
import matplotlib.pyplot as plt
from multiprocessing import Process
import numpy as np
from copy import copy
from datetime import datetime

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


def adaptive_hist_eq(img, slider_len, shared_mem=None):
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

    print("test")
    
    # for each pixel in the center of the image, apply adaptive histogram equalization
    for i in range(gap, n - gap):
        for j in range(gap, m - gap):
            center_pixel_val = img[i, j]
            final_img[i, j] = window_hist(img[i-gap:i+gap, j-gap:j+gap], center_pixel_val, slider_len)
    print(final_img.astype(int))
    if shared_mem is not None:
        shared_mem.append(final_img.astype(int))
        return

    return final_img.astype(int)

if __name__ == "__main__":

    img = plt.imread("test_image3.jpeg")
    gray = rgb2gray(img)
    clean_image = np.matrix.round(gray).astype(int)

    n_processes = 5
data_per = 225//5
splits = []
processes = []
shared_mem = []
for i in range(n_processes):
    splits.append(copy(clean_image[i*data_per:(i+1)*data_per, :]))
    
for i in range(n_processes):
    process = Process(target=adaptive_hist_eq, args=(splits[i], (10,10), shared_mem))
    processes.append(process)
    
    process.start()

for process in processes:
    process.join()

print(shared_mem)
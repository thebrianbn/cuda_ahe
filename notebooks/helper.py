# Author: Brian Nguyen, Daniel Marzec

from collections import OrderedDict
import numpy as np


def rgb2gray(rgb):
    """ Convert an RGB image to grayscale. """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def window_hist(img, center_pixel_val, slider_len):
    """ Calculate new pixel value for the center pixel
    in an image window for adaptive histogram equalization. """

    # dictionaries to keep track of frequencies and probabilities
    pixel_freq = {}
    pdf = {}
    cdf = {}

    # if the slider length is not given, this algorithm is run on the whole
    # image
    if slider_len is not None:
        pixel_count = slider_len[0] * slider_len[1]
        slider_len = (slider_len[0]-1, slider_len[1]-1)
    else:
        pixel_count = len(img) * len(img[0])
        slider_len = (len(img), len(img[0]))

    # for each pixel in the window update pixel frequency
    for i in range(slider_len[0]):
        for j in range(slider_len[1]):
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
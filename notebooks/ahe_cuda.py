import matplotlib.pyplot as plt
from helper import rgb2gray
import pycuda.autoinit
import pycuda.driver
import pycuda.compiler
import numpy as np
from copy import copy
from datetime import datetime
from itertools import repeat

def adaptive_hist_eq_cuda(img, slider_len):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """
    
    # make a copy of original to replace pixels
    final_img = pycuda.driver.mem_alloc(img.nbytes)
    n = len(img)
    m = len(img[0])

    gap = int(slider_len[0]// 2)  # left and right shifts

    for i in range(gap):
        for j in range(gap, m-gap):
            center_pixel_val = np.int32(img[i, j])
            img_subset = copy(img[:i+gap,j-gap:j+gap])
            n_temp = np.int32(len(img_subset))
            m_temp = np.int32(len(img_subset[0]))
            img_temp = pycuda.driver.mem_alloc(img_subset.nbytes)
            pycuda.driver.memcpy_htod(img_temp, img_subset)
            window_hist(img_temp, n_temp, m_temp, center_pixel_val, np.int32(0), np.int32(0), final_img, np.int32(i), np.int32(j), block=(32,32,1), grid=(2,2))
            
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

    	source_module = pycuda.compiler.SourceModule \
		(
		"""
		__global__ void window_hist( 
		    unsigned int* img , int n , 
		    int m , int center_pixel_val , 
		    int slider_X, int slider_Y ,
		    unsigned int* final_img , 
		    int index_X, 
		    int index_Y)
		{


		    int pixel_freq[255];
		    int pdf[255];
		    int cdf[255];

		    int pixel_count;

		    if(slider_X != 0 && slider_Y){
		        pixel_count = slider_X * slider_Y;
		    }
		    else{
		        pixel_count = n * m;
		        slider_X = n;
		        slider_Y = m;
		    }
		    for(int i =0; i<slider_X; i++){
		        for(int j = 0; i<slider_Y; j++){
		            pixel_freq[ img[i,j] ]++;
		        }
		    }

		    for(int k = 0; k<255; k++){
		        pdf[ k ] = pixel_freq[k] / pixel_count;
		    }

		    int prev = 0;
		    for(int l = 0; l<255 ; l++){
		        cdf[l] = prev + pdf[l];
		        prev = cdf[l];
		        int temp = (cdf[l] * 250);
		        //int temp1 = (int)(( temp - floor(temp) > 0.5 ) ? ceil(temp) : floor(temp));
		        cdf[l] = temp;

		        if( l == center_pixel_val){
		            break;
		        }
		            

		    final_img[index_X*n + index_Y] = cdf[center_pixel_val];
		    }
		}
		"""
		)

		img = plt.imread("test_image3.jpeg")
		gray = rgb2gray(img)
		clean_image = np.matrix.round(gray).astype(np.int32)
		plt.figure(figsize=(13,7))

		window_hist = source_module.get_function("window_hist")

		final_img = adaptive_hist_eq_cuda(clean_image, (1, 1))
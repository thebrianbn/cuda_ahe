from mpi4py import MPI
import numpy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from copy import copy

################################################################################################
######## Helper Functions : Begin ########

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
        pixel_count = len(img[0]) * len(img[:,0])
        slider_len = (len(img[0]), len(img[:,0]))
    
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

def adaptive_hist_eq_mpi(img, slider_len, worker):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """
    
    # make a copy of original to replace pixels
    final_img = copy(img)
    n = len(img[0])
    m = len(img[:,0])

    gap = int(slider_len[0]// 2)  # left and right shifts
    
    if worker=="top":
    
        for i in range(gap):
            for j in range(gap, m-gap):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[:i+gap,j-gap:j+gap], center_pixel_val, None)
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
        for i in range(gap):
            for j in range(m-gap, m):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[:i+gap,j-gap:], center_pixel_val, None)
            
    elif worker=="bottom":
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
        for i in range(n-gap, n):
            for j in range(m-gap, m):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[i-gap:,j-gap:], center_pixel_val, None)
        for i in range(n-gap, n):
            for j in range(gap):
                center_pixel_val = img[i, j]
                final_img[i, j] = window_hist(img[i-gap:,:j+gap], center_pixel_val, None)
        
    elif worker=="middle":
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
            
    return final_img

######## Helper Functions : End ########
################################################################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

window_len = 31
half_window_len = ((window_len-1)//2)
image_x = 225
image_y = image_x/(size-1)

################################################################################################
######## Split Up and Send Out Initial Data : Begin ########
if rank == 0:
	#read in image, strip rgb values, convert to numpy array
	img = plt.imread("test_image3.jpeg")
	gray = rgb2gray(img)
	clean_image = np.matrix.round(gray)

	for i in range(size):
		    data_send = clean_image[  image_y*i : image_y*(i+1) , :image_x ]
    		comm.Send(data_send, dest=i)

else:
	# allocate space for incoming data
	data_recv = np.empty( (image_x,image_y) , dtype='int') 
	comm.Recv(data_recv, source=0) 

######## Split Up and Send Out Initial Data : End ########
################################################################################################
#
##
###
####
#Wait for all threads to have received data before computation
Barrier()
####
###
##
#
################################################################################################
######## Pass Necessary Data and Compute New Pixel Values : Begin ########

#comm.Sendrecv(send_data,dest=ipl, recvbuf=recv_data,source=ipr)

if rank == 0:
	#do nothing
	continue

else if rank == 1:
	bottom_row_send = data_recv[ (image_y - half_window_len ): , :image_x ]
	bottom_row_recv = np.empty( (image_x, half_window_len ) ,dtype='int' )
	#Send and receive data from rank below
	MPI.Comm.sendrecv( bottom_row_send , dest = (rank + 1) , bottom_row_recv , source= (rank+1) )
	#combine data with bottom row received
	concat_data = np.concatenate([ data_recv , bottom_row_recv ], axis=0 )
	final_image = adaptive_hist_eq_mpi(concat_data , window_len , "top" )
	final_image = final_image[ :(image_y - half_window_len) , : ]


else if rank != (size-1): 
	top_row_send = data_recv[ :half_window_len , :image_x ]
	top_row_recv = np.empty( (image_x, half_window_len ) ,dtype='int' )
	bottom_row_send = data_recv[ (image_y - half_window_len ): , :image_x ]
	bottom_row_recv = np.empty( (image_x, half_window_len ) ,dtype='int' )
	#Send and receive data from rank below
	MPI.Comm.sendrecv( top_row_send , dest = (rank - 1) , top_row_recv , source= (rank-1) )
	#Send and receive data from rank above
	MPI.Comm.sendrecv( bottom_row_send , dest = (rank + 1) , bottom_row_recv , source= (rank+1) )
	#combine data with top and bottom data received
	concat_data = np.concatenate([ top_row_recv ,data_recv , bottom_row_recv ], axis=0 )	
	final_image = adaptive_hist_eq_mpi(concat_data , window_len , "middle" )
	final_image = final_image[ half_window_len: (image_y - half_window_len) , : ]

else:
	top_row_send = data_recv[ :half_window_len , :image_x ]
	top_row_recv = np.empty( (image_x, half_window_len ) ,dtype='int' )
	#Send and receive data from rank above
	MPI.Comm.sendrecv( bottom_row_send , dest = (rank + 1) , bottom_row_recv , source= (rank+1) )
	#combine data with top row received
	concat_data = np.concatenate([ top_row_recv ,data_recv ], axis=0 )	
	final_image = adaptive_hist_eq_mpi(concat_data , window_len , "bottom" )
	final_image = final_image[ half_window_len: , : ]

######## Pass Necessary Data and Compute New Pixel Values : End ########
################################################################################################
#
##
###
####
#Wait for all threads to have received data before computation
Barrier()
####
###
##
#
################################################################################################
######## Send Data Back to Root and Combine : Begin ########
if rank != 0:
    comm.Send(final_image, dest=0)
else:
	# allocate space for incoming data
	receive_list = []
	for i in range(size):
		final_data_recv = np.empty( (image_x,image_y) , dtype='int') 
		comm.Recv(final_data_recv, source=i)
		receive_list.append(final_data_recv)
	
	output_image = np.concatenate( receive_list , dtype='int') 
	#output image
	scipy.misc.imsave( "output_image.jpg", output_image)

######## Send Data Back to Root and Combine : End ########





# Author: Brian Nguyen, Daniel Marzec

import sys
from mpi4py import MPI
import numpy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from copy import copy
from helper import window_hist, rgb2gray
from datetime import datetime
from multiprocessing import Pool, current_process
from itertools import repeat

################################################################################################
######## Helper Functions : Begin ########

def adaptive_hist_eq_mpi(img, slider_len, worker):
    """ Apply sliding window adaptive histogram equalization to an image
    for improved local contrast. """

    # make a copy of original to replace pixels
    final_img = copy(img)
    n = len(img)
    m = len(img[0])

    gap = slider_len[0]// 2  # left and right shifts 

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

    return final_img.astype(int)

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

######## Helper Functions : End ########
################################################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# parameters for parallelization and AHE
window_len = (3, 3)
half_window_len = ((window_len[0])//2)
image_x = 225
n_processes = 2  # for OMP workers
gap = window_len[0] // 2

# handle any extra rows by assigning it to last worker
if image_x % (size-1) != 0 and rank == (size-1):
    image_y = image_x // (size-1) + image_x % (size-1)
else:
    image_y = image_x//(size-1)
    image_y_bottom = image_x // (size-1) + image_x % (size-1)

################################################################################################
######## Split Up and Send Out Initial Data : Begin ########

# master sends out partitions of data, slaves receive the data
if rank == 0:
    # read in image, strip rgb values, convert to numpy array
    img = plt.imread("test_image3.jpeg")
    gray = rgb2gray(img)
    clean_image = np.matrix.round(gray).astype(int)
    for i in range(1, size):
        # if last worker, handle extra rows
        if i == (size - 1):
            data_send = clean_image[image_y*(i-1):, :image_x]
        else:
            data_send = clean_image[  image_y*(i-1) : image_y*(i) , :image_x ]
        comm.Send(data_send, dest=i)
        print("Master: Image partition sent to slave %d." % i)
        sys.stdout.flush()
else:
    #allocate space for incoming data
    data_recv = np.empty( (image_y, image_x) , dtype='int')
    comm.Recv(data_recv, source=0)
    print("Slave %d: Image partition received from master." % rank)
    sys.stdout.flush()

######## Split Up and Send Out Initial Data : End ########
################################################################################################
#
##
###
####
#Wait for all threads to have received data before computation
comm.barrier()
####
###
##
#
################################################################################################
######## Pass Necessary Data and Compute New Pixel Values : Begin ########

if rank == 0:
    start = datetime.now()
elif rank == 1:
    bottom_row_send = data_recv[ (image_y - half_window_len ): , :image_x ]
    bottom_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    #Send and receive data from rank below
    comm.Sendrecv( [bottom_row_send, MPI.INT] , dest=(rank + 1) , recvbuf=[bottom_row_recv, MPI.INT] , source=(rank+1) )
    #combine data with bottom row received
    concat_image = np.concatenate([ data_recv , bottom_row_recv ], axis=0 )

    n = len(concat_image)
    m = len(concat_image[0])
    data_per = n//n_processes

    # list for worker types
    worker_type = ["top"]
    for i in range(n_processes-1):
        worker_type.append("middle")

    # data for each worker to compute
    splits = [concat_image[:data_per+gap,:]]
    for i in range(1,n_processes-1):
        splits.append(copy(concat_image[(i*data_per)-gap:((i+1)*data_per)+gap, :]))
    splits.append(concat_image[((n_processes-1)*data_per)-gap:, :])

    with Pool(processes = n_processes) as pool:
        results = pool.starmap(adaptive_hist_eq_omp, zip(splits, repeat(window_len), worker_type))
    final_image = np.concatenate(results, axis=0)

    final_image = final_image[ :(image_y - half_window_len) , : ]
elif rank != (size-1):
    top_row_send = data_recv[ :half_window_len , :image_x ]
    top_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    bottom_row_send = data_recv[ (image_y - half_window_len ): , :image_x ]
    bottom_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    #Send and receive data from rank below
    comm.Sendrecv( [top_row_send, MPI.INT] , dest=(rank - 1) , recvbuf=[top_row_recv, MPI.INT] , source=(rank-1) )
    #Send and receive data from rank above
    comm.Sendrecv( [bottom_row_send, MPI.INT] , dest=(rank + 1) , recvbuf=[bottom_row_recv, MPI.INT] , source=(rank+1) )
    #combine data with top and bottom data received
    concat_data = np.concatenate([ top_row_recv ,data_recv , bottom_row_recv ], axis=0 )
    final_image = adaptive_hist_eq_mpi(concat_data , window_len , "middle" )
    final_image = final_image[ half_window_len: (image_y - half_window_len) , : ]
else:
    top_row_send = data_recv[ :half_window_len , :image_x ]
    top_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    #Send and receive data from rank above
    comm.Sendrecv( [top_row_send, MPI.INT] , dest=(rank - 1) , recvbuf=[top_row_recv, MPI.INT] , source=(rank-1) )
    #combine data with top row received
    concat_data = np.concatenate([ top_row_recv ,data_recv ], axis=0 )
    final_image = adaptive_hist_eq_mpi(concat_data , window_len , "bottom" )
    final_image = final_image[ half_window_len: , : ]

if rank != 0:
    print("Slave %d: Adaptive Histogram Equalization finished on image partition." % rank)
    sys.stdout.flush()

######## Pass Necessary Data and Compute New Pixel Values : End ########
################################################################################################
#
##
###
####
#Wait for all threads to have received data before computation
comm.barrier()
####
###
##
#
################################################################################################
######## Send Data Back to Root and Combine : Begin ########
# all slaves send their output to master
if rank != 0:
    comm.Send(final_image, dest=0)
    print("Slave %d: Image partition results sent to master." % rank)
    sys.stdout.flush()
else:
    end = datetime.now()
    # allocate space for incoming data
    receive_list = []

    # retrieve data from each slave and store in local memory
    for i in range(1,size):
        # if last worker, may need extra space for buffer
        if i == size-1:
            final_data_recv = np.empty((image_y_bottom, image_x), dtype='int')
        else:
            final_data_recv = np.empty( (image_y, image_x) , dtype='int')
        comm.Recv(final_data_recv, source=i)
        receive_list.append(final_data_recv)
        print("Master: Image partition results received from slave %d." % i)
        sys.stdout.flush()

    # combine all results
    final_img = np.concatenate( receive_list , axis=0).astype(int)

    print("\nTime taken for MPI implementation: ", end-start)
    sys.stdout.flush()

    # save the image matrix for comparison
    np.savetxt("final_image_mpi.txt", final_img)
######## Send Data Back to Root and Combine : End ########

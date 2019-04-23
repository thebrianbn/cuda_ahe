import sys
from mpi4py import MPI
import numpy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from copy import copy
from helper import window_hist, rgb2gray

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

######## Helper Functions : End ########
################################################################################################
print("Start of Adaptive Histogram Equalization")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# parameters for parallelization and AHE
window_len = (30, 30)
half_window_len = ((window_len[0])//2)
image_x = 225
image_y = image_x//(size-1)

################################################################################################
######## Split Up and Send Out Initial Data : Begin ########

# master sends out partitions of data, slaves receive the data
if rank == 0:
    # read in image, strip rgb values, convert to numpy array
    img = plt.imread("test_image3.jpeg")
    gray = rgb2gray(img)
    clean_image = np.matrix.round(gray).astype(int)
    print("master sending data")
    sys.stdout.flush()
    for i in range(1, size):
        data_send = clean_image[  image_y*(i-1) : image_y*(i) , :image_x ]
        comm.Send(data_send, dest=i)
    print("all data sent")
    sys.stdout.flush()
else:
    #allocate space for incoming data
    print("receiving data from master")
    sys.stdout.flush()
    data_recv = np.empty( (image_y, image_x) , dtype='int')
    comm.Recv(data_recv, source=0)
    print("data received from master")
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
    print("start rank 1")
    sys.stdout.flush()
    print("received data for rank", rank, len(data_recv), len(data_recv[0]))
    sys.stdout.flush()
    bottom_row_send = data_recv[ (image_y - half_window_len ): , :image_x ]
    bottom_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    #Send and receive data from rank below
    comm.Sendrecv( [bottom_row_send, MPI.INT] , dest=(rank + 1) , recvbuf=[bottom_row_recv, MPI.INT] , source=(rank+1) )
    #combine data with bottom row received
    concat_data = np.concatenate([ data_recv , bottom_row_recv ], axis=0 )
    final_image = adaptive_hist_eq_mpi(concat_data , window_len , "top" )
    final_image = final_image[ :(image_y - half_window_len) , : ]
    print("finish rank 1")
    sys.stdout.flush()
elif rank != (size-1):
    print("start middle workers")
    sys.stdout.flush()
    print("receved data for rank", rank, len(data_recv), len(data_recv[0]))
    sys.stdout.flush()
    top_row_send = data_recv[ :half_window_len , :image_x ]
    top_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    bottom_row_send = data_recv[ (image_y - half_window_len ): , :image_x ]
    bottom_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    #Send and receive data from rank below
    print( "Size of top_row_send : " , len(top_row_send), len(top_row_send[0]))
    print( "Size of top_row_recv : " , len(bottom_row_recv), len(bottom_row_recv[0]))
    sys.stdout.flush()
    comm.Sendrecv( [top_row_send, MPI.INT] , dest=(rank - 1) , recvbuf=[top_row_recv, MPI.INT] , source=(rank-1) )
    #Send and receive data from rank above
    comm.Sendrecv( [bottom_row_send, MPI.INT] , dest=(rank + 1) , recvbuf=[bottom_row_recv, MPI.INT] , source=(rank+1) )
    #combine data with top and bottom data received
    concat_data = np.concatenate([ top_row_recv ,data_recv , bottom_row_recv ], axis=0 )
    final_image = adaptive_hist_eq_mpi(concat_data , window_len , "middle" )
    final_image = final_image[ half_window_len: (image_y - half_window_len) , : ]
    print("finish middle worker")
    sys.stdout.flush()
else:
    print("start last worker")
    sys.stdout.flush()
    print("data received for rank", rank, len(data_recv), len(data_recv[0]))
    sys.stdout.flush()
    top_row_send = data_recv[ :half_window_len , :image_x ]
    top_row_recv = np.empty( (half_window_len, image_x ) ,dtype='int' )
    #Send and receive data from rank above
    comm.Sendrecv( [top_row_send, MPI.INT] , dest=(rank - 1) , recvbuf=[top_row_recv, MPI.INT] , source=(rank-1) )
    #combine data with top row received
    concat_data = np.concatenate([ top_row_recv ,data_recv ], axis=0 )
    final_image = adaptive_hist_eq_mpi(concat_data , window_len , "bottom" )
    final_image = final_image[ half_window_len: , : ]
    print("finish last worker")
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
else:
    end = datetime.now()
    # allocate space for incoming data
    receive_list = []

    # retrieve data from each slave and store in local memory
    for i in range(1,size):
        final_data_recv = np.empty( (image_y, image_x) , dtype='int')
        comm.Recv(final_data_recv, source=i)
        receive_list.append(final_data_recv)

    # combine all results
    final_img = np.concatenate( receive_list , axis=0).astype(int)

    print("Time taken for MPI implementation: ", end-start)

    # save the image matrix for comparison
    np.savetxt("final_image_mpi.txt", final_img)
######## Send Data Back to Root and Combine : End ########

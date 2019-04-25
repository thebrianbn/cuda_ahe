# Author: Brian Nguyen, Daniel Marzec

import numpy as np

if __name__ == "__main__":

	# load in numpy matrices
	serial = np.loadtxt("final_image_serial1.txt")
	omp = np.loadtxt("final_image_omp1.txt")
	mpi = np.loadtxt("final_image_mpi1.txt")

	print(mpi)

	n = len(serial)
	m = len(serial[0])

	omp_fail_counter = 0
	mpi_fail_counter = 0
	for i in range(n):
		for j in range(m):
			if serial[i, j] != omp[i, j]:
				omp_fail_counter += 1
			if serial[i, j] != mpi[i, j]:
				mpi_fail_counter += 1

	print("OMP Fail Count: %d" % omp_fail_counter)
	print("MPI Fail Count: %d" % mpi_fail_counter)

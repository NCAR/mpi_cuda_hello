# mpi_cuda_hello
A test program to ensure basic CUDA and CUDA-aware MPI functionality
***
This test program is expected to run with a 1:1 mapping between MPI ranks and GPU devices. Each rank will perform a small calculation on its assigned GPU, and then MPI collectives are used to ensure that each rank calculated the same result. If the message printed at the end of execution is 
"Hello World!"
then the run successfully verifies basic functionality of the GPUs and CUDA-aware MPI

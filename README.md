# mpi_cuda_hello
***
A test program to ensure basic CUDA and CUDA-aware MPI functionality
***
This test program is expected to run with a 1:1 mapping between MPI ranks and GPU devices. Each rank will perform a small calculation on its assigned GPU, and then MPI collectives are used to ensure that each rank calculated the same result. If the message printed at the end of execution is 
"Hello World!"
then the run successfully verifies basic functionality of the GPUs and CUDA-aware MPI
***
## Example Output
```
----- ----- -----
Using 8 MPI Ranks and GPUs
----- ----- -----
Message before GPU computation: xxxxxxxxxxxx
----- ----- -----
rank 0 on host gu0017, GPU 0
rank 1 on host gu0017, GPU 1
rank 2 on host gu0017, GPU 2
rank 3 on host gu0017, GPU 3
rank 4 on host gu0018, GPU 0
rank 5 on host gu0018, GPU 1
rank 6 on host gu0018, GPU 2
rank 7 on host gu0018, GPU 3
----- ----- -----
 Message after GPU computation: Hello World!
----- ----- -----
```

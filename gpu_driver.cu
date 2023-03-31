#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>
#include <mpi.h>
#include "hello.h"
#include "sizedefs.h"

// compute a message on each GPU and use MPI collectives to calculate an agreed message
int gpu_driver(char* data, int gpu_per_node, int comm_rank, int comm_size, MPI_Comm comm, 
               int *dev_ids, char *hostnames,char *output)
{
   // data on hosts, ranks, and devices
   int dev_id = comm_rank % gpu_per_node; 
   char hostname[HN_SIZE];     
   gethostname(hostname, HN_SIZE);
        
   // temporary arrays for hello world computation
   char *min_data = new char [SIZE];
   char *max_data = new char [SIZE];

   //data that will live in device memory
   char *d_data;
   char *d_max_data;
   char *d_min_data;

   // set CUDA device and gather info about device ids and hostname for each rank
   cudaSetDevice(dev_id);
   MPI_Gather(&dev_id, 1, MPI_INT, dev_ids, 1, MPI_INT, 0, comm);
   MPI_Gather(hostname, HN_SIZE, MPI_CHAR, hostnames, HN_SIZE, MPI_CHAR, 0, comm);

   // allocate device arrays
   cudaMalloc((void**)&d_data, SIZE*sizeof(char));
   cudaMalloc((void**)&d_max_data,SIZE*sizeof(char));
   cudaMalloc((void**)&d_min_data,SIZE*sizeof(char));
   
   //copy input data to device
   cudaMemcpy(d_data, data, SIZE, cudaMemcpyHostToDevice);

   // Compute greeting messages on GPUs
   hello_gpu <<<SIZE, 1>>> (d_data, SIZE);

   // use reductions to calculate single message by all GPUs
   MPI_Reduce(d_data, d_max_data, SIZE, MPI_CHAR, MPI_MAX, 0, comm);
   MPI_Reduce(d_data, d_min_data, SIZE, MPI_CHAR, MPI_MIN, 0, comm);

   // copy data back to host, and compute output
   cudaMemcpy(max_data, d_max_data, SIZE, cudaMemcpyDeviceToHost);
   cudaMemcpy(min_data, d_min_data, SIZE, cudaMemcpyDeviceToHost);
   for (int i=0; i<SIZE; i++){
      output[i] = min_data[i] == max_data[i] ? min_data[i] : 'x';
   }

   // clean up memory on host and device
   cudaFree(d_data);
   cudaFree(d_max_data);
   cudaFree(d_min_data);
   delete [] min_data;
   delete [] max_data;
   return(0);
}

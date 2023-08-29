#include <iostream>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sched.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <vector>
#include <tuple>
#include <iomanip>
#include "gpu_driver.h"
#include "hello.h"
#include "sizedefs.h"

// compute a message on each GPU and use MPI collectives to calculate an agreed message
int gpu_driver(char* data, int gpu_per_node, int comm_rank, int comm_size, MPI_Comm comm, 
               int *dev_ids, char *dev_uuids, char *hostnames, int *cpu_ids, char *output)
{
   // Assume CUDA_VISIBLE_DEVICES is set to
   // assign one GPU device for each MPI rank
   int dev_id=0;
   cudaSetDevice(dev_id);
   cudaDeviceProp *dev_prop = new cudaDeviceProp();
   cudaGetDeviceProperties(dev_prop, dev_id);
   char *dev_uuid = uuid_to_str(dev_prop);
   char hostname[HN_LEN];     
   gethostname(hostname, HN_LEN);
   int cpu_id = sched_getcpu();
   // temporary arrays for hello world computation
   char *min_data = new char [SIZE];
   char *max_data = new char [SIZE];

   //data that will live in device memory
   char *d_data;
   char *d_max_data;
   char *d_min_data;

   // set CUDA device and gather info about device ids and hostname for each rank
   //cudaSetDevice(dev_id);
   MPI_Gather(&dev_id, 1, MPI_INT, dev_ids, 1, MPI_INT, 0, comm);
   MPI_Gather(dev_uuid, UUID_LEN, MPI_CHAR, dev_uuids, UUID_LEN, MPI_CHAR, 0, comm);
   MPI_Gather(hostname, HN_LEN, MPI_CHAR, hostnames, HN_LEN, MPI_CHAR, 0, comm);
   MPI_Gather(&cpu_id, 1, MPI_INT, cpu_ids, 1, MPI_INT, 0, comm);

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

// utility funtion to convert a device UUID to a string in nvidia-smi format
char* uuid_to_str(const cudaDeviceProp *dev_prop){
   std::ostringstream uuid_ostr;
   std::vector<std::tuple<int, int> > r = {{0,4}, {4,6}, {6,8}, {8,10}, {10,16}};
   uuid_ostr << "GPU";
   for (auto t : r){
      uuid_ostr << "-";
      for (int i = std::get<0>(t); i < std::get<1>(t); i++)
         uuid_ostr << std::hex << std::setfill('0') << std::setw(2) << (unsigned)(unsigned char)dev_prop->uuid.bytes[i];
   }
   std::string uuid_str = uuid_ostr.str();
   char *rstr = new char [uuid_str.length()+1];
   std::strcpy(rstr, uuid_str.c_str());
   return(rstr);
}


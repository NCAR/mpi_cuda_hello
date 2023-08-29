#ifndef _GPU_DRIVER_H_
#define _GPU_DRIVER_H_
#include<string>
#include<cuda_runtime.h>
// Main GPU entry point
int gpu_driver(char* data, int gpu_per_node, int rank, int size, MPI_Comm comm, 
               int *dev_ids, char *dev_uuids, char *hostnames, int *cpu_ids, char *output);

// utility funtion to convert a device UUID to a string in nvidia-smi format
char* uuid_to_str(const cudaDeviceProp *dev_prop);

#endif

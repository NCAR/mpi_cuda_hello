#ifndef _GPU_DRIVER_H_
#define _GPU_DRIVER_H_

int gpu_driver(char* data, int gpu_per_node, int rank, int size, MPI_Comm comm, 
               int *dev_ids, char *hostnames, char *output);

#endif

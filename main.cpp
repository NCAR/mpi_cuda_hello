/* A Hello World example using MPI + CUDA
 * ---------------------------------------------------
 * This test program is expected to run with a 1:1 
 * mapping between MPI ranks and GPU devices. Each
 * rank will perform a small calculation on its
 * assigned GPU, and then MPI collectives are used to
 * ensure that each rank calculated the same result.
 * If the message printed at the end of execution is
 * "Hello World!", then the run successfully verifies
 * basic functionality of the GPUs and CUDA-aware MPI
 * ---------------------------------------------------
 */
#include <iostream>
#include <mpi.h>
#include "gpu_driver.h"
#include "sizedefs.h"

int main(int argc, char **argv)
{
   MPI_Comm comm = MPI_COMM_WORLD;
   int comm_rank, comm_size;
   int rval;         // to store unchecked return codes
   char *data;       // input data array
   char *output;     // output data array
   int *dev_ids;     // GPU device ids for each rank
   char *hostnames;  // host names for eack rank

   // allocate and initialize input
   data = new char [SIZE];
   data[0]  =  72; data[1]  = 100; data[2]  = 106;
   data[3]  = 105; data[4]  = 107; data[5]  =  27;
   data[6]  =  81; data[7]  = 104; data[8]  = 106;
   data[9]  =  99; data[10] =  90; data[11] =  22;

   // allocate and initialize output
   // value will be overwritten on successful execution
   output = new char [SIZE];
   for (int i=0; i<SIZE; i++) output[i] = 'x';

   // initialize MPI and get basic MPI info
   rval = MPI_Init(&argc, &argv);
   rval = MPI_Comm_rank(comm, &comm_rank);
   rval = MPI_Comm_size(comm, &comm_size);
  
   // allocate storage for device ids and host names
   dev_ids = new int [comm_size];
   hostnames = new char [comm_size*HN_SIZE];

   // print comm info
   if (comm_rank == 0) std::cout << "----- ----- -----" << std::endl;
   if (comm_rank == 0) std::cout << "Using " << comm_size << " MPI Ranks and GPUs" << std::endl;
   if (comm_rank == 0) std::cout << "----- ----- -----" << std::endl;

   // print data before kernel call
   if (comm_rank == 0) std::cout << "Message before GPU computation: " << output << std::endl;

   // main computation
   rval = gpu_driver(data, 4, comm_rank, comm_size, comm, dev_ids, hostnames, output);

   // print host + GPU info
   if (comm_rank == 0) std::cout << "----- ----- -----" << std::endl;
   for (int i=0; i<comm_size; i++){
      if (comm_rank==0) std::cout << "rank " << i << " on host " << &hostnames[HN_SIZE*i] << ", GPU " << dev_ids[i] << std::endl;
   }

   // print data after kernel call
   if (comm_rank == 0) std::cout << "----- ----- -----" << std::endl;
   if (comm_rank == 0) std::cout << " Message after GPU computation: " << output << std::endl;
   if (comm_rank == 0) std::cout << "----- ----- -----" << std::endl;

   // clean up and exit
   rval = MPI_Finalize();
   delete [] data;
   delete [] dev_ids;
   delete [] hostnames;
   return rval;
}

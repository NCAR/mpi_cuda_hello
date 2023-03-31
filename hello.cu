/* hello_gpu
 * This kernel uses multiple thread blocks with a single thread *
 * each block increments it's data element by it's block index  */
__global__ void hello_gpu(char *a, int N){
   int i = blockIdx.x;
   if(i < N) a[i] = a[i] + i;
}

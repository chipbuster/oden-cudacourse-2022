//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//
// kernel routine
// 

__global__ void my_first_kernel(float *x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  x[tid] = (float) threadIdx.x;

  printf("I am thread %d\n", tid);
}


//
// main code
//

int main(int argc, char **argv)
{
  float *h_x, *d_x;
  int   nblocks, nthreads, nsize, n; 

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array

  size_t mem_amt = nsize * sizeof(float);
  //size_t mem_amt = 10'000'000'000 * sizeof(float);

  h_x = (float *)malloc(mem_amt);
  //cudaMalloc((void **)&d_x, nsize*sizeof(float));
  cudaMalloc((void **)&d_x, mem_amt);

  // execute kernel

  my_first_kernel<<<nblocks,nthreads>>>(d_x);

  // copy back results and print them out

  cudaMemcpy(h_x,d_x,mem_amt,cudaMemcpyDeviceToHost);

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  cudaFree(d_x);
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}

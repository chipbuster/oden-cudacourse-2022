//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *x, float* y)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  x[tid] += y[tid];
}


//
// main code
//

int main(int argc, const char **argv)
{
  float *d_x, *d_y;
  int   nblocks, nthreads, nsize, n; 

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 2;
  nthreads = 8;
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  size_t mem_amt = nsize * sizeof(float);

  // VLA---this is naughty but I hate mallocs
  float h_x[nsize], h_y[nsize];
  checkCudaErrors(cudaMalloc((void **)&d_x, mem_amt));
  checkCudaErrors(cudaMalloc((void **)&d_y, mem_amt));

  // Initialize host vectors
  for(int i = 0; i < nsize; ++i){
    h_x[i] = 5 * i - 3;
    h_y[i] = -2 * i + 4;
  }

  // Copy host vectors to device
  checkCudaErrors( cudaMemcpy(d_x,h_x,mem_amt, cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_y,h_y,mem_amt, cudaMemcpyHostToDevice) );

  // execute kernel
  my_first_kernel<<<nblocks,nthreads>>>(d_x, d_y);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x,mem_amt,
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<nsize; n++) printf(" n,  x  =  %d  %f \n",n,h_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(d_x));

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}

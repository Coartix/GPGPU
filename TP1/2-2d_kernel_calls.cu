#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include "kernels.h"

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

int main()
{
  int cols = 2 * 1024;
  int rows = 2 * 1024;
  // int cols=4;  // Use less elements for debug if needed
  // int rows=4;
  float *d_buffer; //< Pointer to the 1D buffer we will manipulate

  //@@ Initialize grid and block sizes for later kernel launches.
  //@@ Use as many threads as possible.
  //@@ create 2D grid and blocks, you need to find the right structure to store those sizes
  dim3 threads(16, 16);
  dim3 blocks(cols / threads.x, rows / threads.y);
  size_t pitch; //< we will store the pitch value in this variable

  // Allocate an 2D buffer with padding
  //@@ use cudaMallocPitch to allocate this buffer
  cudaMallocPitch((void **)&d_buffer, &pitch, cols * sizeof(float), rows);
  printf("Pitch d_buffer: %d\n", pitch);
  cudaCheckError();

  // The value we want our buffer to be filled with
  const float value = 5.f;

  // Initialize the buffer
  //@@ Call the fill2D kernel to fill d_buffer with `value`, see kernels.h for the API
  fill2D<<<blocks, threads>>>(d_buffer, value, cols, rows, pitch);
  // Wait for GPU to finish and check for errors
  cudaDeviceSynchronize();
  cudaCheckError();

  // Check the content of the buffer on the device
  //@@ Call the check2D kernel to control device memory content, see kernels.h for API
  check2D<<<blocks, threads>>>(d_buffer, value, cols, rows, pitch);

  // Wait for GPU to finish and check for errors
  //@@ call CUDA device synchronisation function
  cudaDeviceSynchronize();
  cudaCheckError();

  // Copy back buffer to host memory for inspection
  //@@ Allocate a buffer on the host
  float *host_buffer = (float *)malloc(rows * cols * sizeof(float));

  //@@ Copy the buffer content from device to host
  cudaMemcpy2D(host_buffer, cols * sizeof(float), d_buffer, pitch, cols * sizeof(float), rows, cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Check for errors
  float maxError = 0.0f;
  for (int i = 0; i < rows * cols; i++)
    maxError = std::fmax(maxError, std::fabs(host_buffer[i] - value));
  std::cout << "Max error: " << maxError << std::endl;
  bool noerror = (maxError < 0.0001f); // There is much smarter to do.

  // Clean up
  //@@ free d_buffer using CUDA primitives
  cudaFree(d_buffer);
  cudaCheckError();

  std::free(host_buffer);

  // Useful return value
  if (noerror)
  {
    printf("Test completed successfully.\n");
    return 0;
  }
  else
  {
    printf("WARNING there were some errors.\n");
    return 1;
  }
}

#include <iostream>
#include <math.h>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

// CUDA kernel to add elements of two arrays
__global__ void add(float *x, float *y)
{
  //@@ add a single element of x and y and store the result in y
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1 << 20; // 1M elements
  float *x, *y;

  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Launch kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaCheckError();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i] - 3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  cudaCheckError();

  if (maxError < 0.000001f)
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
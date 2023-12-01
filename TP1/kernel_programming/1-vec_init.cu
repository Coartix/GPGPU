#include <cstdio>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

// Simple vector initialization; element at index i receives value i.
// This is a good alternative to cudaMemset which write bytes only.
__global__ void vecinit(int *a, int N)
{
  //@@ int i = ???; // FIXME derive from block and thread info the index of the value to set
  //@@ if (???) // FIXME check we stay within buffer's bound
  //@@   a[i] = i; // keep this line
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    a[i] = i;
}

int main()
{

  int N = 4097;
  int threads = 128;
  int blocks = (N + threads - 1) / threads; //
  int *a;

  // The "managed" allocation below allows the same pointer to be used by
  // device and host code. Available since CUDA 6, Unified Memory is supported
  // starting with the Kepler GPU architecture (Compute Capability 3.0 or higher).
  // More details: https://devblogs.nvidia.com/unified-memory-in-cuda-6/
  cudaMallocManaged(&a, N * sizeof(int));
  vecinit<<<blocks, threads>>>(a, N);
  cudaDeviceSynchronize(); // This is still needed when using Unified Memory

  bool error = false;
  for (int i = 0; i < N; i++)
    if (error = a[i] != i)
    {
      printf("ERROR at index %d: expected %d, got %d\n", i, i, a[i]);
      break;
    }
  cudaCheckError();
  if (!error)
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

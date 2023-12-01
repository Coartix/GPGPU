#include <cstdio>
#include <numeric>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

// Computes the 1D index of a value in a contiguous buffer
// given its 2D coordinates and the width of a row.
//@@ #define IDX(row, col, width) ((???)*(???)+(???))  // FIXME

// computes c(i,j) = a(i,j) + b(i,j)
__global__ void add(int *a, int *b, int *c, int N, int M)
{
  //@@  int i=???;  // FIXME compute row the coordinates of the value
  //@@  int j=???;  // FIXME compute col the coordinates of the value
  //@@  if(??? && ???) {  // FIXME check boundaries
  //@@    int idx=IDX(i,j,M);  // keep this line
  //@@    c[idx] = a[idx] + b[idx];  // keep this line
  //@@  }
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < M)
  {
    int idx = i * M + j;
    c[idx] = a[idx] + b[idx];
  }
}

int main()
{
  // 2048 rows and cols
  int N = 2 * 1024;
  int M = 2 * 1024;
  int *a, *b, *c;
  dim3 threads(32, 32);
  dim3 blocks(N / threads.x, M / threads.y);

  // Unified memory allocation
  cudaMallocManaged(&a, N * M * sizeof(int));
  cudaMallocManaged(&b, N * M * sizeof(int));
  cudaMallocManaged(&c, N * M * sizeof(int));

  std::iota(a, a + N * M, 1);
  std::iota(b, b + N * M, 1);

  // Kernal launch
  add<<<blocks, threads>>>(a, b, c, N, M);
  cudaDeviceSynchronize();
  cudaCheckError();

  // Check the results
  bool error = false;
  for (auto i = 0; i < N * M; i++)
  {
    if (error = a[i] + b[i] != c[i])
    {
      printf("ERROR at index %d.", i);
      break;
    }
  }

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
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

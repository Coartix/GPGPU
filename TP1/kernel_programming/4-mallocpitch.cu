#include <cstdio>
#include <cstdlib>
// #include <memory>
#include <assert.h>

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(EXIT_FAILURE);                                                              \
    }                                                                                  \
  }

// Computes the pointer address of a given value in a 2D array given:
// baseAddress: the base address of the buffer
// col: the col coordinate of the value
// row: the row coordinate of the value
// pitch: the actual allocation size **in bytes** of a row plus its padding
template <typename T>
__device__ inline T *eltPtr(T *baseAddress, int col, int row, size_t pitch)
{
  //@@  return (T*)((char*)baseAddress + ??? * ??? + ??? * sizeof(???));  // FIXME
  return (T *)((char *)baseAddress + row * pitch + col * sizeof(T));
}

// Simple vector initialization; element at index i receives value i.
// This is a good alternative to cudaMemset which write bytes only
template <typename T>
__global__ void simpleInit2D(T *buffer, T value, int cols, int rows, size_t pitch)
{
  //@@    int col = ???;  // FIXME compute coordinates
  //@@    int row = ???;  // FIXME compute coordinates
  //@@    if(??? && ???) {  // FIXME check boundaries
  //@@        T* eptr = eltPtr<T>(buffer, col, row, pitch);
  //@@        *eptr = value;
  //@@    }
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols)
  {
    T *eptr = eltPtr<T>(buffer, col, row, pitch);
    *eptr = value;
  }
}

// Simply checks that all the values of a given buffer are `expectedValue`
template <typename T>
__global__ void checkOnDevice(T *buffer, T expectedValue, int cols, int rows, size_t pitch)
{
  //@@    int col = ???;  // FIXME compute coordinates
  //@@    int row = ???;  // FIXME compute coordinates
  //@@    if(??? && ???) {  // FIXME check boundaries
  //@@        T* eptr = eltPtr<T>(buffer, col, row, pitch);
  //@@        assert (*eptr == expectedValue);
  //@@    }
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < rows && col < cols)
  {
    T *eptr = eltPtr<T>(buffer, col, row, pitch);
    assert(*eptr == expectedValue);
  }
}

int main()
{
  int cols = 2 * 1024;
  int rows = 2 * 1024;
  // int cols=4;  // Use less elements for debug if needed
  // int rows=4;
  int *d_buffer;
  dim3 threads(32, 32);
  dim3 blocks((cols + threads.x - 1) / threads.x,
              (rows + threads.y - 1) / threads.y);
  size_t pitch;

  // Allocate an 2D buffer with padding
  cudaMallocPitch(&d_buffer, &pitch, cols * sizeof(int), rows);
  printf("Pitch d_buffer: %d\n", pitch);
  cudaCheckError();

  // The value we want our buffer to be filled with
  const int value = 5;

  // USING cudaMemset* FUNCTIONS IS WRONG FOR SETTING INTEGERS!!!
  // https://stackoverflow.com/questions/13387101/cudamemset-does-it-set-bytes-or-integers
  // Why do cudaMemset* functions take int values when they actually set bytes???
  // because std::memset does so… https://en.cppreference.com/w/cpp/string/byte/memset
  //   cudaMemset2D(c, pitch, value, cols * sizeof(int), rows);
  //   cudaDeviceSynchronize();
  //   cudaCheckError();

  // Initialize the buffer
  simpleInit2D<int><<<blocks, threads>>>(d_buffer, value, cols, rows, pitch);
  cudaDeviceSynchronize();
  cudaCheckError();

  // Check the content of the buffer on the device
  checkOnDevice<int><<<blocks, threads>>>(d_buffer, value, cols, rows, pitch);
  cudaDeviceSynchronize();
  cudaCheckError();

  // Copy back d_buffer to host memory for inspection
  int *host_buffer = (int *)std::malloc(rows * cols * sizeof(int));
  cudaMemcpy2D(host_buffer, cols * sizeof(int),
               d_buffer, pitch, cols * sizeof(int), rows,
               cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Check for errors
  bool error = false;
  for (int i = 0; i < rows * cols; i++)
  {
    int val_real = host_buffer[i];
    if (error = val_real != value)
    {
      printf("ERROR at index %d: expected %d but got %d.\n", i, value, val_real);
      break;
    }
  }

  // Clean up
  cudaFree(d_buffer);
  cudaCheckError();

  std::free(host_buffer);

  // Useful return value
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

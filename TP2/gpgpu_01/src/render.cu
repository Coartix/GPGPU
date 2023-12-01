#include "render.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

[[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

struct rgba8_t
{
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

__device__ rgba8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgba8_t{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgba8_t{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgba8_t{r, 255, 0, 255};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgba8_t{255, b, 0, 255};
  }
}

// Device code
__global__ void mykernel(char *buffer, int width, int height, size_t pitch)
{
  // float denum = width * width + height * height;

  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  rgba8_t *lineptr = reinterpret_cast<rgba8_t *>(reinterpret_cast<char *>(buffer) + y * pitch);
  // float    v       = (x * x + y * y) / denum;
  // uint8_t  grayv   = v * 255;

  // Mandelbrot fractal
  double mx0 = (x / static_cast<double>(width)) * 3.5 - 2.5;
  double my0 = (y / static_cast<double>(height)) * 2.0 - 1.0;
  double mx = 0.0;
  double my = 0.0;
  int iteration = 0;
  int n_iterations = 100;

  while (mx * mx + my * my < 4.0 && iteration < n_iterations)
  {
    double mxtemp = mx * mx - my * my + mx0;
    my = 2.0 * mx * my + my0;
    mx = mxtemp;
    iteration++;
  }

  // Map the iteration count to a color using the heat_lut function
  double x_normalized = static_cast<double>(iteration) / n_iterations;
  rgba8_t color = heat_lut(x_normalized);

  // Assign rgba8_t directly to lineptr
  lineptr[x] = color;
}

// Device code
// Kernel to compute the number of iterations of the fractal per pixel
__global__ void compute_iter(char *buffer, int width, int height, size_t pitch, int max_iter)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  uchar4 *lineptr = (uchar4 *)(buffer + y * pitch);

  // Initialize Mandelbrot parameters and variables
  double mx0 = (x / static_cast<double>(width)) * 3.5 - 2.5;
  double my0 = (y / static_cast<double>(height)) * 2.0 - 1.0;
  double mx = 0.0;
  double my = 0.0;
  int iteration = 0;

  // Perform the Mandelbrot iteration
  while (mx * mx + my * my < 4.0 && iteration < max_iter)
  {
    double mxtemp = mx * mx - my * my + mx0;
    my = 2.0 * mx * my + my0;
    mx = mxtemp;
    iteration++;
  }

  // Store the number of iterations in the buffer
  lineptr[x] = make_uchar4(iteration, iteration, iteration, 255);
}

// Kernel to compute the LUT (Look-Up Table)
__global__ void compute_LUT(const char *buffer, int width, int height, size_t pitch, int max_iter, rgba8_t *LUT)
{
  // Single thread for LUT computation
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    // Initialize LUT
    for (int i = 0; i <= max_iter; i++)
    {
      LUT[i] = heat_lut(static_cast<float>(i) / max_iter);
    }

    // Compute the cumulated histogram
    int *histogram = new int[max_iter + 1];
    memset(histogram, 0, sizeof(int) * (max_iter + 1));

    for (int y = 0; y < height; y++)
    {
      const uchar4 *lineptr = reinterpret_cast<const uchar4 *>(buffer + y * pitch);

      for (int x = 0; x < width; x++)
      {
        int iteration = lineptr[x].x;
        histogram[iteration]++;
      }
    }

    // Normalize the LUT based on the histogram
    for (int i = 1; i <= max_iter; i++)
    {
      histogram[i] += histogram[i - 1];
    }

    for (int i = 0; i <= max_iter; i++)
    {
      if (histogram[i] > 0)
      {
        float normalized_value = static_cast<float>(histogram[i]) / (width * height);
        LUT[i] = heat_lut(normalized_value);
      }
    }

    delete[] histogram;
  }
}

// Kernel to apply the LUT and color map to the image
__global__ void apply_LUT(char *buffer, int width, int height, size_t pitch, int max_iter, const rgba8_t *LUT)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  rgba8_t *lineptr = reinterpret_cast<rgba8_t *>(buffer + y * pitch);
  int iteration = lineptr[x].r;

  // Apply the LUT and color map to the image
  rgba8_t color = LUT[iteration];

  // Assign the color directly to lineptr
  lineptr[x] = color;
}

void render(char *hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char *devBuffer;
  size_t pitch;

  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");
  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w = std::ceil((float)width / bsize);
    int h = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    mykernel<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}

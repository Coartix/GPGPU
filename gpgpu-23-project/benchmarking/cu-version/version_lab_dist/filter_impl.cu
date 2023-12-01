#include "filter_impl.h"
#include <chrono> // chrono_listerals
#include <thread> // sleep_for

#include <cstring> // memcpy
#include <cmath>   // pow, sqrt

#include <vector>
#include <algorithm> // rotate, nth_element

#include <cstdio> // fprintf, stderr, exit

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

// RGB structure
struct rgb
{
    uint8_t r, g, b;
};

extern "C"
{

    // Back ground model (frame)
    uint8_t *back_ground_model = nullptr;

    // Frame count to know when to uptade
    int frame_count = 0;

    uint8_t *d_input;
    uint8_t *d_output;

    void initialize_cuda_memory(int size)
    {
        cudaError_t err;
        err = cudaMalloc(&d_input, size);
        CHECK_CUDA_ERROR(err);

        err = cudaMalloc(&d_output, size);
        CHECK_CUDA_ERROR(err);
    }

    void cleanup_cuda_memory()
    {
        cudaFree(d_input);
        cudaFree(d_output);
    }

    // Global history buffer
    std::vector<std::vector<std::vector<uint8_t>>> pixel_history;

    // Define the transformation matrix from RGB to XYZ

    // Declare the constant 2D array on the device for rgb_to_xyz
    __constant__ double rgb_to_xyz_device[3][3];

    // Define the XYZ tristimulus values for illuminant "D65"
    // __constant__ std::vector<double> xyz_ref_white = {0.95047, 1.0, 1.08883};
    __constant__ double xyz_ref_white_device[3];

    // Device function to initialize the constant rgb_to_xyz
    __device__ void initialize_rgb_to_xyz_device(double data[3][3])
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                rgb_to_xyz_device[i][j] = data[i][j];
            }
        }
    }

    // Device function to initialize the constant
    __device__ void initialize_xyz_ref_white_device(double x, double y, double z)
    {
        xyz_ref_white_device[0] = x;
        xyz_ref_white_device[1] = y;
        xyz_ref_white_device[2] = z;
    }

    // Function to convert RGB to LAB
    __device__ void rgb_to_lab(const rgb &color, double xyz[3], double lab[3])
    {
        // Normalize by the XYZ tristimulus values of the reference white point
        for (size_t i = 0; i < 3; ++i)
        {
            xyz[i] /= xyz_ref_white_device[i];
        }

        // Nonlinear distortion and linear transformation
        for (size_t i = 0; i < 3; ++i)
        {
            if (xyz[i] > 0.008856)
            {
                xyz[i] = std::pow(xyz[i], 1.0 / 3.0);
            }
            else
            {
                xyz[i] = 7.787 * xyz[i] + 16.0 / 116.0;
            }
        }
    }

    /*
     * Apply erosion to current frame
     *
     * Get minimun pixel value of neighbors in order to reduce noise
     * Copy buffer to temporary first to avoid propagation
     */
    __global__ void erosionKernel(uint8_t *input, uint8_t *output, int width, int height, int stride, int pixel_stride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ uint8_t shared_output[];

        int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
        int global_index = y * stride + x * pixel_stride;

        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            uint8_t min_val = 255;
            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int i = (y + ky) * stride + (x + kx) * pixel_stride;
                    min_val = min(min_val, input[i]);
                    // max_val = max(max_val, input[i]);
                }
            }

            shared_output[shared_index] = min_val; // For dilation

            __syncthreads();

            // Copy from shared memory to global memory
            output[global_index] = shared_output[shared_index];
        }
    }

    void apply_erosion(uint8_t *buffer, int width, int height, int stride, int pixel_stride)
    {
        size_t size = width * height * pixel_stride;

        cudaMemcpy(d_input, buffer, size, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // Calculate shared memory size (bytes)
        int sharedMemorySize = blockSize.x * blockSize.y * sizeof(uint8_t);

        erosionKernel<<<gridSize, blockSize, sharedMemorySize>>>(d_input, d_output, width, height, stride, pixel_stride);

        cudaMemcpy(buffer, d_output, size, cudaMemcpyDeviceToHost);
    }

    /*
     * Apply dilatation to current frame
     *
     * Get maximum pixel value of neighbors in order to reconstruct object
     * Copy buffer to temporary first to avoid propagation
     */
    __global__ void dilationKernel(uint8_t *input, uint8_t *output, int width, int height, int stride, int pixel_stride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        extern __shared__ uint8_t shared_output[];

        int shared_index = threadIdx.y * blockDim.x + threadIdx.x;
        int global_index = y * stride + x * pixel_stride;

        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            uint8_t max_val = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int i = (y + ky) * stride + (x + kx) * pixel_stride;
                    max_val = max(max_val, input[i]);
                }
            }

            shared_output[shared_index] = max_val; // For dilation

            __syncthreads();

            // Copy from shared memory to global memory
            output[global_index] = shared_output[shared_index];
        }
    }

    void apply_dilation(uint8_t *buffer, int width, int height, int stride, int pixel_stride)
    {
        size_t size = width * height * pixel_stride;

        cudaMemcpy(d_input, buffer, size, cudaMemcpyHostToDevice);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // Calculate shared memory size (bytes)
        int sharedMemorySize = blockSize.x * blockSize.y * sizeof(uint8_t);

        dilationKernel<<<gridSize, blockSize, sharedMemorySize>>>(d_input, d_output, width, height, stride, pixel_stride);

        cudaMemcpy(buffer, d_output, size, cudaMemcpyDeviceToHost);
    }

    // Check if pixel is a strong edge (above upper threshold)
    bool is_strong_edge(int x, int y, std::vector<std::vector<bool>> &strongEdges,
                        int width, int height)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return false;
        }
        return strongEdges[y][x];
    }

    /*
        Apply hysteresis to current frame
        - First pass: mark all pixels above upper threshold as strong edges
        - Second pass: mark all pixels above lower threshold as weak edges
            - If weak edge has strong edge neighbor, mark as strong edge
    */
    void hysteresis_thresholding(std::vector<std::vector<int>> &image,
                                 std::vector<std::vector<bool>> &strong_edges,
                                 std::vector<std::vector<bool>> &weak_edges,
                                 int low_threshold, int high_threshold,
                                 int width, int height)
    {

        std::vector<std::vector<bool>> strong_edges_cpy =
            std::vector<std::vector<bool>>(height, std::vector<bool>(width, false));
        // First pass
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (image[y][x] > high_threshold)
                {
                    strong_edges[y][x] = true;
                    strong_edges_cpy[y][x] = true;
                }
                else if (image[y][x] > low_threshold)
                {
                    weak_edges[y][x] = true;
                }
            }
        }

        // Second pass
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (weak_edges[y][x])
                {
                    // Check 8-connected neighbors
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0)
                                continue;
                            if (is_strong_edge(x + dx, y + dy,
                                               strong_edges_cpy, width, height))
                            {
                                strong_edges[y][x] = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Initialize history vector to capacity n
    void initialize_pixel_history(int width, int height, int n)
    {
        pixel_history.resize(height,
                             std::vector<std::vector<uint8_t>>(width, std::vector<uint8_t>(n, 0)));
    }

    // Update history, remove oldest and then add current
    void update_pixel_history(uint8_t *buffer, int width, int height,
                              int stride, int pixel_stride)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                // Assume grayscale for simplicity
                uint8_t new_value = buffer[y * stride + x * pixel_stride];

                // Update history, removing oldest, adding newest
                auto &history = pixel_history[y][x];
                std::rotate(history.begin(), history.begin() + 1,
                            history.end());
                history.back() = new_value;
            }
        }
    }

    // Apply median on specific vector
    uint8_t median_of_vector(const std::vector<uint8_t> &v)
    {
        std::vector<uint8_t> temp = v;
        size_t n = temp.size() / 2;
        std::nth_element(temp.begin(), temp.begin() + n, temp.end());
        return temp[n];
    }

    // Update background model using median
    void update_background_model_median(uint8_t *back_ground_model, int width,
                                        int height, int stride, int pixel_stride)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                back_ground_model[y * stride + x * pixel_stride] =
                    median_of_vector(pixel_history[y][x]);
            }
        }
    }

    /*
     * Kernel function to apply LAB distance to current frame
     *
     * Convert RGB to LAB
     * Calculate distance between background model and current frame
     * Normalize distance to [0, 255]
     * Update result buffer
     */
    __global__ void labDistanceKernel(uint8_t *bg_model, uint8_t *frame, uint8_t *result,
                                      int width, int height, int stride, int pixel_stride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height)
        {
            int index = y * stride + x * pixel_stride;

            rgb color_bg;
            color_bg.r = bg_model[index];
            color_bg.g = bg_model[index + 1];
            color_bg.b = bg_model[index + 2];

            rgb color_frame;
            color_frame.r = frame[index];
            color_frame.g = frame[index + 1];
            color_frame.b = frame[index + 2];

            // Convert RGB to XYZ
            double xyz_bg[3] = {0.0, 0.0, 0.0};

            for (size_t i = 0; i < 3; ++i)
            {
                xyz_bg[i] = 0.0;
                xyz_bg[i] += color_bg.r / 255.0 * rgb_to_xyz_device[i][0];
                xyz_bg[i] += color_bg.g / 255.0 * rgb_to_xyz_device[i][1];
                xyz_bg[i] += color_bg.b / 255.0 * rgb_to_xyz_device[i][2];
            }

            // Calculate LAB components
            double lab_bg[3] = {0.0, 0.0, 0.0};

            lab_bg[0] = 116.0 * xyz_bg[1] - 16.0;        // L
            lab_bg[1] = 500.0 * (xyz_bg[0] - xyz_bg[1]); // a
            lab_bg[2] = 200.0 * (xyz_bg[1] - xyz_bg[2]); // b

            // Convert RGB to XYZ
            double xyz_frame[3] = {0.0, 0.0, 0.0};

            for (size_t i = 0; i < 3; ++i)
            {
                xyz_frame[i] = 0.0;
                xyz_frame[i] += color_frame.r / 255.0 * rgb_to_xyz_device[i][0];
                xyz_frame[i] += color_frame.g / 255.0 * rgb_to_xyz_device[i][1];
                xyz_frame[i] += color_frame.b / 255.0 * rgb_to_xyz_device[i][2];
            }

            // Calculate LAB components
            double lab_frame[3] = {0.0, 0.0, 0.0};

            lab_frame[0] = 116.0 * xyz_frame[1] - 16.0;           // L
            lab_frame[1] = 500.0 * (xyz_frame[0] - xyz_frame[1]); // a
            lab_frame[2] = 200.0 * (xyz_frame[1] - xyz_frame[2]); // b

            // Convert RGB to LAB
            rgb_to_lab(color_bg, xyz_bg, lab_bg);
            rgb_to_lab(color_frame, xyz_frame, lab_frame);

            // Calculate LAB distance
            double distance = sqrt(pow(lab_bg[0] - lab_frame[0], 2) +
                                   pow(lab_bg[1] - lab_frame[1], 2) +
                                   pow(lab_bg[2] - lab_frame[2], 2));

            // Normalize distance to [0, 255]

            // uint8_t normalized = static_cast<uint8_t>(std::min(255.0, std::max(0.0, distance * 255.0 / 100.0)));
            uint8_t normalized = (distance * 255.0 / 100.0 > 255.0) ? 255 : ((distance * 255.0 / 100.0 < 0.0) ? 0 : static_cast<uint8_t>(distance * 255.0 / 100.0));

            // Update result buffer
            result[index] = normalized;
            result[index + 1] = normalized;
            result[index + 2] = normalized;
        }
    }

    // Update buffer_cpy with LAB distance parallelized
    void update_buffer_with_lab_distance(uint8_t *back_ground_model, uint8_t *buffer_cpy,
                                         int width, int height, int stride, int pixel_stride)
    {
        size_t size = width * height * pixel_stride;

        // Allocate device memory
        uint8_t *d_bg_model;
        uint8_t *d_buffer_cpy;
        uint8_t *d_result;

        cudaError_t err;

        err = cudaMalloc(&d_bg_model, size);
        CHECK_CUDA_ERROR(err);
        err = cudaMalloc(&d_buffer_cpy, size);
        CHECK_CUDA_ERROR(err);
        err = cudaMalloc(&d_result, size);
        CHECK_CUDA_ERROR(err);

        // Copy data to device
        err = cudaMemcpy(d_bg_model, back_ground_model, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy(d_buffer_cpy, buffer_cpy, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // Set the constant values for rgb_to_xyz before launching the kernel
        double rgb_to_xyz_host[3][3] = {
            {0.412453, 0.357580, 0.180423},
            {0.212671, 0.715160, 0.072169},
            {0.019334, 0.119193, 0.950227}};
        err = cudaMemcpyToSymbol(rgb_to_xyz_device, rgb_to_xyz_host, sizeof(double) * 3 * 3);
        CHECK_CUDA_ERROR(err);

        // Set the constant values before launching the kernel
        double xyz_ref_white_host[] = {0.95047, 1.0, 1.08883};
        err = cudaMemcpyToSymbol(xyz_ref_white_device, xyz_ref_white_host, sizeof(double) * 3);
        CHECK_CUDA_ERROR(err);

        // Launch kernel
        labDistanceKernel<<<gridSize, blockSize>>>(d_bg_model, d_buffer_cpy, d_result, width, height, stride, pixel_stride);

        // Copy result back to host
        err = cudaMemcpy(buffer_cpy, d_result, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);

        // Sync
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        // Free device memory
        cudaFree(d_bg_model);
        cudaFree(d_buffer_cpy);
        cudaFree(d_result);
    }

    /*
        Host function to apply filter to current frame
            Called for each frame
    */
    void filter_impl(uint8_t *buffer, int width, int height, int stride,
                     int pixel_stride)
    {
        const int n = 10; // Number of frames to keep in history

        if (back_ground_model == nullptr)
        {
            back_ground_model = new uint8_t[width * height * pixel_stride];
            std::memcpy(back_ground_model, buffer, width * height * pixel_stride);
            initialize_pixel_history(width, height, n);
            initialize_cuda_memory(width * height * pixel_stride);
        }

        // Update pixel history with new frame
        update_pixel_history(buffer, width, height, stride, pixel_stride);

        // Update background model using median
        if (++frame_count >= n)
        {
            update_background_model_median(back_ground_model, width, height,
                                           stride, pixel_stride);
        }

        uint8_t *buffer_cpy = new uint8_t[width * height * pixel_stride];
        std::memcpy(buffer_cpy, buffer, width * height * pixel_stride);

        /* CUDA
            Apply LAB distance to current frame
        */

        update_buffer_with_lab_distance(back_ground_model, buffer_cpy, width, height, stride, pixel_stride);

        /* CUDA
            Apply erosion and dilation to current frame
                - Erosion: get minimun pixel value of neighbors in order to reduce noise
                - Dilation: get maximum pixel value of neighbors in order to reconstruct object
        */

        apply_erosion(buffer_cpy, width, height, stride, pixel_stride);
        apply_dilation(buffer_cpy, width, height, stride, pixel_stride);

        /**** Convert to 2D buffer ****/

        std::vector<std::vector<int>> buffer2D(height, std::vector<int>(width));

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                buffer2D[i][j] = buffer_cpy[i * stride + j * pixel_stride];
            }
        }

        /****** Apply hysteresis thresholding ******/

        // Prepare matrices for strong and weak edges
        std::vector<std::vector<bool>> strong_edges(height, std::vector<bool>(width, false));
        std::vector<std::vector<bool>> weak_edges(height, std::vector<bool>(width, false));

        int low_threshold = 4;   // Define your low threshold
        int high_threshold = 30; // Define your high threshold

        hysteresis_thresholding(buffer2D, strong_edges, weak_edges,
                                low_threshold, high_threshold, width, height);

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int index = i * stride + j * pixel_stride;
                if (strong_edges[i][j])
                {
                    // If strong edge, set the pixel to white
                    // (or another color)
                    buffer_cpy[index] = 255;     // Red
                    buffer_cpy[index + 1] = 255; // Green
                    buffer_cpy[index + 2] = 255; // Blue
                }
                else
                {
                    // If not a strong edge, set the pixel to black
                    // (or another color)
                    buffer_cpy[index] = 0;     // Red
                    buffer_cpy[index + 1] = 0; // Green
                    buffer_cpy[index + 2] = 0; // Blue
                }
            }
        }

        // Update buffer with result
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int index = i * stride + j * pixel_stride;
                buffer[index] = std::min(255, static_cast<int>(buffer[index] +
                                                               0.5 * buffer_cpy[index]));
            }
        }

        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}

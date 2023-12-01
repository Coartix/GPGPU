#include "filter_impl.h"
#include <chrono> // chrono_listerals
#include <thread> // sleep_for

#include <cstring> // memcpy
#include <cmath>   // pow, sqrt

#include <vector>
#include <algorithm> // rotate, nth_element

#include <cstdio> // fprintf, stderr, exit

#include <uchar.h> // uchar3

#include "kernel.cu"
#include "cuda_utils.cu"

extern "C"
{
    /********************/
    /* Global variables */
    /********************/
    struct rgb
    {
        uint8_t r, g, b;
    };

    uint8_t *backg_model = nullptr;
    uint8_t *buffer_cpy = nullptr;
    int frame_count = 0;

    uint8_t *d_input;
    uint8_t *d_output;

    uchar3 *d_bg_model = nullptr;
    uchar3 *d_buffer_cpy = nullptr;

    /* CUDA
        Apply erosion and dilation to current frame
            - Erosion: get minimun pixel value of neighbors in order to reduce noise
            - Dilation: get maximum pixel value of neighbors in order to reconstruct object
    */
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

    void update_background_model(uint8_t* current_frame, uint8_t* mask, int width, int height, int stride, int pixel_stride) {
        // Update the background model using a running average where mask is 0
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Update only if the mask is 255
                if (mask[y * stride + x * pixel_stride] == 0) {
                    for (int channel = 0; channel < 3; ++channel) {
                        int index = y * stride + x * pixel_stride + channel;
                        backg_model[index] = (frame_count * backg_model[index] + current_frame[index]) / (frame_count + 1);
                    }
                }
            }
        }

        // Increment the frame count
        frame_count++;
    }


    // Update buffer_cpy with LAB distance parallelized
    void lab_dist_filter(uint8_t *backg_model, uint8_t *buffer_cpy,
                         int width, int height, int stride, int pixel_stride)
    {
        size_t size = width * height * pixel_stride;
        cudaError_t err;

        if (d_bg_model == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void **>(&d_bg_model), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(reinterpret_cast<void **>(&d_buffer_cpy), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
        }

        // Copy to device
        err = cudaMemcpy(d_bg_model, backg_model, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy(d_buffer_cpy, buffer_cpy, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        // Launch kernel
        kernel_dist_lab<<<gridSize, blockSize>>>(d_bg_model, d_buffer_cpy, width, height, stride, pixel_stride);
        cudaDeviceSynchronize();

        // Copy result back to host
        err = cudaMemcpy(buffer_cpy, d_buffer_cpy, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);
    }

    /*
        Host function to apply filter to current frame
            Called for each frame
    */
    void filter_impl(uint8_t *buffer, int width, int height, int stride,
                     int pixel_stride)
    {
        const int n = 10; // Number of frames to keep in history
        cudaError_t err;

        if (backg_model == nullptr)
        {
            size_t size = width * height * pixel_stride;

            backg_model = new uint8_t[size];
            err = cudaMemcpy(backg_model, buffer, size, cudaMemcpyHostToHost);
            CHECK_CUDA_ERROR(err);

            buffer_cpy = new uint8_t[size];

            err = cudaMalloc(&d_input, size);
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(&d_output, size);
            CHECK_CUDA_ERROR(err);
        }
        err = cudaMemcpy(buffer_cpy, buffer, width * height * pixel_stride, cudaMemcpyHostToHost);
        CHECK_CUDA_ERROR(err);


        // Update background model using mean
        update_background_model(buffer, buffer_cpy, width, height, stride, pixel_stride);

        /* CUDA
            Apply LAB distance to current frame
        */

        lab_dist_filter(backg_model, buffer_cpy, width, height, stride, pixel_stride);

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

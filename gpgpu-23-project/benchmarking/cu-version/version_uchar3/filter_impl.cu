#include "filter_impl.h"
#include <chrono> // chrono_listerals
#include <thread> // sleep_for

#include <cstring> // memcpy
#include <cmath>   // pow, sqrt

#include <uchar.h> // uchar3

#include "kernel.cu"
#include "cuda_utils.cu"

extern "C"
{
    /********************/
    /* Global variables */
    /********************/
    uint8_t *backg_model = nullptr;
    uint8_t *buffer_cpy = nullptr;
    int frame_count = 0;

    uchar3 *d_input;

    uchar3 *d_bg_model = nullptr;
    uchar3 *d_buffer_cpy = nullptr;

    /* Check if pixel is a strong edge */
    bool is_strong_edge(int x, int y, bool *strongEdges, int width, int height)
    {
        if (x < 0 || x >= width || y < 0 || y >= height)
        {
            return false;
        }
        return strongEdges[y * width + x];
    }

    /*
        Apply hysteresis to current frame
        - First pass: mark all pixels above upper threshold as strong edges
        - Second pass: mark all pixels above lower threshold as weak edges
            - If weak edge has strong edge neighbor, mark as strong edge
    */
    void hysteresis_thresholding(uint8_t *image, bool *strongEdges, bool *weakEdges,
                                 int low_threshold, int high_threshold, int width, int height, int pixel_stride)
    {

        bool *strongEdgesCpy = new bool[width * height];
        memset(strongEdgesCpy, 0, width * height * sizeof(bool));

        // First pass
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int index = (y * width + x) * pixel_stride;
                if (image[index] > high_threshold)
                {
                    strongEdges[y * width + x] = true;
                    strongEdgesCpy[y * width + x] = true;
                }
                else if (image[index] > low_threshold)
                {
                    weakEdges[y * width + x] = true;
                }
            }
        }

        // Second pass
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                if (weakEdges[y * width + x])
                {
                    // Check 8-connected neighbors
                    for (int dy = -1; dy <= 1; ++dy)
                    {
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            if (dx == 0 && dy == 0)
                                continue;
                            if (is_strong_edge(x + dx, y + dy, strongEdgesCpy, width, height))
                            {
                                strongEdges[y * width + x] = true;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    /*
        Host function to launch kernels
        Called for each frame
    */
    void host_kernels(uint8_t *backg_model, uint8_t *buffer_cpy,
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
            err = cudaMalloc(reinterpret_cast<void **>(&d_input), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
            // err = cudaMalloc(&d_output, size);
            // CHECK_CUDA_ERROR(err);
        }
        /////// Copy to device ///////
        err = cudaMemcpy(d_bg_model, backg_model, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy(d_buffer_cpy, buffer_cpy, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        /////// Define block and grid sizes //////
        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        ///////////////////////////////////////////////
        //////// Update of background model ///////////
        updateBackgroundModelKernel<<<gridSize, blockSize>>>(d_bg_model, d_buffer_cpy, d_buffer_cpy,
                                                             width, height, stride, pixel_stride, frame_count);
        cudaDeviceSynchronize();
        frame_count++;

        ////////////////////////////////////
        ///////// LAB dist kernel //////////
        kernel_dist_lab<<<gridSize, blockSize>>>(d_bg_model, d_buffer_cpy, width, height, stride, pixel_stride);
        cudaDeviceSynchronize();

        /////////////////////////////////////////////
        ///////// Apply erosion and dilation ////////
        err = cudaMemcpy(d_input, d_buffer_cpy, size, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR(err);

        erosionKernel<<<gridSize, blockSize>>>(d_buffer_cpy, d_input, width, height);
        cudaDeviceSynchronize();
        dilationKernel<<<gridSize, blockSize>>>(d_input, d_buffer_cpy, width, height);

        ////////////////////////////////////////
        /////// Copy result back to host ///////
        err = cudaMemcpy(buffer_cpy, d_buffer_cpy, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy(backg_model, d_bg_model, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);
    }

    /*
        Host function to apply filter to current frame
            Called for each frame
    */
    void filter_impl(uint8_t *buffer, int width, int height, int stride,
                     int pixel_stride)
    {
        cudaError_t err;

        if (backg_model == nullptr)
        {
            size_t size = width * height * pixel_stride;

            buffer_cpy = new uint8_t[size];
            backg_model = new uint8_t[size];
            err = cudaMemcpy(backg_model, buffer, size, cudaMemcpyHostToHost);
            CHECK_CUDA_ERROR(err);
        }
        err = cudaMemcpy(buffer_cpy, buffer, width * height * pixel_stride, cudaMemcpyHostToHost);
        CHECK_CUDA_ERROR(err);

        /* CUDA
            Host function to launch kernels
        */

        host_kernels(backg_model, buffer_cpy, width, height, stride, pixel_stride);

        /****** Apply hysteresis thresholding ******/

        // Define strong_edges and weak_edges as bool arrays
        bool *strong_edges = new bool[width * height];
        bool *weak_edges = new bool[width * height];

        int low_threshold = 4;   // Define your low threshold
        int high_threshold = 30; // Define your high threshold

        hysteresis_thresholding(buffer_cpy, strong_edges, weak_edges,
                                low_threshold, high_threshold, width, height, pixel_stride);

        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                int index = i * stride + j * pixel_stride;

                // Check if it's a strong edge
                if (strong_edges[i * width + j])
                {
                    // If strong edge, set the pixel to white
                    buffer_cpy[index] = 255;     // Red
                    buffer_cpy[index + 1] = 255; // Green
                    buffer_cpy[index + 2] = 255; // Blue
                }
                else
                {
                    // If not a strong edge, set the pixel to black
                    buffer_cpy[index] = 0;     // Red
                    buffer_cpy[index + 1] = 0; // Green
                    buffer_cpy[index + 2] = 0; // Blue
                }

                // Update buffer with result
                buffer[index] = std::min(255, static_cast<int>(buffer[index] + 0.5 * buffer_cpy[index]));
            }
        }

        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}

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
    int frame_count = 0;

    uchar3 *d_input;
    uchar3 *buffer_cpy = nullptr;
    uchar3 *d_bg_model = nullptr;
    uchar3 *d_buffer_cpy = nullptr;

    /*
        Host function to launch kernels
        Called for each frame
    */
    void host_kernels(int width, int height, int stride, int pixel_stride)
    {
        size_t size = width * height * pixel_stride;
        cudaError_t err;

        if (d_buffer_cpy == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void **>(&d_buffer_cpy), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(reinterpret_cast<void **>(&d_input), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
        }
        /////// Copy to device ///////
        err = cudaMemcpy(d_buffer_cpy, buffer_cpy, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        /////// Define block and grid sizes //////
        dim3 blockSize(32, 32);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        ///////////////////////////////////////////////
        //////// Update of background model ///////////
        updateBackgroundModelKernel<<<gridSize, blockSize>>>(d_bg_model, d_buffer_cpy, d_buffer_cpy,
                                                             width, height, stride, pixel_stride, frame_count);
        frame_count++;

        ////////////////////////////////////
        ///////// LAB dist kernel //////////
        kernel_dist_lab<<<gridSize, blockSize>>>(d_bg_model, d_buffer_cpy, width, height, stride, pixel_stride);

        /////////////////////////////////////////////
        ///////// Apply erosion and dilation ////////
        err = cudaMemcpy(d_input, d_buffer_cpy, size, cudaMemcpyDeviceToDevice);
        CHECK_CUDA_ERROR(err);

        erosionKernel<<<gridSize, blockSize>>>(d_buffer_cpy, d_input, width, height);
        dilationKernel<<<gridSize, blockSize>>>(d_input, d_buffer_cpy, width, height);

        //////////////////////////////////////////////////////
        ///////// Apply hysterisis thresholding kernel ///////
        bool *d_strongEdges, *d_weakEdges;
        err = cudaMalloc((void **)&d_strongEdges, width * height * sizeof(bool));
        CHECK_CUDA_ERROR(err);
        err = cudaMalloc((void **)&d_weakEdges, width * height * sizeof(bool));
        CHECK_CUDA_ERROR(err);

        hysteresis_thresholding_kernel<<<gridSize, blockSize>>>(d_buffer_cpy, d_strongEdges, d_weakEdges,
                                                                width, height);

        apply_color_to_buffer_kernel<<<gridSize, blockSize>>>(d_buffer_cpy, d_strongEdges, width, height);

        ////////////////////////////////////////
        ///////// Apply min values kernel ///////
        compute_min_values_kernel<<<gridSize, blockSize>>>(buffer_cpy, d_buffer_cpy,
                                                           width, height);
    }

    /*
        Host function to apply filter to current frame
            Called for each frame
    */
    void filter_impl(uint8_t *buffer, int width, int height, int stride,
                     int pixel_stride)
    {
        cudaError_t err;
        size_t size = width * height * pixel_stride;

        if (d_bg_model == nullptr)
        {
            err = cudaMalloc(reinterpret_cast<void **>(&buffer_cpy), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
            err = cudaMalloc(reinterpret_cast<void **>(&d_bg_model), size * sizeof(uchar3));
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpy(d_bg_model, buffer, size, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);
        }
        // Copy to device
        err = cudaMemcpy(buffer_cpy, buffer, size, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        /* CUDA
            Host function to launch kernels
        */

        host_kernels(width, height, stride, pixel_stride);

        // Copy back to host
        err = cudaMemcpy(buffer, buffer_cpy, size, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);

        {
            using namespace std::chrono_literals;
            // std::this_thread::sleep_for(100ms);
        }
    }
}

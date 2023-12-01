#include "filter_impl.h"
#include <chrono> // chrono_listerals
#include <thread> // sleep_for

#include <cstring> // memcpy
#include <cmath> // pow, sqrt

#include <vector>
#include <algorithm> // rotate, nth_element

// RGB structure
struct rgb {
    uint8_t r, g, b;
};

extern "C" {

    // Back ground model (frame)
    uint8_t* back_ground_model = nullptr;

    // Frame count to know when to uptade
    int frame_count = 0;

    uint8_t* d_input;
    uint8_t* d_output;

    void initialize_cuda_memory(int size)
    {
        cudaMalloc(&d_input, size);
        cudaMalloc(&d_output, size);
    }

    void cleanup_cuda_memory()
    {
        cudaFree(d_input);
        cudaFree(d_output);
    }

    // Global history buffer
    std::vector<std::vector<std::vector<uint8_t>>> pixel_history;

    // Define the transformation matrix from RGB to XYZ
    std::vector<std::vector<double>> rgb_to_xyz = {
        {0.412453, 0.357580, 0.180423},
        {0.212671, 0.715160, 0.072169},
        {0.019334, 0.119193, 0.950227}};

    // Define the XYZ tristimulus values for illuminant "D65"
    std::vector<double> xyz_ref_white = {0.95047, 1.0, 1.08883};

    // Function to convert RGB to LAB
    std::vector<double> rgb_to_lab(const rgb &color)
    {
        // Convert RGB to XYZ
        std::vector<double> xyz(3, 0.0);

        for (size_t i = 0; i < 3; ++i)
        {
            xyz[i] = 0.0;
            xyz[i] += color.r / 255.0 * rgb_to_xyz[i][0];
            xyz[i] += color.g / 255.0 * rgb_to_xyz[i][1];
            xyz[i] += color.b / 255.0 * rgb_to_xyz[i][2];
        }

        // Normalize by the XYZ tristimulus values of the reference white point
        for (size_t i = 0; i < 3; ++i)
        {
            xyz[i] /= xyz_ref_white[i];
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

        // Calculate LAB components
        std::vector<double> lab(3, 0.0);

        lab[0] = 116.0 * xyz[1] - 16.0;     // L
        lab[1] = 500.0 * (xyz[0] - xyz[1]); // a
        lab[2] = 200.0 * (xyz[1] - xyz[2]); // b

        return lab;
    }

    /*
     * Apply erosion to current frame
     *
     * Get minimun pixel value of neighbors in order to reduce noise
     * Copy buffer to temporary first to avoid propagation
    */
    __global__ void erosionKernel(uint8_t* input, uint8_t* output, int width, int height, int stride, int pixel_stride)
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

    void apply_erosion(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
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

    /* void apply_erosion(uint8_t* buffer, int width, int height, int stride,
            int pixel_stride) {
        // Temporary buffer to store the results
        uint8_t* temp = new uint8_t[width * height * pixel_stride];
        std::memcpy(temp, buffer, width * height * pixel_stride);

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                uint8_t min_val = 255;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int i = (y + ky) * stride + (x + kx) * pixel_stride;
                        min_val = std::min(min_val, buffer[i]);
                    }
                }
                temp[y * stride + x * pixel_stride] = min_val;
            }
        }

        std::memcpy(buffer, temp, width * height * pixel_stride);
        delete[] temp;
    }*/

    /*
     * Apply dilatation to current frame
     *
     * Get maximum pixel value of neighbors in order to reconstruct object
     * Copy buffer to temporary first to avoid propagation
    */
    __global__ void dilationKernel(uint8_t* input, uint8_t* output, int width, int height, int stride, int pixel_stride)
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

    void apply_dilation(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
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
    bool is_strong_edge(int x, int y, std::vector<std::vector<bool>>& strongEdges,
            int width, int height) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return false;
        }
        return strongEdges[y][x];
    }

    // Apply hysteresis to current frame
    void hysteresis_thresholding(std::vector<std::vector<int>>& image,
                                std::vector<std::vector<bool>>& strong_edges,
                                std::vector<std::vector<bool>>& weak_edges,
                                int low_threshold, int high_threshold,
                                int width, int height) {

        std::vector<std::vector<bool>> strong_edges_cpy =
            std::vector<std::vector<bool>>(height, std::vector<bool>(width, false));
        // First pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (image[y][x] > high_threshold) {
                    strong_edges[y][x] = true;
                    strong_edges_cpy[y][x] = true;
                } else if (image[y][x] > low_threshold) {
                    weak_edges[y][x] = true;
                }
            }
        }

        // Second pass
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (weak_edges[y][x]) {
                    // Check 8-connected neighbors
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            if (is_strong_edge(x + dx, y + dy,
                                        strong_edges_cpy, width, height)) {
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
    void initialize_pixel_history(int width, int height, int n) {
        pixel_history.resize(height,
                std::vector<std::vector<uint8_t>>(width, std::vector<uint8_t>(n, 0)));
    }

    // Update history, remove oldest and then add current
    void update_pixel_history(uint8_t* buffer, int width, int height,
            int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Assume grayscale for simplicity
                uint8_t new_value = buffer[y * stride + x * pixel_stride];

                // Update history, removing oldest, adding newest
                auto& history = pixel_history[y][x];
                std::rotate(history.begin(), history.begin() + 1,
                        history.end());
                history.back() = new_value;
            }
        }
    }

    // Apply median on specific vector
    uint8_t median_of_vector(const std::vector<uint8_t>& v) {
        std::vector<uint8_t> temp = v;
        size_t n = temp.size() / 2;
        std::nth_element(temp.begin(), temp.begin() + n, temp.end());
        return temp[n];
    }

    // Update background model using median
    void update_background_model_median(uint8_t* back_ground_model, int width,
            int height, int stride, int pixel_stride) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                back_ground_model[y * stride + x * pixel_stride] =
                    median_of_vector(pixel_history[y][x]);
            }
        }
    }

    void filter_impl(uint8_t* buffer, int width, int height, int stride,
            int pixel_stride) {
        const int n = 10; // Number of frames to keep in history

        if (back_ground_model == nullptr) {
            back_ground_model = new uint8_t[width * height * pixel_stride];
            std::memcpy(back_ground_model, buffer, width * height * pixel_stride);
            initialize_pixel_history(width, height, n);
            initialize_cuda_memory(width * height * pixel_stride);
        }

        // Update pixel history with new frame
        update_pixel_history(buffer, width, height, stride, pixel_stride);

        // Update background model using median
        if (++frame_count >= n) {
            update_background_model_median(back_ground_model, width, height,
                    stride, pixel_stride);
        }

        uint8_t *buffer_cpy = new uint8_t[width * height * pixel_stride];
        std::memcpy(buffer_cpy, buffer, width * height * pixel_stride);

        // Here we'll transfer all channels from each pixel to the LAB values
        // so that the buffer image is in grayscale from lab values
        for (int i = 0; i < height; ++i)
        {
            uint8_t *row_bg = back_ground_model + i * stride;
            uint8_t *row_frame = buffer_cpy + i * stride;
            for (int j = 0; j < width; ++j)
            {
                rgb color_bg;
                color_bg.r = row_bg[j * pixel_stride];
                color_bg.g = row_bg[j * pixel_stride + 1];
                color_bg.b = row_bg[j * pixel_stride + 2];

                rgb color_frame;
                color_frame.r = row_frame[j * pixel_stride];
                color_frame.g = row_frame[j * pixel_stride + 1];
                color_frame.b = row_frame[j * pixel_stride + 2];

                std::vector<double> lab_bg = rgb_to_lab(color_bg);
                std::vector<double> lab_frame = rgb_to_lab(color_frame);

                // Transfer the euclidean distance of LAB values between
                // the current frame and the first frame to the grayscale value
                // of the pixel inside the buffer_cpy
                double distance = std::sqrt(std::pow(lab_bg[0] - lab_frame[0], 2) +
                        std::pow(lab_bg[1] - lab_frame[1], 2) +
                        std::pow(lab_bg[2] - lab_frame[2], 2));


                double normalizedValue =
                    static_cast<double>(distance) * 255.0 / 100.0;

                // Ensure the normalized value is within the range [0, 255]
                uint8_t normalized =
                    static_cast<uint8_t>(std::min(255.0, std::max(0.0, normalizedValue)));

                row_frame[j * pixel_stride] = normalized;
                row_frame[j * pixel_stride + 1] = normalized;
                row_frame[j * pixel_stride + 2] = normalized;
            }
        }

        // Apply erosion
        apply_erosion(buffer_cpy, width, height, stride, pixel_stride);

        // Apply dilation
        apply_dilation(buffer_cpy, width, height, stride, pixel_stride);

        // apply hysteresis thresholding
        std::vector<std::vector<int>> grayscale_image(height, std::vector<int>(width));

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                grayscale_image[i][j] = buffer_cpy[i * stride + j * pixel_stride];
            }
        }

        // Prepare matrices for strong and weak edges
        std::vector<std::vector<bool>> strong_edges(height, std::vector<bool>(width, false));
        std::vector<std::vector<bool>> weak_edges(height, std::vector<bool>(width, false));

        // Apply hysteresis thresholding
        int low_threshold = 4;  // Define your low threshold
        int high_threshold = 30; // Define your high threshold
        hysteresis_thresholding(grayscale_image, strong_edges, weak_edges,
                low_threshold, high_threshold, width, height);

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int index = i * stride + j * pixel_stride;
                if (strong_edges[i][j]) {
                    // If strong edge, set the pixel to white
                    // (or another color)
                    buffer_cpy[index] = 255; // Red
                    buffer_cpy[index + 1] = 255; // Green
                    buffer_cpy[index + 2] = 255; // Blue
                } else {
                    // If not a strong edge, set the pixel to black
                    // (or another color)
                    buffer_cpy[index] = 0; // Red
                    buffer_cpy[index + 1] = 0; // Green
                    buffer_cpy[index + 2] = 0; // Blue
                }
            }
        }


        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                int index = i * stride + j * pixel_stride;
                buffer[index] = std::min(255, static_cast<int>(buffer[index] +
                            0.5 * buffer_cpy[index]));
            }
        }

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }
}

// kernel.cu

#include <cstdint>
#include <cmath>
#include <uchar.h>

__constant__ double rgb_to_xyz_device[3][3] = {
    {0.412453, 0.357580, 0.180423},
    {0.212671, 0.715160, 0.072169},
    {0.019334, 0.119193, 0.950227}};

__constant__ double inv255 = 1.0 / 255.0;

__constant__ double invxyz_ref_white[3] = {1.0 / 0.95047, 1.0 / 1.0, 1.0 / 1.08883};

// Function to convert RGB to LAB
__device__ void rgb_to_lab(const uchar3 &color, double *lab)
{
    // Convert RGB to XYZ with unrolling
    double xyz[3] = {
        color.x * inv255 * rgb_to_xyz_device[0][0] + color.y * inv255 * rgb_to_xyz_device[0][1] + color.z * inv255 * rgb_to_xyz_device[0][2],
        color.x * inv255 * rgb_to_xyz_device[1][0] + color.y * inv255 * rgb_to_xyz_device[1][1] + color.z * inv255 * rgb_to_xyz_device[1][2],
        color.x * inv255 * rgb_to_xyz_device[2][0] + color.y * inv255 * rgb_to_xyz_device[2][1] + color.z * inv255 * rgb_to_xyz_device[2][2]};

    // Normalize by the XYZ tristimulus values of the reference white point
    xyz[0] *= invxyz_ref_white[0];
    xyz[1] *= invxyz_ref_white[1];
    xyz[2] *= invxyz_ref_white[2];

    // Nonlinear distortion and linear transformation
    if (xyz[0] > 0.008856)
        xyz[0] = pow(xyz[0], 1.0 / 3.0);
    else
        xyz[0] = 7.787 * xyz[0] + 16.0 / 116.0;

    if (xyz[1] > 0.008856)
        xyz[1] = pow(xyz[1], 1.0 / 3.0);
    else
        xyz[1] = 7.787 * xyz[1] + 16.0 / 116.0;

    if (xyz[2] > 0.008856)
        xyz[2] = pow(xyz[2], 1.0 / 3.0);
    else
        xyz[2] = 7.787 * xyz[2] + 16.0 / 116.0;

    // Calculate LAB components
    lab[0] = 116.0 * xyz[1] - 16.0;     // L
    lab[1] = 500.0 * (xyz[0] - xyz[1]); // a
    lab[2] = 200.0 * (xyz[1] - xyz[2]); // b
}

extern "C"
{
    __global__ void erosionKernel(uchar3 *input, uchar3 *output, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Handle border cases
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            uchar3 min_val = make_uchar3(255, 255, 255);

#pragma unroll
            for (int ky = -1; ky <= 1; ky++)
            {
#pragma unroll
                for (int kx = -1; kx <= 1; kx++)
                {
                    int i = (y + ky) * width + (x + kx);
                    min_val.x = min(min_val.x, input[i].x);
                    min_val.y = min(min_val.y, input[i].y);
                    min_val.z = min(min_val.z, input[i].z);
                }
            }

            output[y * width + x] = min_val;
        }
    }

    __global__ void dilationKernel(uchar3 *input, uchar3 *output, int width, int height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Handle border cases
        if (x > 0 && x < width - 1 && y > 0 && y < height - 1)
        {
            uchar3 max_val = make_uchar3(0, 0, 0);

#pragma unroll
            for (int ky = -1; ky <= 1; ky++)
            {
#pragma unroll
                for (int kx = -1; kx <= 1; kx++)
                {
                    int i = (y + ky) * width + (x + kx);
                    max_val.x = max(max_val.x, input[i].x);
                    max_val.y = max(max_val.y, input[i].y);
                    max_val.z = max(max_val.z, input[i].z);
                }
            }

            output[y * width + x] = max_val;
        }
    }

    __global__ void kernel_dist_lab(uchar3 *bg_model, uchar3 *frame, int width, int height, int stride, int pixel_stride)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        uchar3 *lineptr = (uchar3 *)(frame + y * stride);
        uchar3 *bg_lineptr = (uchar3 *)(bg_model + y * stride);

        uchar3 color_bg, color_frame;
        color_bg = bg_lineptr[x];
        color_frame = lineptr[x];

        // Convert RGB to LAB
        double lab_bg[3], lab_frame[3];
        rgb_to_lab(color_bg, lab_bg);
        rgb_to_lab(color_frame, lab_frame);

        // Calculate LAB distance
        double distance = sqrtf(powf(lab_bg[0] - lab_frame[0], 2) +
                                powf(lab_bg[1] - lab_frame[1], 2) +
                                powf(lab_bg[2] - lab_frame[2], 2));

        uint8_t normalized = (distance * 255.0 / 100.0 > 255.0) ? 255 : ((distance * 255.0 / 100.0 < 0.0) ? 0 : static_cast<uint8_t>(distance * 255.0 / 100.0));

        // Update frame with normalized distance
        lineptr[x].x = normalized;
        lineptr[x].y = normalized;
        lineptr[x].z = normalized;
    }

    __global__ void updateBackgroundModelKernel(uchar3 *bckg_model, const uchar3 *frame, const uchar3 *mask,
                                                int width, int height, int stride, int pixelStride, int frameCount)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        int index = y * stride + x * pixelStride;

        // Update only if the mask is 255
        if (mask[index].x == 0)
        {
            bckg_model[index].x = (frameCount * bckg_model[index].x + frame[index].x) / (frameCount + 1);
            bckg_model[index].y = (frameCount * bckg_model[index].y + frame[index].y) / (frameCount + 1);
            bckg_model[index].z = (frameCount * bckg_model[index].z + frame[index].z) / (frameCount + 1);
        }
    }
}

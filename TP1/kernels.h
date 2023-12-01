#ifndef __KERNEL_H__
#define __KERNEL_H__

// Fills a float 1D buffer with length capacity with a given value.
__global__ void fill1D(float *buffer, float value, size_t length);

// Adds a value to all values of a 1D buffer with some length.
__global__ void inc1D(float *buffer, float value, size_t length);

// Checks whether all values of a 1D buffer with some length are all 
// close to some expected value.
__global__ void check1D(float *buffer, float expectedValue, size_t length);

// Fills a float 2D buffer with (rows*cols) elements and `pitch` total byte size
// for rows (with padding) with a given value.
__global__ void fill2D(float *buffer, float value, int cols, int rows, size_t pitch);

// Adds a value to all values of a 2D buffer with (rows*cols) elements and `pitch` 
// total byte size (with padding) for rows.
__global__ void inc2D(float *buffer, float value, int cols, int rows, size_t pitch);

// Checks whether all values of a 2D buffer (rows*cols) elements and `pitch` total 
// byte size for rows (with padding) are all close to some expected value.
__global__ void check2D(float *buffer, float expectedValue, int cols, int rows, size_t pitch);


#endif // __KERNEL_H__
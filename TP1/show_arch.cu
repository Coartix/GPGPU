
__global__ void printAtRuntime()
{
// #warning "Device code is being compiled."
#ifdef __CUDA_ARCH__
    printf("CUDA architecture of the current running device code: %d.\n", __CUDA_ARCH__);
#endif
}

int main(void)
{
// #warning "Host code is being compiled."
#ifdef __CUDA_ARCH__
    printf("CUDA architecture of the current running host code: %d.\n", __CUDA_ARCH__);
#endif
    printAtRuntime<<<1, 1>>>();
    cudaDeviceSynchronize();
}
#include <cstdio>
#include <cuda_runtime.h>

__global__ void mul_kernel(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] * b[idx];
}

int main()
{
    const int N = 65536;
    const size_t bytes = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    mul_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("c[0] = %f\n", h_c[0]); // expected 2.0

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}


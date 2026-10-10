#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include <random>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

static int div_up(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void dp4a_matmul_kernel(const int8_t* __restrict__ a,
                                   const int8_t* __restrict__ b,
                                   int32_t* __restrict__ c,
                                   int batch,
                                   int out_features,
                                   int in_features) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || col >= out_features) return;

    int acc = 0;
    const int packed = in_features / 4;
    const int32_t* a4 = reinterpret_cast<const int32_t*>(a + row * in_features);
    const int32_t* b4 = reinterpret_cast<const int32_t*>(b + col * in_features);

    for (int k = 0; k < packed; ++k) {
        acc = __dp4a(a4[k], b4[k], acc);
    }
    c[row * out_features + col] = acc;
}

static void cpu_ref(const std::vector<int8_t>& a,
                    const std::vector<int8_t>& b,
                    std::vector<int32_t>& c,
                    int batch,
                    int out_features,
                    int in_features) {
    for (int row = 0; row < batch; ++row) {
        for (int col = 0; col < out_features; ++col) {
            int32_t acc = 0;
            for (int k = 0; k < in_features; ++k) {
                acc += static_cast<int32_t>(a[row * in_features + k]) *
                       static_cast<int32_t>(b[col * in_features + k]);
            }
            c[row * out_features + col] = acc;
        }
    }
}

int main() {
    const int batch = 128;
    const int out_features = 512;
    const int in_features = 4096; // Must be divisible by 4 for DP4A packing.
    const int repeats = 200;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(-8, 7);

    std::vector<int8_t> h_a(batch * in_features);
    std::vector<int8_t> h_b(out_features * in_features);
    std::vector<int32_t> h_c(batch * out_features, 0);
    std::vector<int32_t> h_ref(batch * out_features, 0);

    for (auto& v : h_a) v = static_cast<int8_t>(dist(rng));
    for (auto& v : h_b) v = static_cast<int8_t>(dist(rng));

    int8_t* d_a = nullptr;
    int8_t* d_b = nullptr;
    int32_t* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_b, h_b.size() * sizeof(int8_t)));
    CHECK_CUDA(cudaMalloc(&d_c, h_c.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(int8_t), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid(div_up(out_features, block.x), div_up(batch, block.y));

    for (int i = 0; i < 10; ++i) {
        dp4a_matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, batch, out_features, in_features);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < repeats; ++i) {
        dp4a_matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, batch, out_features, in_features);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));

    cpu_ref(h_a, h_b, h_ref, batch, out_features, in_features);
    int errors = 0;
    int32_t max_abs_error = 0;
    for (size_t i = 0; i < h_c.size(); ++i) {
        int32_t err = abs(h_c[i] - h_ref[i]);
        if (err != 0) ++errors;
        if (err > max_abs_error) max_abs_error = err;
    }

    const double avg_ms = static_cast<double>(ms) / repeats;
    const double ops = static_cast<double>(batch) * out_features * in_features * 2.0;
    const double tops = (ops / (avg_ms / 1000.0)) / 1.0e12;

    printf("DP4A int8 matmul benchmark\n");
    printf("shape: batch=%d out=%d in=%d\n", batch, out_features, in_features);
    printf("average kernel latency: %.6f ms\n", avg_ms);
    printf("effective throughput: %.6f TOPS\n", tops);
    printf("correctness: errors=%d max_abs_error=%d\n", errors, max_abs_error);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return errors == 0 ? 0 : 2;
}

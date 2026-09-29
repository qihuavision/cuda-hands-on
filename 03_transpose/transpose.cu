// transpose.cu —— 手撕第3题：naive vs SMEM+padding
// 编译: nvcc -O3 -arch=sm_86 -o transpose transpose.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s at %s:%d : %s\n",              \
                    cudaGetErrorName(err_), __FILE__, __LINE__,           \
                    cudaGetErrorString(err_));                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define TILE 32

// ---- 1. naive：读合并 / 写跨步 ----
__global__ void transpose_naive(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < n && y < n)
        out[x * n + y] = in[y * n + x];
}

// ---- 2. SMEM 中转 + TILE+1 padding ----
__global__ void transpose_smem(const float* __restrict__ in,
                               float* __restrict__ out, int n) {
    __shared__ float tile[TILE][TILE + 1];          // ← 关键 +1
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    int xg = blockIdx.y * TILE + threadIdx.x;       // 转置后的全局坐标
    int yg = blockIdx.x * TILE + threadIdx.y;
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = in[y * n + x];   // ① 行读：合并
        __syncthreads();                                   // ② 块内对齐
        if (xg < n && yg < n)
            out[yg * n + xg] = tile[threadIdx.x][threadIdx.y]; // ③ 行写：合并
    }
}

void transpose_cpu(const float* a, float* b, int n) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            b[j * n + i] = a[i * n + j];
}

template <typename F>
static double bench(F&& launch, int repeat = 20) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    float best = 1e30f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(beg));
        launch();
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return best;
}

int main() {
    const int n = 8192;                       // 8192² × 4B ×2 ≈ 512MB 显存
    std::vector<float> h(n * n), h_ref(n * n), h_o(n * n);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (float)(rand() % 1000) / 1000.f;
    transpose_cpu(h.data(), h_ref.data(), n);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  h.size() * 4));
    CUDA_CHECK(cudaMalloc(&d_out, h.size() * 4));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), h.size() * 4, cudaMemcpyHostToDevice));

    dim3 grid(n / TILE, n / TILE), block(TILE, TILE);
    // 对拍（SMEM 版）
    transpose_smem<<<grid, block>>>(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_o.data(), d_out, h.size() * 4, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < h.size(); i += 997)
        if (fabsf(h_o[i] - h_ref[i]) > 1e-6f) { puts("MISMATCH"); return 1; }
    puts("对拍通过 ✓");

    double bytes = 2.0 * (double)n * n * 4;    // 读一遍 + 写一遍
    double ms1 = bench([&]{ transpose_naive<<<grid, block>>>(d_in, d_out, n); });
    double ms2 = bench([&]{ transpose_smem  <<<grid, block>>>(d_in, d_out, n); });
    printf("naive : %7.2f ms  → %7.1f GB/s\n", ms1, bytes / ms1 / 1e6);
    printf("smem+1: %7.2f ms  → %7.1f GB/s\n", ms2, bytes / ms2 / 1e6);
    printf("加速比: %.2fx | 3060 峰值 336 GB/s 的 %.0f%%\n",
           ms1 / ms2, 100.0 * (bytes / ms2 / 1e6) / 336.0);
    return 0;
}

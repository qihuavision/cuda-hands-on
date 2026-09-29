// va.cu —— 手撕第1题：vector add（naive vs grid-stride）+ 有效带宽基准
// 教材：讲解/Day053_手撕第1题_VectorAdd与GridStride.md
// 编译(WSL):  nvcc -O3 -arch=sm_86 -o va va.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---- 0. 错误检查宏：所有 CUDA API 与 kernel 之后都用 ----
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

// ---- 1. naive 版：一个线程管一个元素 ----
__global__ void vec_add_naive(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ c, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];          // 生死线：越界保护
}

// ---- 2. grid-stride 版：固定线程总量，循环扫全量 ----
__global__ void vec_add_gs(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, long n) {
    long stride = (long)gridDim.x * blockDim.x;      // 全网格一次跨多远
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// ---- 3. CPU 参考实现（对拍基准，Level 0 思想）----
void vec_add_cpu(const float* a, const float* b, float* c, long n) {
    for (long i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// ---- 4. 两版 launch 的小包装（统一计时接口）----
static void launch_naive(const float* a, const float* b, float* c,
                         long n, int blocks, int threads) {
    vec_add_naive<<<blocks, threads>>>(a, b, c, n);
}
static void launch_gs(const float* a, const float* b, float* c,
                      long n, int blocks, int threads) {
    vec_add_gs<<<blocks, threads>>>(a, b, c, n);
}

// ---- 5. 事件计时 + 有效带宽：热身后取 R 次最优 ----
static double bench(void (*launch)(const float*, const float*, float*,
                                   long, int, int),
                    const float* d_a, const float* d_b, float* d_c,
                    long n, int blocks, int threads, int repeat) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    float best = 1e30f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(beg));
        launch(d_a, d_b, d_c, n, blocks, threads);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    double bytes = 3.0 * (double)n * sizeof(float);   // 读a + 读b + 写c
    return bytes / (best * 1e-3) / 1e9;               // GB/s
}

int main() {
    const long sizes[] = {1L<<20, 1L<<22, 1L<<24, 1L<<26, 1L<<27};
    const int  nsz = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const double PEAK = 336.0;                        // 3060 6G 理论带宽

    printf("%10s | %11s | %11s | %10s | %7s\n",
           "N", "naive GB/s", "gs GB/s", "gs ms", "%peak");
    printf("-----------------------------------------------------------\n");

    for (int s = 0; s < nsz; ++s) {
        const long n = sizes[s];

        // host 端造数据
        std::vector<float> h_a(n), h_b(n), h_c(n), h_ref(n);
        for (long i = 0; i < n; ++i) {
            h_a[i] = (float)(rand() % 1000) / 1000.f;
            h_b[i] = (float)(rand() % 1000) / 1000.f;
        }
        vec_add_cpu(h_a.data(), h_b.data(), h_ref.data(), n);  // CPU 基准

        // device 显存 + 上传
        float *d_a, *d_b, *d_c;
        CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n*sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(float),
                              cudaMemcpyHostToDevice));

        // launch 配置
        const int threads = 256;
        const int blocks_naive = (int)((n + threads - 1) / threads);
        const int blocks_gs    = 4096;   // 固定 4096 块 = 约 105 万线程

        // 对拍：gs 版跑一次，拷回验证（前 4096 个 + 随机抽 64 个）
        launch_gs(d_a, d_b, d_c, n, blocks_gs, threads);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n*sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (long i = 0; i < 4096 && i < n; ++i)
            if (fabsf(h_c[i] - h_ref[i]) > 1e-5f) {
                fprintf(stderr, "MISMATCH at %ld\n", i); return 1;
            }
        for (int k = 0; k < 64; ++k) {
            long i = (long)(rand() % n);
            if (fabsf(h_c[i] - h_ref[i]) > 1e-5f) {
                fprintf(stderr, "MISMATCH at %ld\n", i); return 1;
            }
        }

        // 基准：naive 与 gs 各测（内部自动 warm-up + 取最优）
        double bw_naive = bench(launch_naive, d_a, d_b, d_c,
                                n, blocks_naive, threads, 20);
        double bw_gs    = bench(launch_gs,    d_a, d_b, d_c,
                                n, blocks_gs,    threads, 20);

        printf("%10ld | %11.1f | %11.1f | %10.4f | %6.1f%%\n",
               n, bw_naive, bw_gs,
               3.0 * n * sizeof(float) / bw_gs / 1e6,   // gs 耗时(ms)
               100.0 * bw_gs / PEAK);

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }
    return 0;
}

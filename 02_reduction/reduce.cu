// reduce.cu —— 手撕第2题：两级归约（方案A 两次launch / 方案B 最后块聚合）
// nvcc -O3 -arch=sm_86 -o reduce reduce.cu
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

#define BLOCK 256
#define NWARP (BLOCK / 32)                 // 8

// ---------- 方案A-1：每块算部分和 ----------
__global__ void reduce_partial(const float* __restrict__ in,
                               float* __restrict__ partials, long n) {
    __shared__ float smem[NWARP];          // 只需"每warp 4B"，共 32B
    const int tid  = threadIdx.x;
    const int lane = tid & 31;             // warp 内编号 0-31
    const int wid  = tid >> 5;             // warp 编号 0-7
    float acc = 0.f;
    // 粗化 + grid-stride（七级的第7级）：warp 内相邻线程地址相邻 → 合并访存
    for (long i = (long)blockIdx.x * BLOCK + tid;
         i < n; i += (long)gridDim.x * BLOCK)
        acc += in[i];
    // warp 内 5 轮 shuffle 归约：无 SMEM、无同步（32线程天然锁步）
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) smem[wid] = acc;        // 每 warp 的 lane0 落 SMEM
    __syncthreads();
    // warp0 前 8 个线程把 8 个部分和收口
    if (wid == 0 && lane < NWARP) {
        acc = smem[lane];
        for (int off = 4; off > 0; off >>= 1)      // 8→4→2→1 三轮
            acc += __shfl_down_sync(0xffu, acc, off);  // mask=低8位=真正执行者
        if (lane == 0) partials[blockIdx.x] = acc;
    }
}

// ---------- 方案A-2：单块终归约 ----------
__global__ void reduce_final(const float* __restrict__ partials,
                             float* __restrict__ out, int nparts) {
    __shared__ float smem[NWARP];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    float acc = 0.f;
    for (int i = tid; i < nparts; i += BLOCK)      // 单块粗化读小数组
        acc += partials[i];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) smem[wid] = acc;
    __syncthreads();
    if (wid == 0 && lane < NWARP) {
        acc = smem[lane];
        for (int off = 4; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffu, acc, off);
        if (lane == 0) *out = acc;
    }
}

// ---------- 方案B：单kernel，最后完成的块负责收口 ----------
__global__ void reduce_lastblock(const float* __restrict__ in,
                                 float* __restrict__ out, long n,
                                 unsigned int* __restrict__ count,
                                 float* __restrict__ scratch) {
    __shared__ float smem[NWARP];
    __shared__ bool  amLast;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int wid  = tid >> 5;
    float acc = 0.f;
    for (long i = (long)blockIdx.x * BLOCK + tid;
         i < n; i += (long)gridDim.x * BLOCK)
        acc += in[i];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) smem[wid] = acc;
    __syncthreads();
    if (wid == 0 && lane < NWARP) {
        acc = smem[lane];
        for (int off = 4; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffu, acc, off);
        if (lane == 0) scratch[blockIdx.x] = acc;  // 数据发布
    }
    __syncthreads();                        // lane0 的写对全块可见（块内屏障）
    if (tid == 0) {
        __threadfence();                    // 铁律：发布数据之后，取票之前
        unsigned int ticket = atomicInc(count, gridDim.x);
        amLast = (ticket == gridDim.x - 1); // 只有 tid0 写 amLast
    }
    __syncthreads();                        // amLast 广播全块；syncthreads 不在发散分支里
    if (amLast) {
        acc = 0.f;                          // 全体线程粗化读 scratch
        for (int i = tid; i < (int)gridDim.x; i += BLOCK)
            acc += scratch[i];
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        if (lane == 0) smem[wid] = acc;
        __syncthreads();
        if (wid == 0 && lane < NWARP) {
            acc = smem[lane];
            for (int off = 4; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xffu, acc, off);
            if (lane == 0) { *out = acc; *count = 0; }  // 显式复位计数器
        }
    }
}

// ---------- CPU 参考（double 累加，误差基准） ----------
static double reduce_cpu(const float* a, long n) {
    double s = 0.0;
    for (long i = 0; i < n; ++i) s += (double)a[i];
    return s;
}

static double bench_event(void (*launch_all)(const float*, float*, long,
                                             float*, unsigned int*, float*),
                          const float* d_in, float* d_out, long n,
                          float* d_partial, unsigned int* d_count,
                          float* d_scratch, int repeat) {
    cudaEvent_t beg, end;
    CUDA_CHECK(cudaEventCreate(&beg));
    CUDA_CHECK(cudaEventCreate(&end));
    float best = 1e30f;
    for (int r = 0; r < repeat; ++r) {
        CUDA_CHECK(cudaEventRecord(beg));
        launch_all(d_in, d_out, n, d_partial, d_count, d_scratch);
        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, beg, end));
        if (ms < best) best = ms;
    }
    CUDA_CHECK(cudaEventDestroy(beg));
    CUDA_CHECK(cudaEventDestroy(end));
    return (double)n * sizeof(float) / (best * 1e-3) / 1e9;  // 读N为主
}

// 两方案的"一次完整调用"包装（方案A 包住两个 kernel 一起计时）
static void call_planA(const float* in, float* out, long n,
                       float* partial, unsigned int*, float*) {
    reduce_partial<<<512, BLOCK>>>(in, partial, n);
    CUDA_CHECK(cudaGetLastError());
    reduce_final<<<1, BLOCK>>>(partial, out, 512);
    CUDA_CHECK(cudaGetLastError());
}
static void call_planB(const float* in, float* out, long n,
                       float*, unsigned int* count, float* scratch) {
    CUDA_CHECK(cudaMemset(count, 0, sizeof(unsigned int)));  // 每次前清零
    reduce_lastblock<<<512, BLOCK>>>(in, out, n, count, scratch);
    CUDA_CHECK(cudaGetLastError());
}

int main() {
    const long sizes[] = {1L<<20, 1L<<24, 1L<<26};   // 1M 16M 64M
    const int  nsz = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const double PEAK = 336.0;

    printf("%10s | %10s | %10s | %8s | %8s | %8s | %10s\n",
           "N", "A GB/s", "B GB/s", "A %peak", "B %peak",
           "B ms", "rel_err");
    printf("------------------------------------------------------------------\n");

    float *d_in, *d_partial, *d_scratch, *d_out;
    unsigned int* d_count;
    CUDA_CHECK(cudaMalloc(&d_in,     (size_t)(1L<<26) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial, 512 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch, 512 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_count,   sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_out,     sizeof(float)));

    for (int s = 0; s < nsz; ++s) {
        const long n = sizes[s];
        std::vector<float> h(n);
        for (long i = 0; i < n; ++i) h[i] = (float)(rand() % 1000) / 1000.f;
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float),
                              cudaMemcpyHostToDevice));

        double cpu_ref = reduce_cpu(h.data(), n);

        // 正确性：两方案各跑一次对拍（浮点非结合 → 用相对误差）
        float got;
        call_planA(d_in, d_out, n, d_partial, d_count, d_scratch);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double errA = fabs(got - cpu_ref) / cpu_ref;
        call_planB(d_in, d_out, n, d_partial, d_count, d_scratch);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&got, d_out, sizeof(float), cudaMemcpyDeviceToHost));
        double errB = fabs(got - cpu_ref) / cpu_ref;

        double bwA = bench_event(call_planA, d_in, d_out, n,
                                 d_partial, d_count, d_scratch, 20);
        double bwB = bench_event(call_planB, d_in, d_out, n,
                                 d_partial, d_count, d_scratch, 20);

        printf("%10ld | %10.1f | %10.1f | %7.1f%% | %7.1f%% | %8.4f | %9.2e\n",
               n, bwA, bwB, 100*bwA/PEAK, 100*bwB/PEAK,
               (double)n * sizeof(float) / bwB / 1e6, (errA > errB ? errA : errB));
    }
    return 0;
}

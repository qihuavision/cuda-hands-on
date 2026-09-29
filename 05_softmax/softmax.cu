// softmax.cu —— 手撕第5题：3-pass vs online softmax（每 block 一行，行归约四段式）
// 编译: nvcc -O3 -arch=sm_86 -o softmax softmax.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#define CUDA_CHECK(call) do{cudaError_t e_=(call);if(e_!=cudaSuccess){\
  fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);}}while(0)
#define BLOCK 256
#define NWARP (BLOCK/32)

// 归约骨架：粗化 → warp 内 shuffle 5 轮 → SMEM 跨 warp → warp0 收口
// （题 2 的四段式原样复用——只换"合并算子"：max 或 + 或 配对换算）
__device__ float block_max(float val) {
    __shared__ float sm[NWARP];
    for (int off=16; off>0; off>>=1)                    // warp 内：32→1
        val = fmaxf(val, __shfl_down_sync(~0u, val, off));
    if ((threadIdx.x & 31)==0) sm[threadIdx.x>>5] = val; // 各 warp 首线程落 SMEM
    __syncthreads();
    if (threadIdx.x < NWARP) {                           // warp0 收口
        float v = sm[threadIdx.x];
        for (int off=4; off>0; off>>=1)
            v = fmaxf(v, __shfl_down_sync(0xffu, v, off));
        if (threadIdx.x==0) sm[0] = v;
    }
    __syncthreads();
    return sm[0];
}
__device__ float block_sum(float val) {                  // 同骨架，max→+
    __shared__ float sm[NWARP];
    for (int off=16; off>0; off>>=1)
        val += __shfl_down_sync(~0u, val, off);
    if ((threadIdx.x & 31)==0) sm[threadIdx.x>>5] = val;
    __syncthreads();
    if (threadIdx.x < NWARP) {
        float v = sm[threadIdx.x];
        for (int off=4; off>0; off>>=1) v += __shfl_down_sync(0xffu, v, off);
        if (threadIdx.x==0) sm[0] = v;
    }
    __syncthreads();
    return sm[0];
}

// ---------- 版本 1：3-pass（教科书，读 3 遍写 1 遍） ----------
__global__ void softmax_3pass(float* __restrict__ x, int d) {
    int row = blockIdx.x;
    float* r = x + (size_t)row * d;
    // pass1：全行最大（读第1遍）
    float local = -INFINITY;
    for (int i = threadIdx.x; i < d; i += BLOCK) local = fmaxf(local, r[i]);
    const float m = block_max(local);
    // pass2：指数和（读第2遍）
    float lsum = 0.f;
    for (int i = threadIdx.x; i < d; i += BLOCK) lsum += __expf(r[i]-m);
    const float inv_l = 1.0f / block_sum(lsum);
    // pass3：归一化写回（读第3遍+写）
    for (int i = threadIdx.x; i < d; i += BLOCK) r[i] = __expf(r[i]-m) * inv_l;
}

// ---------- 版本 2：online（读 1 遍养 m/ℓ + 读 1 遍写 1 遍归一化） ----------
__global__ void softmax_online(float* __restrict__ x, int d) {
    int row = blockIdx.x;
    float* r = x + (size_t)row * d;
    float m = -INFINITY, l = 0.f;                        // 每线程私有运行态
    for (int i = threadIdx.x; i < d; i += BLOCK) {       // ★唯一一遍：同时养 m 和 ℓ
        float v = r[i];
        float m_new = fmaxf(m, v);
        l = l * __expf(m - m_new) + __expf(v - m_new);   // ★换算律②（线程级 online）
        m = m_new;
    }
    // 块级归约：合并 256 份 (m,l)——不能直接 l+=l'！基准不同要配对换算
    __shared__ float sm_m[NWARP], sm_l[NWARP];
    for (int off=16; off>0; off>>=1) {                   // warp 内配对合并
        float mo = __shfl_down_sync(~0u, m, off), lo = __shfl_down_sync(~0u, l, off);
        float mn = fmaxf(m, mo);
        l = l*__expf(m-mn) + lo*__expf(mo-mn);           // ★两份账折算到共同基准
        m = mn;
    }
    if ((threadIdx.x & 31)==0) { sm_m[threadIdx.x>>5]=m; sm_l[threadIdx.x>>5]=l; }
    __syncthreads();
    if (threadIdx.x < NWARP) {                           // 跨 warp 再配对
        float mm = sm_m[threadIdx.x], ll = sm_l[threadIdx.x];
        for (int off=4; off>0; off>>=1) {
            float mo=__shfl_down_sync(0xffu,mm,off), lo=__shfl_down_sync(0xffu,ll,off);
            float mn=fmaxf(mm,mo);
            ll = ll*__expf(mm-mn) + lo*__expf(mo-mn); mm = mn;
        }
        if (threadIdx.x==0){ sm_m[0]=mm; sm_l[0]=ll; }
    }
    __syncthreads();
    const float inv_l = 1.0f / sm_l[0];
    for (int i = threadIdx.x; i < d; i += BLOCK)         // 第二遍：归一化（要最终 m）
        r[i] = __expf(r[i] - sm_m[0]) * inv_l;
}
// main：对拍（CPU 参考逐行减 max）+ cudaEvent 基准 + GB/s 与 %peak——同题 1/2 模板（略，见仓库）

#include <iostream>
#include <cmath>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "types.h"

namespace ffrnn {
namespace {

#define CHECK_RUNTIME_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__device__ inline bool in_radius(float* x, float* y, float R) {
    float r = 0.0;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        r += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return r <= R * R;
}

__device__ inline bool inside_grid(int i, int j, int k) {
    return i >= 0 && j >= 0 && k >= 0 && i < KERNEL_SIZE && j < KERNEL_SIZE && k < KERNEL_SIZE;
}

__device__ inline float trilinear_w(float d, int b) {
    return b * d + (1 - b) * (1 - d);
}

__device__ inline float window_smooth_weight(float* r) {
    float v = 1.0f - (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    return v * v * v;
}
__device__ inline void ball2cyl(float x, float y, float z, float& rx, float& ry, float& rz) {
    float x2_y2 = x * x + y * y;
    float z2 = z * z;
    float x2_y2_z2 = x2_y2 + z2;
    if (x2_y2_z2 == 0) {
        rx = ry = rz = 0;
        return;
    }
    float rt_x2_y2_z2 = std::sqrt(x2_y2_z2);
    // std::cout << "rt_x2_y2_z2: " << rt_x2_y2_z2 << std::endl;
    if (5.0f / 4.0f * z2 <= x2_y2) {
        // std::cout << "ball2cyl: " << "case 2" << std::endl; 
        float p = rt_x2_y2_z2 / std::sqrt(x2_y2);
        rx = x * p;
        ry = y * p;
        rz = 3.0f / 2.0f * z;
    } else {
        // std::cout << "ball2cyl: " << "case 3" << std::endl;
        float p = std::sqrt((3.0f * rt_x2_y2_z2) / (rt_x2_y2_z2 + std::abs(z)));
        rx = x * p;
        ry = y * p;
        if (z > 0) rz = rt_x2_y2_z2;
        else rz = -rt_x2_y2_z2;
    }
}


__device__ inline void cyl2cube(float x, float y, float z, float& rx, float& ry, float& rz) {
    if (x == 0 && y == 0) {
        rx = ry = 0;
        rz = z;
        return;
    }
    float rt_x2_y2 = std::sqrt(x * x + y * y);
    if (std::abs(y) <= std::abs(x)) {
        // std::cout << "cyl2cube: " << "case 2" << std::endl; 
        if (x > 0) {
            rx = rt_x2_y2;
            ry = rx * std::atan2(y, x) * 4.0f / PI;
        } else {
            rx = -rt_x2_y2;
            ry = rx * std::atan2(-y, -x) * 4.0f / PI;
        }

        rz = z;
        
    } else {
        // std::cout << "cyl2cube: " << "case 3" << std::endl; 
        if (y > 0) {
            ry = rt_x2_y2;
            rx = ry * std::atan2(x, y) * 4.0f / PI;
        } else {
            ry = -rt_x2_y2;
            rx = ry * std::atan2(-x, -y) * 4.0f / PI;
        }
        rz = z;
    }
}

__device__ inline void ball2cube(float* r_ptr, float* h_ptr) {
    float rx, ry, rz;
    // std::cout << "ball: " << r_ptr[0] << " " << r_ptr[1] << " " << r_ptr[2] << std::endl;
    ball2cyl(r_ptr[0], r_ptr[1], r_ptr[2], rx, ry, rz);
    // std::cout << "cyl: " << rx << " " << ry << " " << rz << std::endl;
    cyl2cube(rx, ry, rz, h_ptr[0], h_ptr[1], h_ptr[2]);
    // std::cout << "cube: " << h_ptr[0] << " " << h_ptr[1] << " " << h_ptr[2] << std::endl;
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        h_ptr[i] = (h_ptr[i] + 1.0f) / 2.0f;
    }
}

__device__ inline void weighted_ball2grid(float* r_ptr, float* grid_ptr, float smooth_weight) {
    float h[3];
    ball2cube(r_ptr, h);
    // std::cout << "h: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    h[0] *= (KERNEL_SIZE - 1);
    h[1] *= (KERNEL_SIZE - 1);
    h[2] *= (KERNEL_SIZE - 1);
    
    int idx[3];
    float dx[3]; 
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        idx[i] = static_cast<int>(floor(h[i]));
        if (idx[i] == KERNEL_SIZE - 1) --idx[i];
        dx[i] = h[i] - idx[i];
    }

    int fi[] = {0, 1, 1, 0, 0, 1, 1, 0};
    int fj[] = {0, 0, 1, 1, 0, 0, 1, 1};
    int fk[] = {0, 0, 0, 0, 1, 1, 1, 1};

    // std::cout << "h * size: " << h[0] << " " << h[1] << " " << h[2] << std::endl;
    // std::cout << "idx[]: " << idx[0] << " " << idx[1] << " " << idx[2] << std::endl;

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        int i = idx[0] + fi[t];
        int j = idx[1] + fj[t];
        int k = idx[2] + fk[t];
        // std::cout << i << " " << j << " " << k << std::endl;
        // Safe to remove it.
        // if (!inside_grid(i, j, k)) continue;
        // std::cout << "work" << i << " " << j << " " << k << std::endl;
        grid_ptr[i * KERNEL_SIZE * KERNEL_SIZE + j * KERNEL_SIZE + k] = 
            smooth_weight * trilinear_w(dx[0], fi[t]) *
            trilinear_w(dx[1], fj[t]) * trilinear_w(dx[2], fk[t]);
    }
}

__device__ inline void ball2grid_with_window(float* r_ptr, float* grid_ptr) {
    // std::cout << "r_ptr: " << r_ptr[0] << " " << r_ptr[1] << " " << r_ptr[3] << std::endl;
    // std::cout << window_smooth_weight(r_ptr) << std::endl;
    weighted_ball2grid(r_ptr, grid_ptr, window_smooth_weight(r_ptr));
}
    
}  // namespace

#define DIM_X 32
#define DIM_Y 32

__global__ void debug_kernel() {
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    printf("debug kernel %d %d %d %d\n", bi, bj, ti, tj);
}

__global__ void offset_kernel(
    int N, int M, float R, bool include_diag,
    float* __restrict__ X, float* __restrict__ Y,
    int* __restrict__ offset, int* __restrict__ grad_offset) {
    
    // printf("start offset_kernel %d\n", 0);

    __shared__ float pts_x[DIM_X * 3];
    __shared__ float pts_y[DIM_Y * 3];
    
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    
    int I = bi * DIM_X + ti;
    int J = bj * DIM_Y + tj;
    
    if (I < N && J < M) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            pts_x[ti * 3 + i] = X[I * 3 + i];
        }
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            pts_y[tj * 3 + j] = Y[J * 3 + j];
        }
    }
    __syncthreads();
    // printf("done sync %d %d %d %d %d %d\n", bi, bj, ti, tj, I, J);

    if (I >= N || J >= M) return;
    if (I == J && !include_diag) return;
    // printf("enter offset %d %d\n", I, J);

    if (in_radius(pts_x + ti * 3, pts_y + tj * 3, R)) {
        atomicAdd(offset + I, 1);
        atomicAdd(grad_offset + J, 1);
    }
}

__global__ void neighbor_kernel(int N, int M, float R, bool include_diag,
    float* __restrict__ X, float* __restrict__ Y,
    int* __restrict__ offset, int* __restrict__ grad_offset,
    int* __restrict__ fwd_cnt, int* __restrict__ bkd_cnt,
    int* __restrict__ nn_list, float* __restrict__ nw_list, int* grad_nn_list) {

    __shared__ float pts_x[DIM_X * 3];
    __shared__ float pts_y[DIM_Y * 3];
    float r[3];
    
    int bi = blockIdx.x;
    int bj = blockIdx.y;
    int ti = threadIdx.x;
    int tj = threadIdx.y;
    
    int I = bi * DIM_X + ti;
    int J = bj * DIM_Y + tj;
    if (I < N && J < M) {
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            pts_x[ti * 3 + i] = X[I * 3 + i];
        }
        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            pts_y[tj * 3 + j] = Y[J * 3 + j];
        }
    }
    // printf("start sync %d %d %d %d\n", bi, bj, ti, tj);
    __syncthreads();
    // printf("done sync %d %d %d %d\n", bi, bj, ti, tj);
    if (I >= N || J >= M) return;
    if (I == J && !include_diag) return;

    if (in_radius(pts_x + ti * 3, pts_y + tj * 3, R)) {
        int u_offset = offset[I] + atomicAdd(fwd_cnt + I, 1);
        int v_offset = grad_offset[J] + atomicAdd(bkd_cnt + J, 1);

        nn_list[u_offset] = J;
        grad_nn_list[v_offset * 2] = I;
        grad_nn_list[v_offset * 2 + 1] = u_offset;
        
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            r[i] = (pts_y[tj * 3 + i] - pts_x[ti * 3 + i]) / R;
        }
        ball2grid_with_window(r, nw_list + u_offset * SPATIAL_SIZE);
    }
}

std::vector<at::Tensor> bf_gpu(torch::Tensor sources, torch::Tensor candidates, float R, bool include_diag) {
    CHECK_CUDA(sources);
    CHECK_CUDA(candidates);

    std::cout << "start gpu" << std::endl;
    int N = torch::size(sources, 0);
    int M =  torch::size(candidates, 0);
    float* X = sources.data_ptr<float>();
    float* Y = candidates.data_ptr<float>();

    auto th_option = sources.options();
    std::cout << "Option: " << th_option << std::endl; 
    
    // Allocate offset memory
    torch::Tensor th_nn_offset = torch::zeros({N + 1}, th_option.dtype(torch::kInt32));
    torch::Tensor th_grad_nn_offset = torch::zeros({M + 1}, th_option.dtype(torch::kInt32));
    int* offset = th_nn_offset.data_ptr<int>();
    int* grad_offset = th_grad_nn_offset.data_ptr<int>();
    std::cout << "done allocate offset memory" << std::endl;
     
    // Get nn offset and grad offset
    const dim3 block(DIM_X, DIM_Y);
    const dim3 grid((N + DIM_X - 1) / DIM_X, (M + DIM_Y - 1) / DIM_Y);
    
    // printf("N M %d %d\n", N, M);
    // printf("block size %d %d\n", block.x, block.y);
    // printf("grid size %d %d\n", grid.x, grid.y);

    offset_kernel<<< grid, block >>>(N, M, R, include_diag, X, Y, offset, grad_offset);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());

    // cudaDeviceSynchronize();
    // CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    // std::cout << "done counter, start prefix sum" << std::endl;

    thrust::device_ptr<int> offset_wrap(offset);
    thrust::exclusive_scan(offset_wrap, offset_wrap + N + 1, offset_wrap);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    thrust::device_ptr<int> grad_offset_wrap(grad_offset);
    thrust::exclusive_scan(grad_offset_wrap, grad_offset_wrap + M + 1, grad_offset_wrap);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());

    // cudaDeviceSynchronize();
    // CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    // std::cout << "done offset" << std::endl;
    
    // Allocate neighbor memory
    int total_neighbor;
    cudaMemcpy((void*) (&total_neighbor), (void*) (offset + N), sizeof(int), cudaMemcpyDeviceToHost);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());

    std::cout << "total neighbor: " << total_neighbor << std::endl;

    torch::Tensor th_nn_list = torch::zeros(total_neighbor, th_option.dtype(torch::kInt32));
    torch::Tensor th_nw_list = torch::zeros(total_neighbor * SPATIAL_SIZE, th_option.dtype(torch::kFloat32));
    torch::Tensor th_grad_nn_list = torch::zeros(
        total_neighbor * 2, th_option.dtype(torch::kInt32));

    int* nn_list = th_nn_list.data_ptr<int>();
    float* nw_list = th_nw_list.data_ptr<float>();
    int* grad_nn_list = th_grad_nn_list.data_ptr<int>();
    
    int* fwd_cnt = nullptr;
    int* bkd_cnt = nullptr;
    cudaMalloc((void**) &fwd_cnt, N * sizeof(int));
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    cudaMalloc((void**) &bkd_cnt, M * sizeof(int));
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());

    cudaMemset((void*) fwd_cnt, 0, N * sizeof(int));
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    cudaMemset((void*) bkd_cnt, 0, M * sizeof(int));
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());

    std::cout << "done allocation" << std::endl;

    // Get neighbor and grad neighbor
    neighbor_kernel<<< grid, block >>>(
        N, M, R, include_diag, X, Y, offset, grad_offset,
        fwd_cnt, bkd_cnt, nn_list, nw_list, grad_nn_list);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());

    std::cout << "done get neighbor" << std::endl;

    // Release counter memory
    cudaFree((void*) fwd_cnt);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    cudaFree((void*) bkd_cnt);
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    
    std::cout << "done release memory" << std::endl;
    cudaDeviceSynchronize();
    CHECK_RUNTIME_ERROR(cudaPeekAtLastError());
    return {th_nn_offset, th_nn_list, th_nw_list, th_grad_nn_offset, th_grad_nn_list};
}

}  // namespace ffrnn
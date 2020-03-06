#include "conus_gpu.h"

#include <iostream>
#include <Random123/threefry.h>
#include <Random123/ReinterpretCtr.hpp>

#include "example_seeds.h"
#include "util_cuda.h"


using namespace r123;
using namespace std;

void deleteRandomsGPU(double * arr)
{
    CHECKCALL(cudaFree(arr));
}

// TODO: will need to figure out what to do with the generate1 VF
// TODO: this will only work for buf_ptr < 4
__device__ double ConusUniformGPU::get()
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    double elt = buf_d[tid+4*buf_ptr[tid]];
    buf_ptr[tid] ++;
    return elt;
}

__device__ void ConusUniformGPU::fill_buf_d()
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    buf_ptr[tid] = 0;
    // uniform_ct_gpu<double>(ulseed, buf_d);
    // typedef Threefry4x64 G;
    union {
        G::ctr_type c;
        long4 i;
    }u;
    int n_cycle = (int)(_N/_nthreads+1) ;
    for (int i_cycle=0; i_cycle<n_cycle; ++i_cycle){
        int idx = 4*tid + i_cycle*_nthreads;
        // Don't advance the RNG if not going to use result
        if (idx + 3 < 4*_N) {
            G rng;
            G::key_type k = {{tid, ulseed}};
            G::ctr_type c = {{}};
            // Grab previous chunck's output state
            if (i_cycle > 0 ) c = buf_state[tid - (i_cycle-1)*_nthreads];

            c.incr();
            u.c = rng(c, k);

            buf_d[idx]   = ((double)((uint64_t)u.i.x))/((double)ULONG_MAX);
            buf_d[idx+1] = ((double)((uint64_t)u.i.y))/((double)ULONG_MAX);
            buf_d[idx+2] = ((double)((uint64_t)u.i.z))/((double)ULONG_MAX);
            buf_d[idx+3] = ((double)((uint64_t)u.i.w))/((double)ULONG_MAX);

            buf_state[tid+i_cycle*_nthreads] = c;
        }
    }
}

__host__ void ConusUniformGPU::initialize()
{
    // TODO: this shouldn't go into the constructor, but we should add
    // a call-guard to prevent repreated calls
    size_t rn_size     = 4*_N * sizeof(double);
    size_t state_size  = _N * sizeof(G::ctr_type);
    size_t ptr_size    = _N * sizeof(int);

    CHECKCALL(cudaMalloc(& buf_d, rn_size));
    CHECKCALL(cudaMalloc(& buf_state, state_size ));
    CHECKCALL(cudaMalloc(& buf_ptr, ptr_size));
    buf_h = (double * ) malloc(4*_N*sizeof(double));
};

// TODO: build destructor to safely free arrays

__host__ void ConusUniformGPU::copyToHost()
{
    // TODO: add checks
    cudaMemcpy(buf_h, buf_d, 4*_N*sizeof(double),
               cudaMemcpyDeviceToHost);
  _p = -1;
}

// __host__ double ConusUniformGPU::generate1() {
__host__ double ConusUniformGPU::operator()()
{
    _p++;
    if (_p < 4*_N) return buf_h[_p];
    // TODO: figure out what should happen here?
    return -1.;
}


// Entry point
__global__ void generateOnDevice_kernel(ConusUniformGPU * ud_device)
{
    ud_device->fill_buf_d();
}

void generateOnDevice(ConusUniformGPU * ud_host, ConusUniformGPU * ud_device)
{
    unsigned threads_per_block = THREADS_PER_BLOCK;
    // assert(ud.N()%THREADS_PER_BLOCK == 0);
    unsigned blocks_per_grid   = ud_host->N() / threads_per_block;
    generateOnDevice_kernel<<<blocks_per_grid, threads_per_block>>>(ud_device);
}

ConusUniformGPU * sendToDevice(ConusUniformGPU * ud_host)
{
    ConusUniformGPU * ud_device;
    // TODO: add checks
    cudaMalloc(& ud_device, sizeof(ConusUniformGPU));
    cudaMemcpy(ud_device, ud_host, sizeof(ConusUniformGPU),
               cudaMemcpyHostToDevice);
    return ud_device;
}

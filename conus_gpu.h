#ifndef __CONUS_GPU_H
#define __CONUS_GPU_H


#include <Random123/threefry.h>
#include <Random123/ReinterpretCtr.hpp>

typedef long unsigned uint64_t;
#define THREADS_PER_BLOCK 32



void deleteRandomsGPU(double * arr);

double * generateRandomsGPUd(unsigned long N);

int64_t * generateRandomsGPUi(unsigned long N);


#include "Random.h"

// TODO: figure out problems with virtual pointers later
// class ConusUniformGPU : public galsim::BaseDeviate {
class ConusUniformGPU {

    public:

        // __host__ __device__ ConusUniformGPU(long lseed, int N):
        //    galsim::BaseDeviate(lseed), ulseed(lseed), _N(N), _p(0) {}
        __host__ __device__ ConusUniformGPU(long lseed, int nthreads, int N):
            ulseed(lseed), _nthreads(nthreads), _N(N), _p(0) {}


        // TODO: will need to figure out what to do with the generate1 VF
        __device__ double get();

        __device__ void fill_buf_d();

        __host__ void initialize();

        // TODO: build destructor to safely free arrays
        __host__  ~ConusUniformGPU() {};

        __host__ void copyToHost();

        __host__ __device__ int N() { return _N;}

        // __host__ double generate1();
        __host__ double operator()();

    private:
        int _N;
        int _p;
        int _nthreads;
        int * buf_ptr;
        unsigned long ulseed;

        // NOTE: random numbers are buffered on device!
        double * buf_d;
        double * buf_h;
};


// Entry point
__global__ void generateOnDevice_kernel(ConusUniformGPU * ud_device);

void generateOnDevice(ConusUniformGPU * ud_host, ConusUniformGPU * ud_device);

ConusUniformGPU * sendToDevice(ConusUniformGPU * ud_host);

#endif

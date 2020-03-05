#include <iostream>
#include <Random123/philox.h>
#include <Random123/ReinterpretCtr.hpp>

#include "example_seeds.h"
#include "util_cuda.h"

using namespace r123;
using namespace std;

typedef long unsigned uint64_t;
#define THREADS_PER_BLOCK 32

template<typename T>
__global__ void
uniform_ct_gpu(unsigned useed,
               T* arr) {

    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    typedef Philox4x64 G;
    G rng;
    G::key_type k = {{tid, useed}};
    G::ctr_type c = {{}};

    union {
        G::ctr_type c;
        int4 i;
    }u;
    c.incr();
    u.c = rng(c, k);

    if (std::is_same<T, int64_t>::value){
        arr[4*tid]   = u.i.x;
        arr[4*tid+1] = u.i.y;
        arr[4*tid+2] = u.i.z;
        arr[4*tid+3] = u.i.w;
    }else{
        arr[4*tid]   = ((double)((uint64_t)u.i.x))/((double)UINT_MAX);
        arr[4*tid+1] = ((double)((uint64_t)u.i.y))/((double)UINT_MAX);
        arr[4*tid+2] = ((double)((uint64_t)u.i.z))/((double)UINT_MAX);
        arr[4*tid+3] = ((double)((uint64_t)u.i.w))/((double)UINT_MAX);
    }
}

extern unsigned getUseed();

template<typename T>
T *
__generateRandomsGPU_onD(unsigned long N) {
    assert(N%4 ==0);
    unsigned useed = getUseed();
    T * randomNumbers_d;

    size_t rn_size = N * sizeof(T);

    CHECKCALL(cudaMalloc(& randomNumbers_d, rn_size));

    unsigned threads_per_block = THREADS_PER_BLOCK;
    assert(N%THREADS_PER_BLOCK == 0);
    unsigned blocks_per_grid   = N / threads_per_block;
    cerr << threads_per_block << endl;
    cerr << blocks_per_grid   << endl;
    cerr << N                 << endl;

    if (std::is_same<T, int64_t>::value)
        cerr << "VALUE is int64_t "<<endl;
    else
        cerr << "VALUE is double"<<endl;

    uniform_ct_gpu<<<blocks_per_grid, threads_per_block>>>(
        useed, randomNumbers_d);

    return randomNumbers_d;
}

template<typename T>
T *
__generateRandomsGPU(unsigned long N) {

    T * randomNumbers_d, * randomNumbers_h;

    randomNumbers_h = (T *) malloc(N*sizeof(T));

    randomNumbers_d = __generateRandomsGPU_onD<T>(N);
    CHECKCALL(cudaMemcpy(randomNumbers_h, randomNumbers_d,
                N * sizeof(T),
                cudaMemcpyDeviceToHost));

    CHECKCALL(cudaFree(randomNumbers_d));
    return randomNumbers_h;
}


void deleteRandomsGPU(double * arr){
    CHECKCALL(cudaFree(arr));
}

double *
generateRandomsGPUd(unsigned long N){
    return __generateRandomsGPU<double>(N);
}

int64_t *
generateRandomsGPUi(unsigned long N){
    return __generateRandomsGPU<int64_t>(N);
}


#include "Random.h"


class ConusUniformGPU : public galsim::BaseDeviate {

    public:

        ConusUniformGPU(long lseed, int N):
            galsim::BaseDeviate(lseed), buf_len(N), buf_ptr(N) {};
// NOTE: initialize buf_ptr to N so that we're calling fill_buff on the first
// time generate1() is called


        double generate1() {
            buf_ptr++;
            if (buf_ptr < buf_len) return buf_d[buf_ptr];

            cudaFree(buf_d);
            fill_buff();

            // Neet to try again after buffer has been filled. This definitely
            // looks unsafe on device. TODO: fix
            return generate1();
        };

    private:
        int buf_len;
        int buf_ptr;

        // NOTE: random numbers are buffered on device!
        double * buf_d;

        void fill_buff(){
            buf_d = __generateRandomsGPU_onD<double>(buf_len);
        };
};

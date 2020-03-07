#include <Random123/threefry.h>
#include <climits>

#include <omp.h>

#include "conus_cpu.hpp"
#include "conus.h"

using namespace r123;
using namespace std;

// Initializes the RNG on device and generates 4 random int64_t
void ConusUniformCPU::fill_buff()
{
    int ndiv4 = buf_len /4;
#pragma omp parallel for
    for (unsigned n_rng = 0; n_rng < ndiv4; n_rng++) {
        G rng;
        G::key_type k = {{n_rng, ulseed}};
        G::ctr_type c = {{}};

        union {
            G::ctr_type c;
            long4cpu i;
        }u;
        c.incr();
        u.c = rng(c, k);

        rn_array[4*n_rng] = ((double) ((uint64_t) u.i.x)) / ULONG_MAX;
        rn_array[4*n_rng+1] = ((double) ((uint64_t) u.i.y)) / ULONG_MAX;
        rn_array[4*n_rng+2] = ((double) ((uint64_t) u.i.z)) / ULONG_MAX;
        rn_array[4*n_rng+3] = ((double) ((uint64_t) u.i.w)) / ULONG_MAX;
    }
    //_p = -1;
}

//TODO: reuse lseed from BaseDeviate instead of having own private var?
ConusUniformCPU::ConusUniformCPU(long lseed, int N):
    galsim::BaseDeviate(lseed), buf_len(N), _p(0), ulseed(lseed)
{
    cerr << "CPU seed: " << lseed << endl;
    // Because Random123 generates 4 by 4
    assert(N%4 ==0);
    rn_array = (double*)malloc(sizeof(double)*N);
    fill_buff();
}

double ConusUniformCPU::get1()
{
    if (_p >= buf_len){
        //erase old content:TODO change the seed!
        fill_buff();
        _p = 0;
    }
    return rn_array[_p++];
}

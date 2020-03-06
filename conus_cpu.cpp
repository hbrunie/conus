#include <Random123/philox.h>

#include <omp.h>

#include "conus.hpp"
#include "conus_cpu.hpp"

using namespace r123;
using namespace std;

typedef struct int4cpu{
    long x;
    long y;
    long z;
    long w;
}long4cpu;

// Initializes the RNG on device and generates 4 random int64_t
void uniform_ct_cpu(int n, unsigned long ulseed,double * arr)
{
    assert(n%4 == 0);
    int ndiv4 = n /4;
#pragma omp parallel for
    for (unsigned n_rng = 0; n_rng < ndiv4; n_rng++) {
        typedef Threefry4x64 G;
        G rng;
        G::key_type k = {{n_rng, ulseed}};
        G::ctr_type c = {{}};

        union {
            G::ctr_type c;
            long4cpu i;
        }u;
        c.incr();
        u.c = rng(c, k);

        arr[4*n_rng] = ((double) ((uint64_t) u.i.x)) / ULONG_MAX;
        arr[4*n_rng+1] = ((double) ((uint64_t) u.i.y)) / ULONG_MAX;
        arr[4*n_rng+2] = ((double) ((uint64_t) u.i.z)) / ULONG_MAX;
        arr[4*n_rng+3] = ((double) ((uint64_t) u.i.w)) / ULONG_MAX;
    }
}

double * generateRandomsCPU(unsigned long N)
{
    double * randomNumbers = (double*) malloc(sizeof(double)*N);;
    unsigned long ulseed = getULseed();
    uniform_ct_cpu(N, ulseed, randomNumbers);
    return randomNumbers;
}

ConusUniformCPU::ConusUniformCPU(long lseed, int N):
    galsim::BaseDeviate(lseed), buf_len(N), buf_ptr(N) {}
// NOdoubleE: initialize buf_ptr to N so that we're calling fill_buff
// on the first time generate1() is called

void ConusUniformCPU::fill_buff()
{
    buf = std::unique_ptr<double>(generateRandomsCPU(buf_len));
    buf_ptr = -1;
}

double ConusUniformCPU::generate1()
{
    buf_ptr++;
    if (buf_ptr < buf_len) return buf.get()[buf_ptr];
    fill_buff();
    // Need to try again after buffer has been filled
    // (I know this pattern is unsafe)
    return generate1();
}

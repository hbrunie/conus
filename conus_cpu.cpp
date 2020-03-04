#include <iostream>
#include <Random123/philox.h>
#include <Random123/ReinterpretCtr.hpp>

#include "example_seeds.h"

using namespace r123;
using namespace std;
// need to be defined to use Random123
int debug = 0;
const char *progname;

void conusInit(){
    progname = "Prog using Conus";
}

void conusFinalize(){
}

typedef struct int4cpu{
    int x;
    int y;
    int z;
    int w;
}int4cpu;

// Initializes the RNG on device and generates 4 random int64_t
template<typename T>
void uniform_ct_cpu(
            int n, unsigned useed,
            T * arr
        ) {
    assert(n%4 == 0);
    int ndiv4 = n /4;
    for (unsigned n_rng = 0; n_rng < ndiv4; n_rng++) {
        typedef Philox4x64 G;
        G rng;
        G::key_type k = {{n_rng, useed}};
        G::ctr_type c = {{}};

        union {
            G::ctr_type c;
            int4cpu i;
        }u;
        c.incr();
        u.c = rng(c, k);

        arr[4*n_rng] = u.i.x;
        arr[4*n_rng+1] = u.i.y;
        arr[4*n_rng+2] = u.i.z;
        arr[4*n_rng+3] = u.i.w;
    }

}

unsigned getUseed(){
    unsigned seed = 0;//example_seed_u32(EXAMPLE_SEED9_U32);
    return 0;
}

template<typename T>
T * generateRandomsCPU(unsigned long N){
    T * randomNumbers = (T*) malloc(sizeof(T)*N);;
    unsigned useed = getUseed();

    uniform_ct_cpu<T>(N, useed, randomNumbers);
    //cerr << " N " << N << endl;
    //cerr << " size " << size << endl;
    if (std::is_same<T, int64_t>::value){
        return randomNumbers;
    }else{
        for(unsigned i=0; i<N; i++){
            randomNumbers[i] =
                ((T) ((uint64_t) randomNumbers[i])) / UINT_MAX;
        }
    }
    return randomNumbers;
}

void deleteRandomsCPU(double * arr){
    free(arr);
}

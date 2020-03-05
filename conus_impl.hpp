
using namespace r123;
using namespace std;

typedef struct int4cpu{
    long x;
    long y;
    long z;
    long w;
}long4cpu;

// Initializes the RNG on device and generates 4 random int64_t
template<typename T>
void uniform_ct_cpu(
            int n, unsigned long ulseed,
            T * arr
        ) {
    assert(n%4 == 0);
    int ndiv4 = n /4;
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

        arr[4*n_rng] = u.i.x;
        arr[4*n_rng+1] = u.i.y;
        arr[4*n_rng+2] = u.i.z;
        arr[4*n_rng+3] = u.i.w;
    }

}

template<typename T>
T * generateRandomsCPU(unsigned long N){
    T * randomNumbers = (T*) malloc(sizeof(T)*N);;
    unsigned long ulseed = getULseed();

    uniform_ct_cpu<T>(N, ulseed, randomNumbers);
    //cerr << " N " << N << endl;
    //cerr << " size " << size << endl;
    if (std::is_same<T, int64_t>::value){
        return randomNumbers;
    }else{
        for(unsigned i=0; i<N; i++){
            randomNumbers[i] =
                ((T) ((uint64_t) randomNumbers[i])) / ULONG_MAX;
        }
    }
    return randomNumbers;
}

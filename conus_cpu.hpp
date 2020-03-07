#ifndef __CONUS_RANDOM_HPP
#define __CONUS_RANDOM_HPP

#include "Random.h"

typedef struct int4cpu{
    long x;
    long y;
    long z;
    long w;
}long4cpu;

class ConusUniformCPU : public galsim::BaseDeviate {
    public:
        ConusUniformCPU(long lseed, int N);
        ~ConusUniformCPU(){free(rn_array);};
        double get1();

    private:
        int buf_len;
        double * rn_array;
        int _p;
        unsigned long ulseed;
        void fill_buff();
};
#endif

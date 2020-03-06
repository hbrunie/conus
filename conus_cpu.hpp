#ifndef __CONUS_RANDOM_HPP
#define __CONUS_RANDOM_HPP



#include "Random.h"


class ConusUniformCPU : public galsim::BaseDeviate {

    public:

        ConusUniformCPU(long lseed, int N);

        double generate1();

    private:
        int buf_len;
        int buf_ptr;

        std::unique_ptr<double> buf;

        void fill_buff();
};



#endif

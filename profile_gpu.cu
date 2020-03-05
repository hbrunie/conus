#include <iostream>


#include "conus.hpp"
#include "conus_random.hpp"
#include "conus_gpu.h"


using namespace r123;


int main(int argc, char ** argv) {

    long seed = 0;
    long buff_size = 1024;

    ConusUniformCPU ud_cpu(seed, buff_size);
    
    ConusUniformGPU ud_host(seed, buff_size, buff_size);
    // cuda mallocs (and "plain" mallocs) all the internal arrays
    ud_host.initialize();
    // now ud_dev points to a dev copy
    ConusUniformGPU * ud_dev = sendToDevice(& ud_host);
    // now its internal `buf_d` is filled with random numbers
    generateOnDevice(& ud_host, ud_dev);
    //copies the buf_d to buf_h array on the host <-- my design should hold in theory
    ud_host.copyToHost();
}

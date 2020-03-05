#include <iostream>


#include "conus.hpp"
#include "conus_random.hpp"
#include "conus_gpu.h"


using namespace r123;


int main(int argc, char ** argv) {

    long seed = 0;
    long buff_size = 1024;
    // UniformDeviate ud(seed);
    // std::cout << "Here is a random number: " << ud() << std::endl;

    ConusUniformCPU ud(seed, buff_size);
    std::cout << "Here are some random numbers: " << std::endl;
    for (int i=0; i<100; i++)
        std::cout << ud() << std::endl;


    ConusUniformGPU ud_host(seed, buff_size);
    // cuda mallocs (and "plain" mallocs) all the internal arrays
    ud_host.initialize();
}

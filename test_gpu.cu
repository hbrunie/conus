#include <iostream>

#include "conus_cpu.hpp"
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

    std::cout << "Here are some random numbers: " << std::endl
              << "I \t CPU \t\t GPU" << std::endl;
    for (int i=0; i<100; i++)
        std::cout << i
                  << "\t" << ud_cpu.get1()
                  << "\t" << ud_host()
                  << std::endl;
}

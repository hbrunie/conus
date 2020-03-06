#include <iostream>
#include "nvToolsExt.h"


#include "conus.hpp"
#include "conus_random.hpp"
#include "conus_gpu.h"


using namespace r123;


int main(int argc, char ** argv) {

    if(argc<2){
        std::cout <<
            "Please enter buf size."
            <<std::endl;
        exit(0);
    }
    long seed = 0;
    long buff_size = atol(argv[1]);

    nvtxRangePushA("ud_cpu constructor");
    ConusUniformCPU ud_cpu(seed, buff_size);
    nvtxRangePop();
    nvtxRangePushA("ud_cpu generate1");
    ud_cpu.generate1();
    nvtxRangePop();
    nvtxRangePushA("ud_host constructor");
    ConusUniformGPU ud_host(seed, buff_size);
    nvtxRangePop();
    // cuda mallocs (and "plain" mallocs) all the internal arrays
    nvtxRangePushA("ud_host initialize");
    ud_host.initialize();
    nvtxRangePop();
    // now ud_dev points to a dev copy
    nvtxRangePushA("ud_dec sendToDevice(Host)");
    ConusUniformGPU * ud_dev = sendToDevice(& ud_host);
    nvtxRangePop();
    // now its internal `buf_d` is filled with random numbers
    nvtxRangePushA("Generate On Device");
    generateOnDevice(& ud_host, ud_dev);
    nvtxRangePop();
    //copies the buf_d to buf_h array on the host <-- my design should hold in theory
    nvtxRangePushA("copyToHost");
    ud_host.copyToHost();
    nvtxRangePop();

    std::cout << "Here are some random numbers: " << std::endl
              << "I \t CPU \t\t GPU" << std::endl;
    for (int i=0; i<10; i++)
        std::cout << i
                  << "\t" << ud_cpu()
                  << "\t" << ud_host()
                  << std::endl;

}
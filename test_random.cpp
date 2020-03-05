#include <cmath>
#include <iomanip>
#include <iostream>
#include "conus_random.hpp"
#include "conus.hpp"

// #include "Random.h"


using namespace galsim;

int main(int argc, char ** argv) {

    long seed = 0;
    long buff_size = 1024;
    // UniformDeviate ud(seed);
    // std::cout << "Here is a random number: " << ud() << std::endl;

    ConusUniform ud(seed, buff_size);
    std::cout << "Here are some random numbers: " << std::endl;
    for (int i=0; i<100; i++)
        std::cout << ud() << std::endl;
}

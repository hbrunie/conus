#include <iostream>
// #include "Random.h"
#include "conus_random.hpp"


using namespace galsim;

int main(int argc, char ** argv) {

    long seed = 0;
    long buff_size = 1024;
    // UniformDeviate ud(seed);
    // std::cout << "Here is a random number: " << ud() << std::endl;

    ConusUniform ud(seed, buff_size);
    std::cout << "Here is a random number: " << ud() << std::endl;
}

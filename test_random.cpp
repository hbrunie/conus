#include <iostream>
#include <Random.h>


using namespace galsim;

int main(int argc, char ** argv) {

    long seed = 0;
    UniformDeviate ud = UniformDeviate(seed);

    std::cout << "Here is a random number: " << ud() << std::endl;
}

#include "conus_random.hpp"

#include "Random123/philox.h"
#include "conus.hpp"


ConusUniformCPU::ConusUniformCPU(long lseed, int N):
    galsim::BaseDeviate(lseed), buf_len(N), buf_ptr(N) {}
// NOTE: initialize buf_ptr to N so that we're calling fill_buff on the first
// time generate1() is called

void ConusUniformCPU::fill_buff() {

    buf = std::unique_ptr<double>(
                generateRandomsCPU<double>(buf_len)
            );

    buf_ptr = -1;
}


double ConusUniformCPU::generate1() {

    buf_ptr++;
    if (buf_ptr < buf_len) return buf.get()[buf_ptr];

    fill_buff();

    // Need to try again after buffer has been filled (I know this pattern is
    // unsafe)
    return generate1();
}

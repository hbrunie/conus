#ifndef __CONUS_HPP
#define __CONUS_HPP

#include <climits>
#include <Random123/threefry.h>
#include <Random123/ReinterpretCtr.hpp>

typedef long int64_t;
// Generate N random numbers (double or single)
// Either on GPU or CPU using random123 library
// randomNumbers array is allocated on HOST
// directly by the method
template<typename T>
T* generateRandomsCPU(unsigned long N);

// randomNumbers array must be allocated on device
// Before calling the method
// TODO: do device allocation inside method

double*
generateRandomsGPUd(unsigned long N);

int64_t*
generateRandomsGPUi(unsigned long N);

void conusInit();
void conusFinalize();
unsigned long getULseed();
void deleteRandomsCPU(double * arr);


#include "conus_impl.hpp"

//void
//conusInit();
//
//void
//conusFinalize();
//void
//deleteRandomsCPU(double *);
//void
//deleteRandomsGPU(double *);

#endif

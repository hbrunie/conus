#ifndef __CONUS_HPP
#define __CONUS_HPP

#include <climits>
#include <Random123/threefry.h>
#include <Random123/ReinterpretCtr.hpp>

typedef long int64_t;
template<typename T>
T* generateRandomsCPU(unsigned long N);

void conusInit();
unsigned long getULseed();
void deleteRandomsCPU(double * arr);
#endif

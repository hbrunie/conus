#include <iostream>
#include "example_seeds.h"



// need to be defined to use Random123
int debug = 0;
const char *progname;


void conusInit(){
    progname = "Prog using Conus";
}

void conusFinalize(){
}



unsigned getUseed(){
    unsigned seed = 0;//example_seed_u32(EXAMPLE_SEED9_U32);
    return 0;
}



void deleteRandomsCPU(double * arr){
    free(arr);
}

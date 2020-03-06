A CUDA + batoid + GalSim wrapper for:
https://www.deshawresearch.com/resources_random123.html

This coordinates the RNGs over all threads to generate random numbers
that are consistent between host and device.

## Testing

Before any push: 

```bash
make cleanall 
make
./testAll.sh #(on Cori gpu)
```

## TODO
1. [ ] clean up code
2. [ ] compare new perf + test case with generation on GPU and memCpy
       DeviceToHost.
    1. [ ] debug `n_streams` < `buf_size`


## Profiling using NSIGHT:

1. Run:
```bash
module load cuda
srun nsys profile ./test_gpu_nvtx.ex BUFSIZE
```

2. Open it with nsight-sys (on NoMachine for example)


## Why the name?

https://en.wikipedia.org/wiki/Conus_textile

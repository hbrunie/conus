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

1. [ ] Clean up code (use Batoid style)
    1. [ ] Add cmake
    2. [ ] Test with LLVM
2. [ ] Compare new perf + test case with generation on GPU and memCpy
       DeviceToHost.
    1. [ ] Debug `n_streams` < `buf_size`
3. [ ] Pass GalSim random tests: GalSim/tests/test\_random.py
    1. [ ] make a shared lib of Conus (pybind)
    2. [ ] call conus from python tests.

### Next steps:

Work with josh to include conus into batoid
1. [ ] Figure out how to reproduce the virtual function mechanism cf:
       https://gist.github.com/jmeyers314/986ac7670b356eed32f2fecf2b55aa18
2. [ ] Make the result not depend on number of threads: i.e. work out
       a “unique” ID for each photon

## Profiling using NSIGHT:

1. Run:
```bash
module load cuda
srun nsys profile ./test_gpu_nvtx.ex BUFSIZE
```

2. Open it with nsight-sys (on NoMachine for example)


## Why the name?

https://en.wikipedia.org/wiki/Conus_textile

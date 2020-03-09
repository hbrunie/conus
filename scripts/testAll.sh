#!/usr/bin/env bash


for f in test_cpu.ex test_gpu.ex test_gpu_nvtx.ex profile_gpu.ex
do
    srun ./$f 1024
done

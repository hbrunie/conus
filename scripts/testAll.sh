#!/usr/bin/env bash

CURDIR=$(dirname ${BASH_SOURCE[0]})
for f in test_cpu.ex test_gpu.ex test_gpu_nvtx.ex profile_gpu.ex
do
  srun ${CURDIR}/../$f 1024
done

#!/usr/bin/env bash

CURDIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
for f in test_cpu.ex test_gpu.ex test_gpu_nvtx.ex profile_gpu.ex
do
  srun ${CURDIR}/../$f 1024
done

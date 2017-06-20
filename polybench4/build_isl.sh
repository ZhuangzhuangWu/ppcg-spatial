#!/bin/bash
./build.sh \
    $1 "--target=c --openmp --tile --isl-schedule-max-coefficient=4 --isl-schedule-max-constant-term=10 --no-isl-schedule-parametric -DPOLYBENCH_USE_C99_PROTO"

#!/bin/bash
./parallel_build.sh \
    orig "" \
    default "--target=c --openmp --tile --spatial-model=none" \
    default.posttile "--target=c --openmp --spatial-model=none --tile --posttile-reorder=pluto" \
    spatial "--target=c --openmp --tile --isl-schedule-single-outer-coincidence --no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-spatial-fusion --isl-schedule-outer-coincidence --isl-schedule-max-coefficient=4 --isl-schedule-max-constant-term=10 --no-isl-schedule-parametric -DPOLYBENCH_USE_C99_PROTO" \
    spatial.posttile "--target=c --openmp --tile --isl-schedule-single-outer-coincidence --no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-spatial-fusion --isl-schedule-outer-coincidence --isl-schedule-max-coefficient=4 --isl-schedule-max-constant-term=10 --no-isl-schedule-parametric -DPOLYBENCH_USE_C99_PROTO --posttile-reorder=spatial"


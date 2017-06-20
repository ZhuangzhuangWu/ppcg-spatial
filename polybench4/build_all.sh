#!/bin/bash
./build.sh \
    orig "" \
    pluto "" \
    spatial "--target=c --openmp --tile --isl-schedule-single-outer-coincidence --no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-spatial-fusion  --isl-schedule-max-coefficient=4 --isl-schedule-max-constant-term=10 --no-isl-schedule-parametric -DPOLYBENCH_USE_C99_PROTO  --isl-schedule-avoid-inner-coincidence  --wavefront=single --isl-schedule-outer-coincidence --no-isl-schedule-force-outer-coincidence" \
    posttile "--target=c --openmp --tile --isl-schedule-single-outer-coincidence --no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-spatial-fusion --isl-schedule-max-coefficient=4 --isl-schedule-max-constant-term=10 --no-isl-schedule-parametric -DPOLYBENCH_USE_C99_PROTO --isl-schedule-avoid-inner-coincidence --wavefront=single --isl-schedule-outer-coincidence --no-isl-schedule-force-outer-coincidence --posttile-reorder=pluto"

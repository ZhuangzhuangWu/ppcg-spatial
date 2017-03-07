#!/bin/bash
./build.sh \
    pluto "pluto" \
    ppcg "ppcg" \
    default "--openmp" \
    outer "--isl-schedule-outer-coincidence --openmp" \
    outer.single "--isl-schedule-single-outer-coincidence --isl-schedule-outer-coincidence --openmp" \
    single "--isl-schedule-single-outer-coincidence --openmp" \
    typedfuse "--no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --openmp" \
    typedfuse.outer "--no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-outer-coincidence --openmp" \
    typedfuse.single "--no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-single-outer-coincidence" \
    typedfuse.outer.single "--no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-single-outer-coincidence --isl-schedule-outer-coincidence"


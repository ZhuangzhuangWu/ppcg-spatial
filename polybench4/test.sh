#!/bin/bash
for i in `seq 1 5`; do
  ./run.sh orig pluto spatial spatial.posttile > ficus_0612_0050_${i}.txt;
done


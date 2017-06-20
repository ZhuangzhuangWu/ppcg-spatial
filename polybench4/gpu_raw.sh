#!/bin/sh
(for i in `cat benchmark_list`; do d=`dirname $i`; bn=`basename $i`; b=${bn/.c/}; echo $b; cd $d; make clean; make ${b}_cuda.exe ${b}_endsgrp_cuda.exe; diff -q ${b}_kernel.cu ${b}_endsgrp_kernel.cu; df=$?; if [ $df -ne 0 ]; then make run; else echo "$b same"; fi; cd -; done) > ~/res.txt 2>&1

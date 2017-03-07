#!/bin/bash

prefixes=$@
left=""

for i in $prefixes; do
  left="$left $i "
  for j in $prefixes; do
    if [[ $left = *" $j "* ]]; then
      continue;
    fi;
    for b in `cat utilities/benchmark_list`; do
      dir=`dirname $b`;
      fn=`basename $b`;
      fnc=${fn/.c/};
      oldpwd=`pwd`;
      cd "$dir"

      diff ${fnc}.${i}.c ${fnc}.${j}.c >/dev/null;
#      diff ${fnc}.${i}.out ${fnc}.${j}.out >/dev/null;
      dr=$?

      if [ "z$FULLNAMES" = "z1" ]; then
        fname=$b
      else
        fname=$fnc
      fi

      if [ "z$DIFF_ONLY" != "z1" -o $dr -ne 0 ]; then
        if [ "z$NODECORATE" != "z1" ]; then
          if [ $dr -eq 0 ]; then
            drtext="[\033[32m SAME\033[0m ]";
          else
            drtext="[\033[31m DIFF\033[0m ]";
          fi;
          echo -e "$drtext $fname $i $j";
        else
          echo "$fname $i $j $dr"
        fi;
      fi

      cd "$oldpwd"
    done;
  done;
done

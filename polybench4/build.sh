#!/bin/bash

even=`expr $# % 2`

if [ $even -ne 0 ]; then
  echo "Usage: suffix_1 \"flags\" [suffix_2 \"flags\" ...]";
fi

if [ $# -eq 0 ]; then
  echo "Usage: suffix_1 \"flags\" [suffix_2 \"flags\" ...]";
fi

step="one"
for i in "$@"; do
  if [ $step = "one" ]; then
    step="two"
    suffix="$i"
  elif [ $step = "two" ]; then
    step="one"
    flags="$i"

    for b in `cat utilities/benchmark_list`; do
      dir=`dirname $b`;
      fn=`basename $b`;
      fnc=${fn/.c/};
      oldpwd=`pwd`;
      cd "$dir"
      cat > Makefile <<EOF
SRC := $fnc
include ${oldpwd}/common.mk
EOF

      if [ "z$NODECORATE" != "z1" ]; then
        echo -n "[      ] $suffix $fnc"
      fi

      if [ "z$FULLNAMES" = "z1" ]; then
        fname=$b
      else
        fname=$fnc
      fi

      if [ $suffix = "ppcg" ]; then
        make ppcg_basic 2>/dev/null >/dev/null;
      elif [ $suffix = "pluto" ]; then
        make pluto 2>/dev/null >/dev/null;
      elif [ $suffix = "orig" ]; then
        cp $fnc.c $fnc.orig.c > /dev/null;
      elif [ $suffix = "cuda" ]; then
        PPCG_FLAGS="$flags" make spat_cuda 2>/dev/null >/dev/null
      else
        PPCG_FLAGS="$flags -o ${fnc}.${suffix}.c" make spat_cuda 2>/dev/null >/dev/null
      fi
      success=$?

      if [ "z$NODECORATE" != "z1" ]; then
        if [ $success -eq 0 ]; then
          succes_text="[\033[32m OK  \033[0m ]"
        else
          succes_text="[\033[41m FAIL\033[0m ]"
        fi
        echo -e "\r$succes_text $suffix $fname"
      else
        echo "$suffix $fname $success"
      fi
      cd "$oldpwd"
    done
  fi
done


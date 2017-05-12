#!/bin/bash

function usage {
    echo "Usage: $0 [--help] [--list <filename>] suffix_1 [suffix_2 ...]"
    echo ""
    echo "--help        show this information"
    echo "--list        run only the benchmarks listed in file instead of all"
    echo ""
    echo "if DIFF_ONLY=1, run only the benchmarks that are textually different"
    echo "     between the first suffix and the current one"
}

if [ $# -eq 0 ]; then
    usage;
    exit 1
fi

# parse arguments
expect_list_filename=0
suffixes=""
first_suffix=""
second_suffix=""
for i in "$@"; do
    if [ $expect_list_filename -eq 1 ]; then
        list_filename="$i";
        expect_list_filename=0
	continue;
    fi
    if [ $i = "--list" ]; then
        if [ "z$list_filename" = "z" ]; then
            expect_list_filename=1;
        else
            echo "'--list' repeated!";
            usage;
            exit 2;
        fi
    elif [ $i = "--help" ]; then
        if [ $# -ne 1 ]; then
            echo "'--help' given, ignoring other arguments";
        fi
        usage;
        exit 0;
    else
        suffixes="$suffixes $i";
        if [ "z$first_suffix" = "z" ]; then
            first_suffix="$i";
        elif [ "z$second_suffix" = "z" ]; then
            second_suffix="$i";
        fi
    fi
done
if [ $expect_list_filename -eq 1 ]; then
    echo "no file name given for '--list'";
    usage;
    exit 3;
fi

if [ "z$list_filename" = "z" ]; then
    list_filename="utilities/benchmark_list";
fi
if [ ! -f "$list_filename" ]; then
    echo "file $list_filename does not exist";
    exit 4;
fi
benchmarks=`cat $list_filename`

# $1 == list of all
# #2 == current step
function progress_bar {
    found=0
    complete=1
    echo -n -e "\r["
    for elem in "$1"; do
        if [ $found -eq 0 ]; then
            echo -n "#"
        else
            echo -n " "
            complete=0
        fi
        if [ "z$2" = "z$elem" ]; then
            found=1
        fi
    done
    echo -n "]"
    if [ $complete -eq 1 ]; then
        echo
    fi
}

## table output
echo -n "bench "
for suffix in $suffixes; do
    echo -n "status.${suffix} time.${suffix} "
#    if [ "z$first_suffix" == "z" ]; then
#        first_suffix="$suffix"
#    fi
done
echo

for bench in $benchmarks; do
    dname=`dirname $bench`
    fname=`basename $bench`
    fnc=${fname/.c/}
    oldpwd=`pwd`
    cd "$dname"
    if [ "z$CMP_TWO" = "z1" -a "z$second_suffix" != "z" ]; then
        diff -q "${fnc}.${first_suffix}.c" "${fnc}.${second_suffix}.c" > /dev/null
        dr=$?
        if [ $dr -eq 0 ]; then
            cd "$oldpwd";
            continue;
        fi
    fi
    echo -n "${fnc} "
    cat > Makefile <<EOF
SRC:=$fnc
include ${oldpwd}/common.mk
EOF
    for suffix in $suffixes; do
        dr=1
        if [ "$suffix" != "$first_suffix" ]; then
            diff "${fnc}.${first_suffix}.c" "${fnc}.${suffix}.c" >/dev/null
            dr=$?
        fi
        if [ "z$DIFF_ONLY" = "z1" -a $dr -eq 0 ]; then
            echo -n "1 NA"
        else
            make "${fnc}.${suffix}.exe" >/dev/null 2>/dev/null
            build_ok=$?
            if [ $build_ok -eq 0 ]; then
                "./${fnc}.${suffix}.exe" > /tmp/polybench_run 2>"${fnc}.${suffix}.out"
                t=`cat /tmp/polybench_run`
                dr=$[1 - $dr]
                echo -n "$dr $t"
            else
                echo -n "2 NA"
            fi
        fi
        echo -n " "
    done
    echo
    cd "$oldpwd"
done

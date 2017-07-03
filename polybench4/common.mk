ROOT=/home/ftynse/Projects/spatial/ppcg
BUILD=${ROOT}

PPCG=${BUILD}/ppcg
INC=${ROOT}/polybench4/utilities

PPCG_EXE=${BUILD}/.libs/ppcg
LD_PATH=${BUILD}/.libs/:${BUILD}/isl/.libs:${BUILD}/pet/.libs

PBO=${ROOT}/polybench4/utilities/polybench.o
PBC=${ROOT}/polybench4/utilities/polybench.c
POLYBENCH=${ROOT}/polybench4
POLYFLAGS = -DPOLYBENCH_TIME #-DLARGE_DATASET #-DPOLYBENCH_USE_C99_PROTO #-DPOLTBENCH_DUMP_ARRAYS 

TAR=cuda
SPAT=endsgrp
CC=gcc
CUDA_FLAGS=--no-isl-schedule-maximize-coincidence --isl-schedule-outer-typed-fusion --isl-schedule-spatial-fusion --tile --isl-schedule-max-coefficient=4 --isl-schedule-max-constant-term=10 --no-isl-schedule-parametric -DPOLYBENCH_USE_C99_PROTO --posttile-reorder=spatial

${SRC}_${SPAT}_host.cu: ${SRC}.c
	${PPCG} ${CUDA_FLAGS} --target=cuda --spatial-model=${SPAT} -I${POLYBENCH}/utilities ${SRC}.c --dump-schedule -o ${SRC}_${SPAT} ${POLYFLAGS}

${SRC}_${SPAT}_cuda.exe: ${SRC}_${SPAT}_host.cu
	nvcc -O3 ${SRC}_${SPAT}_host.cu ${SRC}_${SPAT}_kernel.cu   ${POLYBENCH}/utilities/polybench.cu  -I${POLYBENCH}/utilities ${POLYFLAGS} -o ${SRC}_${SPAT}_cuda.exe 

cuda_${SPAT}: ${SRC}_${SPAT}_cuda.exe
	nvprof ./${SRC}_${SPAT}_cuda.exe

${SRC}_host.cu: ${SRC}.c
	${PPCG} --target=cuda --spatial-model=${SPAT} -I${POLYBENCH}/utilities ${SRC}.c --dump-schedule ${POLYFLAGS}

${SRC}_cuda.exe: ${SRC}_host.cu
	nvcc -O3 ${SRC}_host.cu ${SRC}_kernel.cu   ${POLYBENCH}/utilities/polybench.cu  -I${POLYBENCH}/utilities ${POLYFLAGS} -o ${SRC}_cuda.exe 

cuda: ${SRC}_cuda.exe
	nvprof ./${SRC}_cuda.exe

run: ${SRC}_cuda.exe ${SRC}_${SPAT}_cuda.exe
	nvprof ./${SRC}_cuda.exe
	nvprof ./${SRC}_${SPAT}_cuda.exe

comprare: ${SRC}_cuda.exe ${SRC}_${SPAT}_cuda.exe
	./${SRC}_cuda.exe 2>out_orig
	./${SRC}_${SPAT}_cuda.exe 2>out_spat
	diff -q out_orig out_spat

prof: ${SRC}_cuda.exe ${SRC}_${SPAT}_cuda.exe
	echo ${SRC} >>${DUMP}
	echo "PPCG" >>${DUMP}
	nvprof ./${SRC}_cuda.exe |& sed -n "s/.*\(  [0-9]*.[0-9]*[mu]s  kernel[0-9]\).*/\1/p" >>${DUMP}
	echo "PPCG"${SPAT} >>${DUMP}
	nvprof ./${SRC}_${SPAT}_cuda.exe |& sed -n "s/.*\(  [0-9]*.[0-9]*[mu]s  kernel[0-9]\).*/\1/p" >>${DUMP}
orig: ${SRC}.c
	cp ${SRC}.c ${SRC}.orig.c

spat: ${SRC}.c
	${PPCG} -I${INC} ${SRC}.c --target=c  -o ${SRC}.spat.c --dump-schedule --openmp

spat_debug:
	LD_LIBRARY_PATH=${LD_PATH}:$$LD_LIBRARY_PATH gdb ${PPCG_EXE} -- "${PPCG_FLAGS}"

spat_valgrind:
	LD_LIBRARY_PATH=${LD_PATH}:$$LD_LIBRARY_PATH valgrind -- ${PPCG_EXE} --target=c ${SRC}.c -I${INC} -o ${SRC}.spat.c

spat_custom: ${SRC}.c
	${PPCG} -I${INC} ${SRC}.c --target=c $(PPCG_FLAGS)

spat_cuda: ${SRC}.c
	${PPCG} -I${INC} ${SRC}.c $(PPCG_FLAGS)

ppcg_basic:
	ppcg -I${INC} ${SRC}.c --target=c  -o ${SRC}.ppcg.c --dump-final-schedule --openmp

ppcg_basic_tiled: ${SRC}.c
	ppcg -I${INC} ${SRC}.c --target=c --tile -o ${SRC}.ppcg.tile.c --openmp

pluto: ${SRC}.c
	polycc ${SRC}.c -q -o ${SRC}.pluto.c --parallel --tile --noprevector

pluto_verbose: ${SRC}.c
	polycc ${SRC}.c --parallel --tile -o ${SRC}.pluto.c

pluto_tiled: ${SRC}.c
	polycc ${SRC}.c --tile --partlbtile --intratileopt -q -o ${SRC}.pluto.tile.c --parallel

pluto_seq: ${SRC}.c
	polycc --tile -o ${SRC}.pluto_seq.c $<

clean:
	rm -rf *.ppcg.c *.cu *.spatial.c *.only_cache_deps.c *.typedfuse.c *.outer.c *.single.c *.spat.c *.typedfuse.outer.c *.typedfuse.outer.single.c *.typedfuse.single.c *.cl *.hu *.default.c *.exe

pbo: $(PBO)

%.spat.c: %.c
	$(MAKE) spat

%.ppcg.c: %.c
	$(MAKE) ppcg_basic

%.spat.tile.c: %.c
	$(MAKE) spat_tiled

%.pluto.c: %.c
	$(MAKE) pluto

%.pluto.tile.c: %.c
	$(MAKE) pluto_tiled

%.exe: %.o $(PBO)
	$(CC) -O3 -fopenmp $< $(PBO) -lm -lrt -o $@

%.o: %.c
#	$(CC) -O3 -fopenmp -DPOLYBENCH_TIME  -DPOLYBENCH_DUMP_ARRAYS -DMINI_DATASET -I$(INC) -c $< -o $@
	$(CC) -O3 -fopenmp -DPOLYBENCH_TIME  -I$(INC) -c $< -o $@

$(PBO): $(PBC)
	$(CC) -O3 -fopenmp -DPOLYBENCH_TIME -c $< -o $@

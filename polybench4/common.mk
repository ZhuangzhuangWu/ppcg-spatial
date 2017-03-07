PPCG=/home/ftynse/ppcg/build2/ppcg
INC=/home/ftynse/ppcg/polybench4/utilities

PPCG_EXE=/home/ftynse/ppcg/build2/.libs/ppcg
LD_PATH=/home/ftynse/ppcg/build2/.libs/:/home/ftynse/ppcg/build2/isl/.libs:/home/ftynse/ppcg/build2/pet/.libs

PBO=/home/ftynse/ppcg/polybench4/utilities/polybench.o
PBC=/home/ftynse/ppcg/polybench4/utilities/polybench.c

CC=gcc

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
	polycc ${SRC}.c -q -o ${SRC}.pluto.c --parallel

pluto_tiled: ${SRC}.c
	polycc ${SRC}.c --tile --partlbtile --intratileopt -q -o ${SRC}.pluto.tile.c --parallel

clean:
	rm -rf *.ppcg.c *.cu *.spatial.c *.only_cache_deps.c

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
	$(CC) -O3 -fopenmp -DPOLYBENCH_TIME  -DPOLYBENCH_DUMP_ARRAYS -DMINI_DATASET -I$(INC) -c $< -o $@
#	$(CC) -O3 -fopenmp -DPOLYBENCH_TIME  -I$(INC) -c $< -o $@

$(PBO): $(PBC)
	$(CC) -O3 -fopenmp -DPOLYBENCH_TIME -c $< -o $@

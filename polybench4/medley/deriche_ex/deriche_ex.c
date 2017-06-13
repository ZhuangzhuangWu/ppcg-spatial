/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "deriche_ex.h"


/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h),
       DATA_TYPE POLYBENCH_1D(_e_xm1, W, w),
       DATA_TYPE POLYBENCH_1D(_e_ym1, W, w),
       DATA_TYPE POLYBENCH_1D(_e_ym2, W, w),
       DATA_TYPE POLYBENCH_1D(_e_yp1, W, w),
       DATA_TYPE POLYBENCH_1D(_e_yp2, W, w),
       DATA_TYPE POLYBENCH_1D(_e_xp1, W, w),
       DATA_TYPE POLYBENCH_1D(_e_xp2, W, w),
       DATA_TYPE POLYBENCH_1D(_e_tm11, H, h),
       DATA_TYPE POLYBENCH_1D(_e_ym11, H, h),
       DATA_TYPE POLYBENCH_1D(_e_ym21, H, h),
       DATA_TYPE POLYBENCH_1D(_e_tp11, H, h),
       DATA_TYPE POLYBENCH_1D(_e_tp21, H, h),
       DATA_TYPE POLYBENCH_1D(_e_yp11, H, h),
       DATA_TYPE POLYBENCH_1D(_e_yp21, H, h))
{
    int i,j;

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   k = (SCALAR_VAL(1.0)-EXP_FUN(-alpha))*(SCALAR_VAL(1.0)-EXP_FUN(-alpha))/(SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*alpha*EXP_FUN(-alpha)-EXP_FUN(SCALAR_VAL(2.0)*alpha));
   a1 = a5 = k;
   a2 = a6 = k*EXP_FUN(-alpha)*(alpha-SCALAR_VAL(1.0));
   a3 = a7 = k*EXP_FUN(-alpha)*(alpha+SCALAR_VAL(1.0));
   a4 = a8 = -k*EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   b1 =  POW_FUN(SCALAR_VAL(2.0),-alpha);
   b2 = -EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   c1 = c2 = 1;

    for (i=0; i<_PB_W; i++) {
        _e_ym1[i] = SCALAR_VAL(0.0);
        _e_ym2[i] = SCALAR_VAL(0.0);
        _e_xm1[i] = SCALAR_VAL(0.0);
        for (j=0; j<_PB_H; j++) {
            y1[i][j] = a1*imgIn[i][j] + a2*_e_xm1[i] + b1*_e_ym1[i] + b2*_e_ym2[i];
            _e_xm1[i] = imgIn[i][j];
            _e_ym2[i] = _e_ym1[i];
            _e_ym1[i] = y1[i][j];
        }
    }

    for (i=0; i<_PB_W; i++) {
        _e_yp1[i] = SCALAR_VAL(0.0);
        _e_yp2[i] = SCALAR_VAL(0.0);
        _e_xp1[i] = SCALAR_VAL(0.0);
        _e_xp2[i] = SCALAR_VAL(0.0);
        for (j=_PB_H-1; j>=0; j--) {
            y2[i][j] = a3*_e_xp1[i] + a4*_e_xp2[i] + b1*_e_yp1[i] + b2*_e_yp2[i];
            _e_xp2[i] = _e_xp1[i];
            _e_xp1[i] = imgIn[i][j];
            _e_yp2[i] = _e_yp1[i];
            _e_yp1[i] = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }

    for (j=0; j<_PB_H; j++) {
        _e_tm11[j] = SCALAR_VAL(0.0);
        _e_ym11[j] = SCALAR_VAL(0.0);
        _e_ym21[j] = SCALAR_VAL(0.0);
        for (i=0; i<_PB_W; i++) {
            y1[i][j] = a5*imgOut[i][j] + a6*_e_tm11[j] + b1*_e_ym11[j] + b2*_e_ym21[j];
            _e_tm11[j] = imgOut[i][j];
            _e_ym21[j] = _e_ym11[j];
            _e_ym11[j] = y1 [i][j];
        }
    }


    for (j=0; j<_PB_H; j++) {
        _e_tp11[j] = SCALAR_VAL(0.0);
        _e_tp21[j] = SCALAR_VAL(0.0);
        _e_yp11[j] = SCALAR_VAL(0.0);
        _e_yp21[j] = SCALAR_VAL(0.0);
        for (i=_PB_W-1; i>=0; i--) {
            y2[i][j] = a7*_e_tp11[j] + a8*_e_tp21[j] + b1*_e_yp11[j] + b2*_e_yp21[j];
            _e_tp21[j] = _e_tp11[j];
            _e_tp11[j] = imgOut[i][j];
            _e_yp21[j] = _e_yp11[j];
            _e_yp11[j] = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++)
        for (j=0; j<_PB_H; j++)
            imgOut[i][j] = c2*(y1[i][j] + y2[i][j]);

#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);

  POLYBENCH_1D_ARRAY_DECL(_e_xm1, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_ym1, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_ym2, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_yp1, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_yp2, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_xp1, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_xp2, DATA_TYPE, W, w);
  POLYBENCH_1D_ARRAY_DECL(_e_tm11, DATA_TYPE, H, h);
  POLYBENCH_1D_ARRAY_DECL(_e_ym11, DATA_TYPE, H, h);
  POLYBENCH_1D_ARRAY_DECL(_e_ym21, DATA_TYPE, H, h);
  POLYBENCH_1D_ARRAY_DECL(_e_tp11, DATA_TYPE, H, h);
  POLYBENCH_1D_ARRAY_DECL(_e_tp21, DATA_TYPE, H, h);
  POLYBENCH_1D_ARRAY_DECL(_e_yp11, DATA_TYPE, H, h);
  POLYBENCH_1D_ARRAY_DECL(_e_yp21, DATA_TYPE, H, h);


  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2),
          POLYBENCH_ARRAY(_e_xm1),
          POLYBENCH_ARRAY(_e_ym1),
          POLYBENCH_ARRAY(_e_ym2),
          POLYBENCH_ARRAY(_e_yp1),
          POLYBENCH_ARRAY(_e_yp2),
          POLYBENCH_ARRAY(_e_xp1),
          POLYBENCH_ARRAY(_e_xp2),
          POLYBENCH_ARRAY(_e_tm11),
          POLYBENCH_ARRAY(_e_ym11),
          POLYBENCH_ARRAY(_e_ym21),
          POLYBENCH_ARRAY(_e_tp11),
          POLYBENCH_ARRAY(_e_tp21),
          POLYBENCH_ARRAY(_e_yp11),
          POLYBENCH_ARRAY(_e_yp21));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

  return 0;
}

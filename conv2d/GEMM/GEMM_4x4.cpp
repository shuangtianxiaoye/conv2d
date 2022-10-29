#include <omp.h>
#include "parameters.h"

/* gemm_4x4 */

#define INPUTM(i,j) inputm[(i)*ldi+(j)]
#define WEIGHTM(i,j) weightm[(i)*ldw+(j)]
#define OUTPUTM(i,j) outputm[(i)*ldo+(j)]

void AddDot_4x4(float*, int, float*, int, float*, int, struct GEMM_SIZE);

void GEMM_4x4(float* inputm, int ldi,
	float* weightm, int ldw,
	float* outputm, int ldo,
	struct GEMM_SIZE gemm_size) {
	/*
	  inputm size: (m, k)
	  weightm size: (k, n)
	  outputm size: (m, n)
	*/

#pragma omp parallel for
	for (int i = 0; i < gemm_size.m; i += 4) {
		for (int j = 0; j < gemm_size.n; j += 4) {
			AddDot_4x4(&INPUTM(i, 0), ldi, &WEIGHTM(0, j), ldw,
				&OUTPUTM(i, j), ldo, gemm_size);
		}
	}

}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE


typedef union
{
	__m128 v;
	float f[4];
}vf;

void AddDot_4x4(float* inputm, int ldi,
	float* weightm, int ldw,
	float* outputm, int ldo,
	struct GEMM_SIZE gemm_size) {

	vf
		c_00_c_01_c_02_c_03_vreg,
		c_10_c_11_c_12_c_13_vreg,
		c_20_c_21_c_22_c_23_vreg,
		c_30_c_31_c_32_c_33_vreg,
		b_p0_b_p1_b_p2_b_p3_vreg,
		a_0p_vreg,
		a_1p_vreg,
		a_2p_vreg,
		a_3p_vreg;
	float
		* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr;

	a_0p_pntr = &INPUTM(0, 0);
	a_1p_pntr = &INPUTM(1, 0);
	a_2p_pntr = &INPUTM(2, 0);
	a_3p_pntr = &INPUTM(3, 0);

	c_00_c_01_c_02_c_03_vreg.v = _mm_setzero_ps();
	c_10_c_11_c_12_c_13_vreg.v = _mm_setzero_ps();
	c_20_c_21_c_22_c_23_vreg.v = _mm_setzero_ps();
	c_30_c_31_c_32_c_33_vreg.v = _mm_setzero_ps();

	for (int p = 0; p < gemm_size.k; p++) {

		b_p0_b_p1_b_p2_b_p3_vreg.v = _mm_load_ps(&WEIGHTM(p, 0));

		a_0p_vreg.v = _mm_load1_ps(a_0p_pntr++);
		a_1p_vreg.v = _mm_load1_ps(a_1p_pntr++);
		a_2p_vreg.v = _mm_load1_ps(a_2p_pntr++);
		a_3p_vreg.v = _mm_load1_ps(a_3p_pntr++);

		/* First cols and second cols*/
		c_00_c_01_c_02_c_03_vreg.v = _mm_add_ps(c_00_c_01_c_02_c_03_vreg.v,
			_mm_mul_ps(a_0p_vreg.v, b_p0_b_p1_b_p2_b_p3_vreg.v));
		c_10_c_11_c_12_c_13_vreg.v = _mm_add_ps(c_10_c_11_c_12_c_13_vreg.v,
			_mm_mul_ps(a_1p_vreg.v, b_p0_b_p1_b_p2_b_p3_vreg.v));

		/* Third cols and fouth cols*/
		c_20_c_21_c_22_c_23_vreg.v = _mm_add_ps(c_20_c_21_c_22_c_23_vreg.v,
			_mm_mul_ps(a_2p_vreg.v, b_p0_b_p1_b_p2_b_p3_vreg.v));

		c_30_c_31_c_32_c_33_vreg.v = _mm_add_ps(c_30_c_31_c_32_c_33_vreg.v,
			_mm_mul_ps(a_3p_vreg.v, b_p0_b_p1_b_p2_b_p3_vreg.v));

	}
	/*
	_mm_store_ps(&OUTPUTM(0, 0), c_00_c_01_c_02_c_03_vreg.v);
	_mm_store_ps(&OUTPUTM(1, 0), c_10_c_11_c_12_c_13_vreg.v);
	_mm_store_ps(&OUTPUTM(2, 0), c_20_c_21_c_22_c_23_vreg.v);
	_mm_store_ps(&OUTPUTM(3, 0), c_30_c_31_c_32_c_33_vreg.v);
	*/

	OUTPUTM(0, 0) += c_00_c_01_c_02_c_03_vreg.f[0];
	OUTPUTM(0, 1) += c_00_c_01_c_02_c_03_vreg.f[1];
	OUTPUTM(0, 2) += c_00_c_01_c_02_c_03_vreg.f[2];
	OUTPUTM(0, 3) += c_00_c_01_c_02_c_03_vreg.f[3];

	OUTPUTM(1, 0) += c_10_c_11_c_12_c_13_vreg.f[0];
	OUTPUTM(1, 1) += c_10_c_11_c_12_c_13_vreg.f[1];
	OUTPUTM(1, 2) += c_10_c_11_c_12_c_13_vreg.f[2];
	OUTPUTM(1, 3) += c_10_c_11_c_12_c_13_vreg.f[3];

	OUTPUTM(2, 0) += c_20_c_21_c_22_c_23_vreg.f[0];
	OUTPUTM(2, 1) += c_20_c_21_c_22_c_23_vreg.f[1];
	OUTPUTM(2, 2) += c_20_c_21_c_22_c_23_vreg.f[2];
	OUTPUTM(2, 3) += c_20_c_21_c_22_c_23_vreg.f[3];

	OUTPUTM(3, 0) += c_30_c_31_c_32_c_33_vreg.f[0];
	OUTPUTM(3, 1) += c_30_c_31_c_32_c_33_vreg.f[1];
	OUTPUTM(3, 2) += c_30_c_31_c_32_c_33_vreg.f[2];
	OUTPUTM(3, 3) += c_30_c_31_c_32_c_33_vreg.f[3];

}
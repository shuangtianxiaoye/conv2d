#include <omp.h>
#include "parameters.h"

/* gemm */

#define INPUTM(i,j) inputm[(i)*ldi+(j)]
#define WEIGHTM(i,j) weightm[(i)*ldw+(j)]
#define OUTPUTM(i,j) outputm[(i)*ldo+(j)]

void AddDot(float*, int, float*, int, float*, int, struct GEMM_SIZE);

void GEMM(float* inputm, int ldi,
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
			AddDot(&INPUTM(i, 0), ldi, &WEIGHTM(0, j), ldw,
				&OUTPUTM(i, j), ldo, gemm_size);
		}
	}

}

void AddDot(float* inputm, int ldi,
	float* weightm, int ldw,
	float* outputm, int ldo,
	struct GEMM_SIZE gemm_size) {

	float
		c_00_vreg, c_01_vreg, c_02_vreg, c_03_vreg,
		c_10_vreg, c_11_vreg, c_12_vreg, c_13_vreg,
		c_20_vreg, c_21_vreg, c_22_vreg, c_23_vreg,
		c_30_vreg, c_31_vreg, c_32_vreg, c_33_vreg;
	float
		b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg;
	float
		a_0p_vreg, a_1p_vreg, a_2p_vreg, a_3p_vreg;

	float
		* a_0p_pntr, * a_1p_pntr, * a_2p_pntr, * a_3p_pntr;

	a_0p_pntr = &INPUTM(0, 0);
	a_1p_pntr = &INPUTM(1, 0);
	a_2p_pntr = &INPUTM(2, 0);
	a_3p_pntr = &INPUTM(3, 0);

	c_00_vreg = 0;  c_01_vreg = 0;  c_02_vreg = 0;  c_03_vreg = 0;
	c_10_vreg = 0;  c_11_vreg = 0;  c_12_vreg = 0;  c_13_vreg = 0;
	c_20_vreg = 0;  c_21_vreg = 0;  c_22_vreg = 0;  c_23_vreg = 0;
	c_30_vreg = 0;  c_31_vreg = 0;  c_32_vreg = 0;  c_33_vreg = 0;

	for (int p = 0; p < gemm_size.k; p++) {

		b_p0_vreg = WEIGHTM(p, 0);
		b_p1_vreg = WEIGHTM(p, 1);
		b_p2_vreg = WEIGHTM(p, 2);
		b_p3_vreg = WEIGHTM(p, 3);

		a_0p_vreg = *a_0p_pntr++;
		a_1p_vreg = *a_1p_pntr++;
		a_2p_vreg = *a_2p_pntr++;
		a_3p_vreg = *a_3p_pntr++;

		/* First cols and second cols*/
		c_00_vreg += a_0p_vreg * b_p0_vreg;
		c_01_vreg += a_0p_vreg * b_p1_vreg;
		c_02_vreg += a_0p_vreg * b_p2_vreg;
		c_03_vreg += a_0p_vreg * b_p3_vreg;

		c_10_vreg += a_1p_vreg * b_p0_vreg;
		c_11_vreg += a_1p_vreg * b_p1_vreg;
		c_12_vreg += a_1p_vreg * b_p2_vreg;
		c_13_vreg += a_1p_vreg * b_p3_vreg;

		/* Third cols and fouth cols*/
		c_20_vreg += a_2p_vreg * b_p0_vreg;
		c_21_vreg += a_2p_vreg * b_p1_vreg;
		c_22_vreg += a_2p_vreg * b_p2_vreg;
		c_23_vreg += a_2p_vreg * b_p3_vreg;

		c_30_vreg += a_3p_vreg * b_p0_vreg;
		c_31_vreg += a_3p_vreg * b_p1_vreg;
		c_32_vreg += a_3p_vreg * b_p2_vreg;
		c_33_vreg += a_3p_vreg * b_p3_vreg;

	}
	/*
	_mm_store_ps(&OUTPUTM(0, 0), c_00_c_01_c_02_c_03_vreg.v);
	_mm_store_ps(&OUTPUTM(1, 0), c_10_c_11_c_12_c_13_vreg.v);
	_mm_store_ps(&OUTPUTM(2, 0), c_20_c_21_c_22_c_23_vreg.v);
	_mm_store_ps(&OUTPUTM(3, 0), c_30_c_31_c_32_c_33_vreg.v);
	*/

	OUTPUTM(0, 0) += c_00_vreg;
	OUTPUTM(0, 1) += c_01_vreg;
	OUTPUTM(0, 2) += c_02_vreg;
	OUTPUTM(0, 3) += c_03_vreg;

	OUTPUTM(1, 0) += c_10_vreg;
	OUTPUTM(1, 1) += c_11_vreg;
	OUTPUTM(1, 2) += c_12_vreg;
	OUTPUTM(1, 3) += c_13_vreg;

	OUTPUTM(2, 0) += c_20_vreg;
	OUTPUTM(2, 1) += c_21_vreg;
	OUTPUTM(2, 2) += c_22_vreg;
	OUTPUTM(2, 3) += c_23_vreg;

	OUTPUTM(3, 0) += c_30_vreg;
	OUTPUTM(3, 1) += c_31_vreg;
	OUTPUTM(3, 2) += c_32_vreg;
	OUTPUTM(3, 3) += c_33_vreg;


}
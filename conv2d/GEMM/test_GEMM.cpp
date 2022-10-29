#include <iostream>
#include <stdlib.h>

#include "parameters.h"

double dclock();
void random_input(float*, struct INPUT_SIZE);
void random_kernel(float*, struct KERNEL_SIZE);
void conv(float*, float*, float*, struct INPUT_SIZE, struct KERNEL_SIZE, struct OUTPUT_SIZE, int ,int);
void kernel2mat(float*, float*, struct KERNEL_SIZE);
void input2mat(float*, float*, struct INPUT_SIZE, struct KERNEL_SIZE, struct OUTPUT_SIZE);
void GEMM(float*, int, float*, int, float*, int, struct GEMM_SIZE);
void GEMM_4x4(float*, int, float*, int, float*, int, struct GEMM_SIZE);
void transpose(float*, float*, struct OUTPUT_SIZE);
int compare_result(float*, float*, struct OUTPUT_SIZE);

int main() {
	using std::cout;
	using std::endl;

	struct INPUT_SIZE in_size;
	struct KERNEL_SIZE w_size;
	struct OUTPUT_SIZE out_size;
	struct GEMM_SIZE gemm_size;

	int stride;
	int padding;

	float* inputc;
	float* kernelc;
	float* outputc;

	float* inputm;
	float* weightm;
	float* outputm;
	float* outputs;
	float* outputm_t;

	double dtime;

	for (int p = PFIRST; p <= PLAST; p += PINC) {

		/* Input size:  in_channel*in_h*in_w */
		in_size.ic = (IC == -1 ? 3 : IC);
		in_size.ih = (IH == -1 ? p : IH);
		in_size.iw = (IW == -1 ? p : IW);
		
		/* Kernel size:  out_channel*in_channel*kernel_h*kernel_w */
		w_size.oc = (OC == -1 ? 8 : OC);
		w_size.ic = (IC == -1 ? 3 : IC);
		w_size.kh = (KH == -1 ? 3 : KH);
		w_size.kw = (KW == -1 ? 3 : KW);
		
		/* Conv stride & padding */
		stride = (STRIDE == -1 ? 1 : STRIDE);
		padding = (PADDING == -1 ? 0 : PADDING);

		/* Output size:  out_channel*out_h*out_w */
		out_size.oc = w_size.oc;
		out_size.oh = (in_size.ih - w_size.kh + 1 + padding * 2) / stride;
		out_size.ow = (in_size.iw - w_size.kw + 1 + padding * 2) / stride;

		/* Allocate space for conv */
		inputc = (float*)malloc(in_size.ic * in_size.ih * in_size.iw * sizeof(float));
		kernelc = (float*)malloc(w_size.oc * w_size.ic * w_size.kh * w_size.kw * sizeof(float));
		outputc = (float*)malloc(out_size.oc * out_size.oh * out_size.ow * sizeof(float));

		/* Allocate space for gemm*/
		inputm = (float*)malloc((out_size.oh * out_size.ow) * (w_size.ic * w_size.kh * w_size.kw) * sizeof(float));
		weightm = (float*)malloc((w_size.ic * w_size.kh * w_size.kw) * out_size.oc * sizeof(float));
		outputm = (float*)calloc((out_size.oh * out_size.ow) * out_size.oc, sizeof(float));
		outputm_t = (float*)malloc(out_size.oc * out_size.oh * out_size.ow * sizeof(float));

		/* Set outputm 0*/
		//memset(outputm, 0, (out_size.ow * out_size.oh) * out_size.oc * sizeof(float));
		
		/* Generate random matrices inputc, kernelc */
		random_input(inputc, in_size);
		random_kernel(kernelc, w_size);
		cout << "***********"<< endl;
		cout << "input size: " << in_size.ic << "*" << in_size.ih << "*" << in_size.iw << endl;
		cout << "kernel size: " << w_size.oc << "*" << w_size.ic << "*" << w_size.kh << "*" << w_size.kw << endl;
		cout << "output size: " << out_size.oc << "*" << out_size.oh << "*" << out_size.ow << endl;

		/* Conv2d*/
		dtime = dclock();
		conv(inputc, kernelc, outputc, in_size, w_size, out_size, stride, padding);
		dtime = dclock() - dtime;
		cout << "conv time: " << (double)dtime << " s" << endl;

		/* im2col */
		kernel2mat(kernelc, weightm, w_size);

		dtime = dclock();
		input2mat(inputc, inputm, in_size, w_size, out_size);
		//dtime = dclock() - dtime;
		//cout << "im2col time: " << (double)dtime << " s" << endl;

		/* gemm */
		//dtime = dclock();
		gemm_size.m = (out_size.oh * out_size.ow);
		gemm_size.k = (w_size.ic * w_size.kh * w_size.kw);
		gemm_size.n = (w_size.oc);
		GEMM(inputm, gemm_size.k, weightm, gemm_size.n, outputm, gemm_size.n,
			gemm_size);
		//dtime = dclock() - dtime;
		//cout << "gemm time: " << (double)dtime << " s" << endl;
		//dtime = dclock();
		transpose(outputm, outputm_t, out_size);
		dtime = dclock() - dtime;
		cout << "gemm time: " << (double)dtime << " s" << endl;

		/* Compare */
		int err = 0;
		err = compare_result(outputc, outputm_t, out_size);
		cout << "num of error: " << err << endl;

		free(outputm);

		/* gemm with SSE */
		outputs = (float*)calloc((out_size.oh * out_size.ow) * out_size.oc, sizeof(float));

		kernel2mat(kernelc, weightm, w_size);

		dtime = dclock();
		input2mat(inputc, inputm, in_size, w_size, out_size);

		/* gemm */
		gemm_size.m = (out_size.oh * out_size.ow);
		gemm_size.k = (w_size.ic * w_size.kh * w_size.kw);
		gemm_size.n = (w_size.oc);
		GEMM_4x4(inputm, gemm_size.k, weightm, gemm_size.n, outputs, gemm_size.n,
			gemm_size);
		transpose(outputs, outputm_t, out_size);
		dtime = dclock() - dtime;
		cout << "gemm with SSE time: " << (double)dtime << " s" << endl;

		/* Compare */
		err = compare_result(outputc, outputm_t, out_size);
		cout << "num of error: " << err << endl;


		free(inputc);
		free(kernelc);
		free(outputc);
		free(inputm);
		free(weightm);
		free(outputs);
		free(outputm_t);

	}

	return 0;

}